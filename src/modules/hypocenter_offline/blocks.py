import torch
import torch.nn as nn
import math
from typing import Optional, Union, Iterable, List


class MLP(nn.Module):
    """
    Bloque MLP (Perceptrón Multicapa) flexible.

    Permite la creación de un MLP con un número arbitrario de capas ocultas,
    control sobre las funciones de activación y el dropout.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        use_activation: bool = True,
        activation_function_selection: str = "tanh",
        hidden_dims: Optional[Union[int, Iterable[int]]] = None,
        final_activation: bool = False,
        final_dropout: bool = False):
        """
        Args:
            input_dim (int): Dimensión de entrada.
            output_dim (int): Dimensión de salida.
            dropout (float): Tasa de dropout (aplicada después de la activación).
            use_activation (bool): Si se usan funciones de activación.
            activation_function_selection (str): Nombre de la activación (p.ej., "relu", "gelu").
            hidden_dims (int or list): Dimensiones de las capas ocultas. Si es None,
                                     es una sola capa lineal.
            final_activation (bool): Si se aplica activación a la capa de salida.
            final_dropout (bool): Si se aplica dropout a la capa de salida.
        """
        super().__init__()

        # --- Activación (mismas opciones que usabas) ---
        if activation_function_selection == "tanh":
            act = nn.Tanh()
        elif activation_function_selection == "relu":
            act = nn.ReLU()
        elif activation_function_selection == "sigmoid":
            act = nn.Sigmoid()
        elif activation_function_selection == "leaky_relu":
            act = nn.LeakyReLU()
        elif activation_function_selection == "elu":
            act = nn.ELU()
        elif activation_function_selection == "selu":
            act = nn.SELU()
        elif activation_function_selection == "gelu":
            act = nn.GELU()
        elif activation_function_selection == "swish":
            act = nn.SiLU()
        elif activation_function_selection == "mish":
            act = nn.Mish()
        elif activation_function_selection == "hard_swish":
            act = nn.Hardswish()
        else:
            raise ValueError(f"Activation function {activation_function_selection} not supported")

        # --- Normaliza hidden_dims a lista ---
        if hidden_dims is None:
            hiddens: List[int] = []
        elif isinstance(hidden_dims, int):
            hiddens = [hidden_dims]
        else:
            hiddens = list(hidden_dims)

        dims = [input_dim] + hiddens + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i+1]
            is_last = (i == len(dims) - 2)

            layers.append(nn.Linear(in_d, out_d))

            # Activación + Dropout en capas intermedias; en la final según flags
            if use_activation and (not is_last or final_activation):
                layers.append(act)
            if dropout and dropout > 0.0 and (not is_last or final_dropout):
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN_embedding(nn.Module):
    """
    Bloque de embedding basado en CNN 1D.

    Aplica una secuencia de capas Conv1d, GELU, Dropout y Pooling (Max o Avg)
    para extraer características de una secuencia.
    """
    def __init__(self, dropout: float = 0.0, 
            layers: list[dict[str, int]] = [{'input_dim': 1, 'out_channels': 400, 'kernel_size': 3, 'padding': 1, 'stride': 1, 'padding_mode': 'same', 'max_pool_kernel_size': 2, 'max_pool_stride': 2, 'pool_type': 'max'}, 
                                            {'input_dim': 400, 'out_channels': 200, 'kernel_size': 3, 'padding': 1, 'stride': 1, 'padding_mode': 'same', 'max_pool_kernel_size': 2, 'max_pool_stride': 2, 'pool_type': 'max'}]):
        """
        Args:
            dropout (float): Tasa de dropout.
            layers (list): Lista de diccionarios, cada uno configurando una
                           capa convolucional (input_dim, out_channels, 
                           kernel_size, pool_type, etc.).
        """
        super().__init__()
        self.dropout = dropout
        self.layers = layers
        self.subnet = nn.Sequential()

        for layer in layers:
            self.subnet.append(nn.Conv1d(layer['input_dim'], layer['out_channels'], 
                                        kernel_size=layer['kernel_size'], 
                                        padding=layer['padding'], 
                                        stride=layer['stride'], 
                                        padding_mode=layer['padding_mode']))
            self.subnet.append(nn.GELU())
            self.subnet.append(nn.Dropout(self.dropout))
            if layer['pool_type'] == 'max':
                self.subnet.append(nn.MaxPool1d(kernel_size=layer['max_pool_kernel_size'], stride=layer['max_pool_stride']))
            elif layer['pool_type'] == 'avg':
                self.subnet.append(nn.AvgPool1d(kernel_size=layer['max_pool_kernel_size'], stride=layer['max_pool_stride']))
            else:
                raise ValueError(f"Pool type {layer['pool_type']} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1,2)
        z = self.subnet(x)
        z = z.transpose(1,2)
        return z

class Multiple_MultiHeadAttention(nn.Module):
    """
    Aplica múltiples capas de MultiHeadAttention de forma secuencial.
    """
    def __init__(self, input_dim: int, num_heads: int, num_modules: int = 1, dropout: float = 0.0):
        """
        Args:
            input_dim (int): Dimensión del embedding (embed_dim).
            num_heads (int): Número de cabezas de atención.
            num_modules (int): Número de bloques de atención a apilar.
            dropout (float): Tasa de dropout en la atención.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_modules = num_modules
        self.dropout = dropout

        self.attention_modules = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first = True) for _ in range(num_modules)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_modules):
            x, _ = self.attention_modules[i](x, x, x)
        return x


#Red
class _ChannelModel(nn.Module):
    """
    x: (B, C, N, T)  ->  y: (B, C, U, T)

    Aplica, por frame temporal, una MLP sobre la dimensión de frecuencia:
      (i, N) -> (i, U)  para i en T, manteniendo T intacto.

    - shared_weights=True  : una sola MLP para todos los canales (vectorizado en B,C,T).
    - shared_weights=False : MLP distinta por canal (C MLPs).

    Usa la clase MLP extendida (con soporte de hidden_dims).
    """
    def __init__(
        self,
        input_dim: int,                 # N
        output_dim: int,                # U
        num_channels: int,              # C (constante)
        dropout: float = 0.0,
        shared_weights: bool = True,
        use_layernorm: bool = True,
        ln_eps: float = 1e-5,
        use_activation: bool = True,
        activation_function_selection: str = "gelu",
        hidden_dims: Optional[Union[int, Iterable[int]]] = 50,  # H1 por defecto
        final_activation: bool = True,      # activar o no en la última capa
        final_dropout: bool = False):        # dropout en la última capa
        """ Args: """
        super().__init__()
        self.N = input_dim
        self.U = output_dim
        self.C = num_channels
        self.shared = shared_weights

        mlp_kwargs = dict(
            dropout=dropout,
            use_activation=use_activation,
            activation_function_selection=activation_function_selection,
            hidden_dims=hidden_dims,
            final_activation=final_activation,
            final_dropout=final_dropout,
        )

        if shared_weights:
            # Una sola MLP aplicada a todos los canales/tiempos (vectorizado).
            self.proj = MLP(
                input_dim=self.N,
                output_dim=self.U,
                **mlp_kwargs
            )
        else:
            # Una MLP por canal (C fijo): bucle solo sobre C, vectorizado sobre B y T.
            self.proj_per_ch = nn.ModuleList([
                MLP(
                    input_dim=self.N,
                    output_dim=self.U,
                    **mlp_kwargs
                )
                for _ in range(self.C)
            ])

        self.norm = nn.LayerNorm(self.U, eps=ln_eps) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T)
        B, C, N, T = x.shape
        assert C == self.C, f"Esperado C={self.C}, recibido C={C}"
        assert N == self.N, f"Esperado N={self.N}, recibido N={N}"

        # Reordenamos a (B, C, T, N) para aplicar la MLP sobre la última dim (N)
        x_bctn = x.permute(0, 1, 3, 2).contiguous()  # (B, C, T, N)

        if self.shared:
            # PyTorch vectoriza Linear sobre las dims de batch (B,C,T)
            y_bctu = self.proj(x_bctn)               # (B, C, T, U)
            y_bctu = self.norm(y_bctu)               # LN sobre U
        else:
            ys = []
            for c in range(C):
                yc = self.proj_per_ch[c](x_bctn[:, c, :, :])  # (B, T, U)
                yc = self.norm(yc)
                ys.append(yc.unsqueeze(1))                    # (B, 1, T, U)
            y_bctu = torch.cat(ys, dim=1)                     # (B, C, T, U)

        # Volvemos a (B, C, U, T)
        y = y_bctu.permute(0, 1, 3, 2).contiguous()           # (B, C, U, T)
        return y
        

class _ConcatToSeq(nn.Module):
    """
    x: (B, C, U, T) -> y: (B, T, C*U)
    Funde canales por concatenación, preservando T. Sin parámetros.
    """
    def __init__(self, num_channels: int, u_dim: int):
        super().__init__()
        self.C = num_channels
        self.U = u_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, U, T)
        B, C, U, T = x.shape
        assert C == self.C and U == self.U, f"Esperado (C={self.C}, U={self.U}), recibido (C={C}, U={U})"
        xt = x.permute(0, 3, 1, 2).contiguous()   # (B, T, C, U)
        y  = xt.view(B, T, C * U)                 # (B, T, C*U)
        return y

class LearnedPositionalEncoding(nn.Module):
    """
    Positional Encoding aprendido para entradas (B, T, D).
    """
    def __init__(self, d_model, max_len):
        """
        Args:
            d_model (int): Dimensión del embedding.
            max_len (int): Longitud máxima de secuencia soportada."""
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)
    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class FixedPositionalEncoding(nn.Module):
    """
    Positional Encoding fijo (sinusoidal) para entradas (B, T, D).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)              # pares
        pe[:, 1::2] = torch.cos(position * div_term)              # impares
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        T = x.size(1)
        pe = self.pe[:T].to(device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, T, D)
        return x + pe


class HypocenterBackbone(nn.Module):
    """
    Pipeline:
      (B,C,N,T) -> ChannelModel -> (B,C,U,T)
      -> ConcatToSeq -> (B,T,C*U)
      -> MLP( C*U -> 50 -> N2 ) -> LN
      -> MultiheadAttention( N2, num_heads )      # por defecto 1 head
      -> Head( 10 -> out_dim )
    """
    def __init__(self,
                 num_channels: int,
                 n_bins: int,
                 u_dim: int,
                 out_dim: int,
                 *,
                 n2_dim: int = 42,                 # embedding después del MLP por frame
                 channel_hidden: Optional[Union[int, Iterable[int]]] = 50,  # p.ej. 130->50->U
                 frame_hidden: Optional[Union[int, Iterable[int]]] = 50,    # (C*U)->50->N2
                 mha_heads: int = 1,
                 dropout: float = 0.1,
                 shared_channel_mlp: bool = True,
                 use_activation: bool = True,
                 use_positional_encoding: bool = True,
                 positional_encoding_type: str = "fixed", # "fixed" o "learned"
                 mean_temporal_features: bool = False, # True => mean temporal features (we skip the multi channel mlp)
                 temporal_processing_module:str="mha",
                 n_temporal_processing_modules: int = 1, # number of temporal attention modules
                 act_sel: str = "gelu",
                 ln_eps: float = 1e-5,
                 use_cnn_embedding: bool = False,
                 cnn_embedding_layers: list[dict[str, int]] = [{'input_dim': 390, 'out_channels': 400, 'kernel_size': 3, 'padding': 1, 'stride': 1, 'padding_mode': 'same', 'max_pool_kernel_size': 2, 'max_pool_stride': 2}, 
                                            {'input_dim': 400, 'out_channels': 200, 'kernel_size': 3, 'padding': 1, 'stride': 1, 'padding_mode': 'same', 'max_pool_kernel_size': 2, 'max_pool_stride': 2}]):
        super().__init__()
        assert temporal_processing_module in ["mha","transformer"]

        if use_cnn_embedding:
            self.N2 = cnn_embedding_layers[-1]['out_channels']
        else:
            self.N2 = n2_dim

        if use_positional_encoding:
            if positional_encoding_type == "fixed":
                self.pe = FixedPositionalEncoding(d_model=self.N2, max_len=59)
            elif positional_encoding_type == "learned":
                self.pe = LearnedPositionalEncoding(d_model=self.N2, max_len=59)
            else:
                raise ValueError(f"Positional encoding type {positional_encoding_type} not supported")


        C, U = num_channels, u_dim
        self.C = C
        self.U = U
        self.N2 = n2_dim
        self.mean_temporal_features = mean_temporal_features
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.temporal_processing_module = temporal_processing_module
        self.num_temporal_attention_modules = n_temporal_processing_modules
        self.use_cnn_embedding = use_cnn_embedding
        self.cnn_embedding_layers = cnn_embedding_layers

        if not self.mean_temporal_features:
            # 1) Por canal:  (N -> H1 -> U) sobre cada (b,c,t,:)
            self.channel = _ChannelModel(
                input_dim=n_bins, output_dim=U, num_channels=C,
                dropout=dropout, shared_weights=shared_channel_mlp,
                use_activation=use_activation, activation_function_selection=act_sel,
                hidden_dims=channel_hidden,        
                final_activation=True, final_dropout=False
            )

            # 2) Concat: (B,C,U,T) -> (B,T,C*U)
            self.fuser = _ConcatToSeq(C, U)

        # 3) MLP por frame: 
        if self.mean_temporal_features: # (N -> frame_hidden -> N2)
            if self.use_cnn_embedding:
                self.frame_mlp = CNN_embedding(dropout=dropout, layers=cnn_embedding_layers)
                self.N2 = cnn_embedding_layers[-1]['out_channels']

            else:
                self.frame_mlp = MLP(
                    input_dim=n_bins, output_dim=self.N2, dropout=dropout,
                    use_activation=use_activation, activation_function_selection=act_sel,
                    hidden_dims=frame_hidden,
                    final_activation=False, final_dropout=False  # luego va LN + MHA
                )
        else: # (C*U -> frame_hidden -> N2)
            self.frame_mlp = MLP(
                input_dim=C*U, output_dim=self.N2, dropout=dropout,
                use_activation=use_activation, activation_function_selection=act_sel,
                hidden_dims=frame_hidden,
                final_activation=False, final_dropout=False  # luego va LN + MHA
            )
        self.ln = nn.LayerNorm(self.N2, eps=ln_eps)

        # 4) Multiple Multihead Attention (self-attn temporal)
        #    embed_dim debe ser divisible por num_heads (con 1 head siempre ok).
        if self.temporal_processing_module =="mha":
            self.temp_processing = Multiple_MultiHeadAttention(
                input_dim=self.N2, num_heads=mha_heads, num_modules=n_temporal_processing_modules, dropout=dropout)
        else:
            self.temp_processing = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.N2, 
                                                                                    nhead=mha_heads,
                                                                                    dim_feedforward=int(self.N2*4),
                                                                                    activation='gelu',
                                                                                    batch_first=True,
                                                                                    dropout=dropout), num_layers=n_temporal_processing_modules)


        self.dummy_head = nn.LazyLinear(out_dim)

    def forward(self, x):                 # x: (B,C,N,T)

        if self.mean_temporal_features: # dataloader already provides mean temporal features
            z = x  
        else:
            z = self.channel(x)               # (B,C,U,T)
            z = self.fuser(z)                 # (B,T,C*U)     

        z = self.frame_mlp(z)             # (B,T,N2)
        z = self.ln(z) 

        if self.use_positional_encoding:
            z = self.pe(z)                    # (B,T,N2)
        z = self.temp_processing(z)          # (B,T,N2) self-attn con 1 head (o más)

        y = self.dummy_head(z.view(z.size(0), -1))                   # (B,T,N2)

        return y

    