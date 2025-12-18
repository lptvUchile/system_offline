import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Union, Iterable, List


#Bloques (Unidades dela red)

class MLP(nn.Module):
    """
    MLP flexible: Linear(+hidden) -> ... -> Linear(out)
    - hidden_dims: int o lista de ints con los tamaños ocultos (p.ej. 50 o [100,50]).
    - dropout: se aplica después de cada capa NO final (y opcionalmente en la final).
    - use_activation: activa/desactiva activación en todas las capas donde aplique.
    - activation_function_selection: 'tanh','relu','gelu', etc. (como en tu código).
    - final_activation / final_dropout: para controlar la última capa.

    Si hidden_dims es None, se comporta como tu MLP original: una sola Linear.
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
        final_dropout: bool = False,
    ):
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
        final_dropout: bool = False,        # dropout en la última capa
    ):
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



class IncidenceBackbone(nn.Module):
    """
    Pipeline:
      (B,C,N,T) -> ChannelModel -> (B,C,U,T)
      -> ConcatToSeq -> (B,T,C*U)
      -> MLP( C*U -> 50 -> N2 ) -> LN
      -> MultiheadAttention( N2, num_heads )      # por defecto 1 head
      -> GRU( N2 -> hidden=10 )
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
                 gru_hidden: int = 10,
                 mha_heads: int = 1,
                 dropout: float = 0.1,
                 shared_channel_mlp: bool = True,
                 use_activation: bool = True,
                 act_sel: str = "gelu",
                 ln_eps: float = 1e-5):
        super().__init__()

        C, U = num_channels, u_dim
        self.C = C
        self.U = U
        self.N2 = n2_dim
        self.gru_h = gru_hidden

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

        # 3) MLP por frame: (C*U -> frame_hidden -> N2)
        self.frame_mlp = MLP(
            input_dim=C*U, output_dim=self.N2, dropout=dropout,
            use_activation=use_activation, activation_function_selection=act_sel,
            hidden_dims=frame_hidden,
            final_activation=False, final_dropout=False  # luego va LN + MHA
        )
        self.ln = nn.LayerNorm(self.N2, eps=ln_eps)

        # 4) Multihead Attention (self-attn temporal)
        #    embed_dim debe ser divisible por num_heads (con 1 head siempre ok).
        self.mha = nn.MultiheadAttention(
            embed_dim=self.N2, num_heads=mha_heads,
            dropout=dropout, batch_first=True
        )

        # 5) GRU sobre tiempo
        self.gru = nn.GRU(input_size=self.N2, hidden_size=self.gru_h, batch_first=True)

        # 6) Head final: (gru_hidden -> out_dim)
        self.head = MLP(
            input_dim=self.gru_h, output_dim=out_dim, dropout=dropout,
            use_activation=use_activation, activation_function_selection=act_sel,
            hidden_dims=None,                 # solo Linear por defecto
            final_activation=False, final_dropout=False
        )

    def forward(self, x):                 # x: (B,C,N,T)
        z = self.channel(x)               # (B,C,U,T)
        z = self.fuser(z)                 # (B,T,C*U)
        z = self.frame_mlp(z)             # (B,T,N2)
        z = self.ln(z)                    # (B,T,N2)
        z, _ = self.mha(z, z, z)          # (B,T,N2) self-attn con 1 head (o más)

        _, h = self.gru(z)                # h: (num_layers*directions=1, B, gru_hidden)
        g = h[-1]                         # (B,gru_hidden)
        y = self.head(g)                  # (B,out_dim)
        return y

    
