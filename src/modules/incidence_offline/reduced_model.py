from .blocks import IncidenceBackbone
import torch

class IncidenceModelTorch: 
    def __init__(self,device,model_cfg):

        try:
            self.DEVICE = torch.device(device)  

            if self.DEVICE.type == "cuda":
                # Forzamos el driver de CUDA para asegurar disponibilidad
                with torch.no_grad():
                    _ = torch.empty(1, device=self.DEVICE) 

            print(f"Device {self.DEVICE} set successfully")

        except Exception as e:
            print(f"CUDA no usable ({e}). Usando CPU.")
            self.DEVICE = torch.device("cpu")

        use_splited_mlp     = model_cfg["use_splited_mlp"]  # True => pesos por canal
        # extras para el backbone:
        self.num_channels   = model_cfg["num_channels"]     # C
        self.u_dim          = model_cfg["u_dim"]            # U (p. ej. N//3)
        self.n2_dim         = model_cfg["n2_dim"]           # embedding despues del MLP por frame
        self.nhead          = model_cfg["nhead"]            # numero de heads para la atencion
        self.act_sel        = model_cfg["act_sel"]          # funcion de activacion
        self.ln_eps         = model_cfg["ln_eps"]           # epsilon para la normalizacion
        self.channel_hidden = model_cfg["channel_hidden"]   # p.ej. 130->50->U
        self.frame_hidden   = model_cfg["frame_hidden"]     # (C*U)->50->N2
        self.gru_hidden     = model_cfg["gru_hidden"]       # hidden para la GRU
        self.dropout        = model_cfg["dropout"]          # dropout
        self.shared_channel_mlp = model_cfg["shared_channel_mlp"] # True => pesos compartidos
        self.use_activation   = model_cfg["use_activation"]   # True => usar activacion
        self.input_dim = model_cfg["input_dim"]          # numero de bins de entrada (features)
        self.output_dim = model_cfg["output_dim"]        # dimension de salida (1)

        # --------- MODEL DEFINITION ----------
        self.model = IncidenceBackbone(
                 num_channels=self.num_channels,
                 n_bins=self.input_dim,
                 u_dim=self.u_dim,
                 out_dim=self.output_dim,
                 n2_dim=self.n2_dim,                 # embedding despues del MLP por frame
                 channel_hidden=self.channel_hidden,  # p.ej. 130->50->U
                 frame_hidden=self.frame_hidden,    # (C*U)->50->N2
                 gru_hidden=self.gru_hidden,
                 mha_heads=self.nhead,
                 dropout=self.dropout,
                 shared_channel_mlp=self.shared_channel_mlp,
                 use_activation=self.use_activation,
                 act_sel=self.act_sel,
                 ln_eps=self.ln_eps
        ).to(self.DEVICE)
        # ------------------------------------
 
    def evaluate_wrapper_mode(self,input: torch.Tensor):
        assert input.size(0)==1, "Input see debe entregar como tensor de batch size 1 (dimensión (0))"
        input = input.to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input).cpu().numpy().item()
        return pred


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.to(self.DEVICE)
        print("Model loaded successfully!")
    
        