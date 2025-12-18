from .blocks import HypocenterBackbone
import torch

class HypocenterModelTorch: 
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


        # --------- MODEL DEFINITION ----------
        self.model = HypocenterBackbone(
                 **model_cfg).to(self.DEVICE)
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
    
        