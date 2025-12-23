import torch
import pandas as pd
import numpy as np
from .features import FeatureExtractor
from typing import List
import warnings


warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor.*",
    category=UserWarning
)

class OfflineDistancePreprocessing:

    def __init__(self, 
                response="VEL", 
                frame_length = 4,
                frame_shift = 2,
                window_limit=[20, 120], 
                filter=None, 
                norm_energy=False,
                square_fft = False,
                log_scale = False,
                mean_temporal_features=False,
                NEW_concat_temporal_features=False):

    
        
        self.preprocessing_params = {"response":response,
                                    "frame_length":frame_length,
                                    "frame_shift":frame_shift,
                                    "window_limit":window_limit,
                                    "filter":filter,
                                    "norm_energy":norm_energy,
                                    "square_fft":square_fft,
                                    "log_scale":log_scale}
        self.preprocessing = FeatureExtractor(**self.preprocessing_params)
        self.mean_temporal_features = mean_temporal_features
        self.NEW_concat_temporal_features = NEW_concat_temporal_features


    def preprocess_data(self,trace, frame_p,inv):
        #trace = canal_sac_Z+canal_sac_E+canal_sac_N

        try:
            feat_canales_temporal = self.preprocessing.get_features(trace=trace,
                                                                    frame_p=frame_p,
                                                                    inv=inv)
        except Exception as e:
            print(f"Error en el preprocesamiento de la señal: {e}")
            return None
                                  
        feat_canales_temporal = feat_canales_temporal[0]

        if self.mean_temporal_features and not self.NEW_concat_temporal_features:
            feat_canales_temporal = torch.tensor(feat_canales_temporal.mean(axis=0), dtype=torch.float32).expand(1, -1, -1)
            feat_canales_temporal = feat_canales_temporal.transpose(2, 1)

        elif self.NEW_concat_temporal_features and not self.mean_temporal_features:
            # Concatena canales en la dimensión de features -> [1, F*C, T]
            x = feat_canales_temporal
            if isinstance(x, (list, tuple)):
                x = torch.cat([torch.as_tensor(a, dtype=torch.float32) for a in x], dim=0)  # [C_total, F, T]
            else:
                x = torch.as_tensor(x, dtype=torch.float32)  # [C, F, T]
            t = x.permute(1, 0, 2).contiguous().view(-1, x.size(2))  # [F*C_total, T]
            feat_canales_temporal = t.unsqueeze(0).transpose(2, 1) 
        
        feat_canales_temporal = torch.tensor(feat_canales_temporal, dtype=torch.float32).expand(1, -1, -1)
        #Repetir primer frame hacia el inicio hasta completar el tamaño (falta señal de inicio)
        #Hasta completar 59 T
        if feat_canales_temporal.shape[1] < 59:
            feat_canales_temporal = torch.cat([feat_canales_temporal[:, 0].unsqueeze(1).repeat(1, 59 - feat_canales_temporal.shape[1], 1), feat_canales_temporal], dim=1)

        return feat_canales_temporal
    