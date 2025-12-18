import torch
import os
import pandas as pd
import numpy as np
from .features import FeatureExtractor
from obspy import UTCDateTime, read, read_inventory
from typing import List
import warnings


warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor.*",
    category=UserWarning
)

class OfflineIncidencePreprocessing:

    def __init__(self, 
                response="VEL", 
                frame_length = 4,
                frame_shift = 2,
                window_limit=[20, 120], 
                filter=None, 
                norm_energy=False,
                square_fft = False,
                log_scale = False):

    
        
        self.preprocessing_params = {"response":response,
                                    "frame_length":frame_length,
                                    "frame_shift":frame_shift,
                                    "window_limit":window_limit,
                                    "filter":filter,
                                    "norm_energy":norm_energy,
                                    "square_fft":square_fft,
                                    "log_scale":log_scale}
        self.preprocessing = FeatureExtractor(**self.preprocessing_params)


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
        
        feat_canales_temporal = torch.tensor(feat_canales_temporal, dtype=torch.float32).expand(1,-1, -1, -1)

        return feat_canales_temporal
    