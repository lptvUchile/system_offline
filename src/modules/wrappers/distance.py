
import os

import tensorflow as tf

from modules.Epicenter_Estimation import pad_and_convert_to_tensor

from modules.Epicenter_Estimation import DistanceModel
from modules.Epicenter_Estimation import DistanceFeatureExtractor

import numpy as np
import yaml


class DistanceWrapper:
    def __init__(self, folder_path, model_name):
        self.folder_path = folder_path
        self.cwd = os.getcwd()
        model_path = os.path.join(self.cwd, self.folder_path, f"{model_name}.h5")
        max_temporal_path = os.path.join(self.cwd, self.folder_path, "max_temporal.npy")
        min_temporal_path = os.path.join(self.cwd, self.folder_path, "min_temporal.npy")
        max_global_path = os.path.join(self.cwd, self.folder_path, "max_global.npy")
        min_global_path = os.path.join(self.cwd, self.folder_path, "min_global.npy")
        max_target_path = os.path.join(self.cwd, self.folder_path, "max_target.npy")
        self.max_temporal = np.load(max_temporal_path, allow_pickle=True)
        self.min_temporal = np.load(min_temporal_path, allow_pickle=True)
        self.max_global = np.load(max_global_path, allow_pickle=True)
        self.min_global = np.load(min_global_path, allow_pickle=True)
        self.max_target = np.load(max_target_path, allow_pickle=True)


        epicentral_yaml_path = os.path.join(self.cwd, self.folder_path, "config.yaml")
        with open(epicentral_yaml_path, "r") as f:
            self.epicentral_config = yaml.safe_load(f)


        self.model = DistanceModel()
        self.model.load_model(
            model_path=model_path,
            max_temporal_path=max_temporal_path,
            min_temporal_path=min_temporal_path,
            max_global_path=max_global_path,
            min_global_path=min_global_path,
            max_target_path=max_target_path,
        )

        self.feature_extractor = DistanceFeatureExtractor(
            window_limit=self.epicentral_config["window_limit"],
            umbral_corte=self.epicentral_config["umbral_corte"],
            filter=self.epicentral_config["filter"],
            path_st=os.path.join(self.cwd, self.folder_path, "stations_index.csv"),
            response=self.epicentral_config["response"],
        )

    def estimate_distance(self, trace, time, inv):
        temporal_features, global_features = self.feature_extractor.get_features(
                trace,  time, inv, one_hot_encoding = False ,use_cnn = False,
                    include_index=self.epicentral_config["include_index"], include_env=self.epicentral_config["include_env"], include_lat_lon=self.epicentral_config["include_lat_lon"],
                    norm_energy=self.epicentral_config["norm_energy"], concat_features=False, square_fft=self.epicentral_config["square_fft"], log_scale=self.epicentral_config["log_scale"],
                    include_p_vector = self.epicentral_config["include_p_vector"], n_energy_frames=self.epicentral_config["n_energy_frames"], how_to_include_p = self.epicentral_config["how_to_include_p"],
                    use_module_p = self.epicentral_config["use_module_p"]
                )
        temporal_features = [temporal_features[0]]
        global_features = [global_features[0]]
        

        feat_norm_test_lstm = np.array([(temporal_features[i]-self.min_temporal)/(self.max_temporal-self.min_temporal)
                            for i in range(len(temporal_features))],dtype=object)
        feat_norm_test_global = np.array([(global_features[i]-self.min_global)/(self.max_global-self.min_global)
                for i in range(len(global_features))],dtype=object)

        X_test_lstm, X_test_mlp = pad_and_convert_to_tensor(feat_norm_test_lstm), tf.convert_to_tensor(feat_norm_test_global, dtype=tf.float32)
        
        test_prediction = self.model.model.predict(
        [
                X_test_lstm,
                X_test_mlp,
        ]
        )[:,0]
        dist = test_prediction[0]*self.max_target
        return dist
