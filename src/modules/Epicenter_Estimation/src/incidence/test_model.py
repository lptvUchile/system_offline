import os
import numpy as np
import time
from tensorflow.python.keras import backend as K
import tensorflow as tf
import pandas as pd
from src.utils import (MyBatchGenerator_lstm_mlp, compute_binary_metrics, save_binary_metrics, 
                       save_results, save_regression_metrics, pad_and_convert_to_tensor, save_target_metrics, sec_div_max)
from src.incidence.model import IncidenceModel
from src.incidence.train_features_extraction import function_features_extraction
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import yaml
import argparse

parser = argparse.ArgumentParser(prog='Distance Test')
parser.add_argument('--test_file', type=str) #.yaml
parser.add_argument('--db_test', type=str) #.csv
parser.add_argument('--model_prefix',type=str,default='acc')
parser.add_argument('--seed',type=int) #seed del modelo a usar

args = parser.parse_args()
test_file = args.test_file
model_test_name = test_file.split(os.path.sep)[-1].split(".")[
    0
]
model_test_folder_name = test_file.split(os.path.sep)[-2]
seed = args.seed

with open(test_file, 'r') as file:
    test_config = yaml.safe_load(file)

feat_type = test_config.get('feature', None)
window_limit = test_config.get('window_limit',[20, 120])
use_accel = test_config.get('accel', True)
global_features = test_config.get('global_feat', True)
response = test_config.get('response', "vel")
filter = test_config.get('filter', None)
umbral_corte = test_config.get('umbral_corte', 0)
dummy = test_config.get('dummy', False)
lr = test_config.get('lr', 0.00015)
epochs = test_config.get('epochs', 1000)
patience = test_config.get('patience', 50)
parametrizador = test_config.get('parametrizador', None)
lstm_units = test_config.get('lstm_units', 10)
frame_length = test_config.get('frame_length', 4)
frame_shift = test_config.get('frame_shift', 2)
batch_size = test_config.get('batch_size', 16)
use_cnn_model = test_config.get('use_cnn_model', False)
use_cnn_features = test_config.get('use_cnn_features', False)
regression = test_config.get('regression', False)
coast = test_config.get('coast', False)
left_right = test_config.get('left_right', False)
one_hot_encoding = test_config.get('one_hot_encoding', False)
include_index = test_config.get('include_index',False)
include_env = test_config.get('include_env',False)
include_lat_lon = test_config.get('include_lat_lon',False)
include_magnitude = test_config.get('include_magnitude',False)
norm_energy = test_config.get('norm_energy',True)
concat_features = test_config.get('concat_features',False)
square_fft = test_config.get('square_fft',False)
log_scale = test_config.get('log_scale',True) 
include_p_vector = test_config.get('include_p_vector',False)
n_energy_frames = test_config.get('n_energy_frames',0)
how_to_include_p = test_config.get('how_to_include_p',0)
use_module_p = test_config.get('use_module_p',False)
use_raw_temporal_feat = test_config.get('use_raw_temporal_feat',False)
use_arcoss = test_config.get('use_arcoss',False)
use_VH_angle = test_config.get('use_VH_angle',False)

prefix = "acc" if use_accel else "vel"
test_name = test_file.split(os.path.sep)[-1].split(".")[
    0
]
test_folder_name = test_file.split(os.path.sep)[-2]

db_csv = args.db_test
feat_type = db_csv.split(os.path.sep)[-1].split(".")[0]


tf.keras.utils.set_random_seed(1)
function_features_extraction(
    BDtype = feat_type, 
    test = False, 
    use_accel = use_accel, 
    response = response, 
    window_limit = window_limit, 
    umbral_corte = umbral_corte, 
    filter = filter, 
    test_name = test_name, 
    test_folder_name = test_folder_name,
    use_cnn = use_cnn_features,
    one_hot_encoding = one_hot_encoding,
    include_index=include_index,
    include_env=include_env,
    include_lat_lon=include_lat_lon,
    include_magnitude=include_magnitude,
    norm_energy=norm_energy,
    concat_features = concat_features,
    square_fft=square_fft,
    log_scale = log_scale,
    include_p_vector = include_p_vector,
    n_energy_frames = n_energy_frames,
    how_to_include_p = how_to_include_p,
    use_module_p = use_module_p,
    use_raw_temporal_feat = use_raw_temporal_feat,
    use_arcoss = use_arcoss,
    use_VH_angle=use_VH_angle,
    conjuntos=["test"],
    path_csv = db_csv
    )

path_feat_in_test_cnn = f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_test_'+feat_type+'.npy'.replace('\\', '/')
path_feat_in_test_mlp = f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_test_'+feat_type+'.npy'.replace('\\', '/')
path_labels_test = f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/incidence_raw_test_'+feat_type+'.npy'.replace('\\', '/')


feat_in_test_cnn = np.load(path_feat_in_test_cnn,allow_pickle=True)
feat_in_test_mlp = np.load(path_feat_in_test_mlp, allow_pickle=True)
labels_test = np.load(path_labels_test, allow_pickle=True)
labels_test = pd.DataFrame(labels_test[()])
id_test = labels_test['event']
stations_test = labels_test['station']

# LOAD FEATURES
features_dir = os.path.join('data','features','incidence',args.model_prefix, model_test_folder_name, model_test_name)
min_f_train_lstm = np.load(os.path.join(features_dir,"min_temporal.npy"),allow_pickle=True)
max_f_train_lstm =  np.load(os.path.join(features_dir,"max_temporal.npy"),allow_pickle=True)
min_f_train_global = np.load(os.path.join(features_dir,"min_global.npy"),allow_pickle=True)
max_f_train_global =  np.load(os.path.join(features_dir,"max_global.npy"),allow_pickle=True)
max_target = np.load(os.path.join(features_dir,"max_target.npy"),allow_pickle=True)

feat_in_test_cnn = np.array([(feat_in_test_cnn[i]-min_f_train_lstm)/(max_f_train_lstm-min_f_train_lstm)
                                for i in range(len(feat_in_test_cnn))],dtype=object)
feat_in_test_mlp = np.array([(feat_in_test_mlp[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                            for i in range(len(feat_in_test_mlp))],dtype=object)

test_target = labels_test['incidence']

X_test_cnn, X_test_mlp = pad_and_convert_to_tensor(feat_in_test_cnn),tf.convert_to_tensor(feat_in_test_mlp, dtype=tf.float32)

#LOAD MODEL
model_path = os.path.join("models","incidence",args.model_prefix, model_test_folder_name, model_test_name,f"{args.seed}.h5")
try:
    model_keras = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")


test_prediction = model_keras.predict(
        [
            X_test_cnn,
            X_test_mlp,
        ]
    )
test_prediction = test_prediction.reshape(-1)*max_target


test_file_dir = os.path.dirname(args.test_file)
tests_csv_path = os.path.join(test_file_dir, "results", f"{feat_type}.csv")
os.makedirs(os.path.dirname(tests_csv_path), exist_ok=True)

regression_results_dir = os.path.join(test_file_dir, "results", "regression_results", feat_type)
os.makedirs(regression_results_dir, exist_ok=True)

mape = np.mean(np.abs(test_target-test_prediction)/test_target)
mae = np.mean(np.abs(test_target-test_prediction))
std = np.std(np.abs(test_target-test_prediction))

tests_csv = pd.DataFrame(columns=["seed", "mae", "mape", "std"])
tests_csv.set_index("seed", inplace=True)
tests_csv.loc[seed] = {}
tests_csv.at[seed,"mae"] = mae
tests_csv.at[seed,"mape"] = mape
tests_csv.at[seed,"std"] = std

tests_csv.to_csv(tests_csv_path, index=True)

df_test_distance_path = os.path.join(regression_results_dir, f"{seed}_test_mode_incidence.csv")
df_test_distance = pd.DataFrame(np.array([id_test,stations_test,test_target,test_prediction]).T,columns = ["id_evento", "Estacion", "Real", "Estimacion"])
df_test_distance.to_csv(df_test_distance_path, index=False)

print(f"Saved results at: {df_test_distance_path}")
print("Done")