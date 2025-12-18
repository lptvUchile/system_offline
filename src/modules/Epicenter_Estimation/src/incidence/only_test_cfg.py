import numpy as np
import time
from tensorflow.python.keras import backend as K
import tensorflow as tf
import os
import pandas as pd
from src.utils import (MyBatchGenerator_lstm_mlp, compute_binary_metrics, save_binary_metrics, 
                       save_results, save_regression_metrics, pad_and_convert_to_tensor, sec_div_max)
from src.incidence import IncidenceModel
from src.incidence.test_features_extraction import function_features_extraction
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import argparse

parser = argparse.ArgumentParser(prog='Incidence Training')
parser.add_argument('--test_file', type=str)
parser.add_argument('--test_csv', type=str)
parser.add_argument('--feat_type', type=str)
parser.add_argument('--use_accel', action=argparse.BooleanOptionalAction)
parser.add_argument('--global_features', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--response', type=str, default='VEL')
parser.add_argument('--extract_features', action=argparse.BooleanOptionalAction)
parser.add_argument("--binary", action=argparse.BooleanOptionalAction)
parser.add_argument("--window_limit", type=list, default=[20, 120])
parser.add_argument("--umbral_corte", type=float, default=0.03)
parser.add_argument("--one_hot_encoding", action=argparse.BooleanOptionalAction)
parser.add_argument("--include_index",action=argparse.BooleanOptionalAction)
parser.add_argument("--include_env",action=argparse.BooleanOptionalAction)
parser.add_argument("--include_lat_lon",action=argparse.BooleanOptionalAction)
parser.add_argument("--include_magnitude",action=argparse.BooleanOptionalAction)
parser.add_argument("--norm_energy",action=argparse.BooleanOptionalAction)
parser.add_argument("--concat_features",action=argparse.BooleanOptionalAction)
parser.add_argument("--square_fft",action=argparse.BooleanOptionalAction)
parser.add_argument("--log_scale",action=argparse.BooleanOptionalAction)

args = parser.parse_args()
test_file = args.test_file
# test_file is a yaml file with the following structure:
# feature: mayores4M
# binary: true
# window_limit: [1, 3]
# seeds: [1, 2, 3, 4, 5]
# accel: true
# global_feat: true
# response: "vel"

# load yaml
import yaml

with open(test_file, 'r') as file:
    test_config = yaml.safe_load(file)

# Extract values from the loaded YAML
feat_type = test_config.get('feature', args.feat_type)
test_csv = test_config.get('test_csv', args.test_csv)
binary = test_config.get('binary', args.binary)
window_limit = test_config.get('window_limit', args.window_limit)
seeds = test_config.get('seeds', [args.seed])
use_accel = test_config.get('accel', args.use_accel)
global_features = test_config.get('global_feat', args.global_features)
test = args.test
response = test_config.get('response', args.response)
filter = test_config.get('filter', None)
umbral_corte = test_config.get('umbral_corte', args.umbral_corte)
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

print("--- PARÁMETROS ---")
print(f"feat_type: {feat_type}")
print(f"test_csv: {test_csv}")
print(f"binary: {binary}")
print(f"window_limit: {window_limit}")
print(f"seeds: {seeds}")
print(f"use_accel: {use_accel}")
print(f"global_features: {global_features}")
print(f"test: {test}")
print(f"response: {response}")
print(f"filter: {filter}")
print(f"umbral_corte: {umbral_corte}")
print(f"dummy: {dummy}")
print(f"lr: {lr}")
print(f"epochs: {epochs}")
print(f"patience: {patience}")
print(f"parametrizador: {parametrizador}")
print(f"lstm_units: {lstm_units}")
print(f"frame_length: {frame_length}")
print(f"frame_shift: {frame_shift}")
print(f"batch_size: {batch_size}")
print(f"use_cnn_features: {use_cnn_features}")
print(f"use_cnn_model: {use_cnn_model}")
print(f"regression: {regression}")
print(f"coast: {coast}")
print(f"left_right: {left_right}")


prefix = "acc" if use_accel else "vel"
test_name = test_file.split(os.sep)[-1].split(".")[
    0
]
test_folder_name = test_file.split(os.sep)[-2]
print(test_file)
print(test_name)
time.sleep(5)
if args.extract_features:
    tf.keras.utils.set_random_seed(1)
    function_features_extraction(
        BDtype = test_csv, 
        test = test, 
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
        conjuntos = ["test"]
        )
    

start_time = time.time()
path_root =  os.getcwd()


path_feat_in_test_lstm = f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/test/feat_lstm_raw_test_'+test_csv+'.npy'.replace('\\', '/')
path_dist_real_test = f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/test/incidence_raw_test_'+test_csv+'.npy'.replace('\\', '/')


feat_in_test_lstm = np.load(path_feat_in_test_lstm, allow_pickle=True)
dist_real_test = np.load(path_dist_real_test, allow_pickle=True)
dist_real_test =  pd.DataFrame(dist_real_test[()])
id_test = dist_real_test['event']
stations_test = dist_real_test['station']

min_f_train_lstm_path = f"./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/min_temporal.npy"
max_f_train_lstm_path = f"./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/max_temporal.npy"

min_f_train_lstm = np.load(min_f_train_lstm_path, allow_pickle=True)
max_f_train_lstm = np.load(max_f_train_lstm_path, allow_pickle=True)

max_target_path = f"./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/max_target.npy"
max_target = np.load(max_target_path, allow_pickle=True)


feat_norm_test_lstm = np.array([(feat_in_test_lstm[i]-min_f_train_lstm)/(max_f_train_lstm-min_f_train_lstm)
                                for i in range(len(feat_in_test_lstm))],dtype=object)






tests_csv = pd.DataFrame(columns=["seed", "mae_dist_train", "mae_dist_val", "mae_dist_test"])
tests_csv.set_index("seed", inplace=True)

labels_test = dist_real_test['incidence']/max_target

# Save temporal and global features
os.makedirs(f'./models/incidence/{prefix}/{test_name}/', exist_ok=True)
os.makedirs(f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/', exist_ok=True)
if not use_cnn_features:
    # also save max_target for regression
    np.save(f'./data/features/incidence/{prefix}/{test_folder_name}/{test_name}/max_target.npy',max_target)


for seed in seeds:
   

    dist_model = IncidenceModel()

    os.makedirs(f"./models/incidence/{prefix}/{test_name}", exist_ok=True)
    model_path = f"./models/incidence/{prefix}/{test_folder_name}/{test_name}/{seed}.h5"

    dist_model.raw_load_model(model_path)
    
    model = dist_model.model

    X_test_lstm = pad_and_convert_to_tensor(feat_norm_test_lstm)

    test_prediction = model.predict(
        [X_test_lstm]
    )
    
    test_prediction = np.hstack(test_prediction)



    test_target = labels_test*max_target


    test_prediction = test_prediction*max_target


    mape_test = np.mean(np.abs(test_target-test_prediction)/test_target)

    mae_test = np.mean(np.abs(test_target-test_prediction))
    
    print('Error MAPE corregido sobre test: ',mape_test)
    print('Error MAE sobre test: ',mae_test)


