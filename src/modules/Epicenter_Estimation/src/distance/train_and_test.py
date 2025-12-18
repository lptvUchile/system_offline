import numpy as np
import time
from tensorflow.python.keras import backend as K
import tensorflow as tf
import os
import pandas as pd
from src.utils import (MyBatchGenerator_lstm_mlp, compute_binary_metrics, save_binary_metrics, 
                       save_results, save_regression_metrics, pad_and_convert_to_tensor, save_target_metrics, sec_div_max)
from src.distance import DistanceModel
from src.distance.train_features_extraction import function_features_extraction
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import argparse

parser = argparse.ArgumentParser(prog='Distance Training')
parser.add_argument('--test_file', type=str)
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
parser.add_argument("--include_p_vector", action = argparse.BooleanOptionalAction)
parser.add_argument("--n_energy_frames", type=int,default=0)
parser.add_argument("--how_to_include_p",type = int,default=0)
parser.add_argument("--use_module_p",action= argparse.BooleanOptionalAction)

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
include_p_vector = test_config.get('include_p_vector',False)
n_energy_frames = test_config.get('n_energy_frames',0)
how_to_include_p = test_config.get('how_to_include_p',0)
use_module_p = test_config.get('use_module_p',False)

print("--- PARÁMETROS ---")
print(f"feat_type: {feat_type}")
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
test_name = test_file.split("/")[-1].split(".")[
    0
]
test_folder_name = test_file.split("/")[-2]
print(test_file)
print(test_name)
time.sleep(5)
if args.extract_features:
    tf.keras.utils.set_random_seed(1)
    function_features_extraction(
        BDtype = feat_type, 
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
        include_p_vector = include_p_vector,
        n_energy_frames = n_energy_frames,
        how_to_include_p = how_to_include_p,
        use_module_p = use_module_p
        )
    

start_time = time.time()
path_root =  os.getcwd()

path_feat_in_train_lstm = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_train_'+feat_type+'.npy'.replace('\\', '/')
path_feat_in_train_mlp  = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_train_'+feat_type+'.npy'.replace('\\', '/')
distth_az_real_train      = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/distance_raw_train_'+feat_type+'.npy'.replace('\\', '/')

path_feat_in_val_lstm = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_val_'+feat_type+'.npy'.replace('\\', '/')
path_feat_in_val_mlp = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_val_'+feat_type+'.npy'.replace('\\', '/')
path_dist_real_val = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/distance_raw_val_'+feat_type+'.npy'.replace('\\', '/')
path_feat_in_test_lstm = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_test_'+feat_type+'.npy'.replace('\\', '/')
path_feat_in_test_mlp = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_test_'+feat_type+'.npy'.replace('\\', '/')
path_dist_real_test = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/distance_raw_test_'+feat_type+'.npy'.replace('\\', '/')


feat_in_train_lstm = np.load(path_feat_in_train_lstm, allow_pickle=True)
feat_in_train_mlp = np.load(path_feat_in_train_mlp, allow_pickle=True)
dist_real_train = np.load(distth_az_real_train, allow_pickle=True)
dist_real_train = pd.DataFrame(dist_real_train[()])
id_train = dist_real_train['event']
stations_train = dist_real_train['station']


feat_in_val_lstm = np.load(path_feat_in_val_lstm, allow_pickle=True)
feat_in_val_mlp = np.load(path_feat_in_val_mlp, allow_pickle=True)
dist_real_val = np.load(path_dist_real_val, allow_pickle=True)
dist_real_val =  pd.DataFrame(dist_real_val[()])
id_val = dist_real_val['event']
stations_val = dist_real_val['station']

feat_in_test_lstm = np.load(path_feat_in_test_lstm, allow_pickle=True)
feat_in_test_mlp = np.load(path_feat_in_test_mlp, allow_pickle=True)
dist_real_test = np.load(path_dist_real_test, allow_pickle=True)
dist_real_test =  pd.DataFrame(dist_real_test[()])
id_test = dist_real_test['event']
stations_test = dist_real_test['station']

feat_in_train_mlp = np.array([feat_in_train_mlp[i] for i in range(len(feat_in_train_mlp))],dtype='float32')
feat_in_val_mlp = np.array([feat_in_val_mlp[i] for i in range(len(feat_in_val_mlp))],dtype='float32')
feat_in_test_mlp = np.array([feat_in_test_mlp[i] for i in range(len(feat_in_test_mlp))],dtype='float32')

if use_cnn_features:
    feat_norm_train_lstm = sec_div_max(feat_in_train_lstm)
    feat_norm_val_lstm = sec_div_max(feat_in_val_lstm)
    feat_norm_test_lstm = sec_div_max(feat_in_test_lstm)

else:
    # spectral features are normalized through min-max normalization
    min_f_train_lstm = np.min([np.min(x,0) for x in feat_in_train_lstm],0)
    max_f_train_lstm =  np.max([np.max(x,0) for x in feat_in_train_lstm],0)

    feat_norm_train_lstm = np.array([(feat_in_train_lstm[i]-min_f_train_lstm)/(max_f_train_lstm-min_f_train_lstm)
                                    for i in range(len(feat_in_train_lstm))],dtype=object)
    feat_norm_val_lstm = np.array([(feat_in_val_lstm[i]-min_f_train_lstm)/(max_f_train_lstm-min_f_train_lstm)
                                    for i in range(len(feat_in_val_lstm))],dtype=object)
    feat_norm_test_lstm = np.array([(feat_in_test_lstm[i]-min_f_train_lstm)/(max_f_train_lstm-min_f_train_lstm)
                                    for i in range(len(feat_in_test_lstm))],dtype=object)

    min_f_train_global = np.min(feat_in_train_mlp,0)
    max_f_train_global =  np.max(feat_in_train_mlp,0)
    feat_norm_train_global = np.array([(feat_in_train_mlp[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                                for i in range(len(feat_in_train_mlp))],dtype=object)
    feat_norm_val_global = np.array([(feat_in_val_mlp[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                                for i in range(len(feat_in_val_mlp))],dtype=object)
    feat_norm_test_global = np.array([(feat_in_test_mlp[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                                for i in range(len(feat_in_test_mlp))],dtype=object)

largo_cota = None #feat_in_train_lstm.shape[1]max_target
tam_feat_mlp = feat_in_train_mlp.shape[1] #### 49
tam_feat_lstm = feat_norm_train_lstm[0].shape[1]


#### Choose between the types of experiment:
# if left_right, the model will predict if the event is to the left or to the right of the station
# if coast, the model will predict if the event is coastal or not
# if regression, the model will predict the distance of the event

if left_right:

    tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
    labels_train_izquierda = dist_real_train["evento_izquierda_estacion"].values
    labels_train_derecha = dist_real_train["evento_derecha_estacion"].values
    labels_train = np.concatenate(
        (labels_train_izquierda.reshape(-1, 1), labels_train_derecha.reshape(-1, 1)),axis=1
    )

    labels_val_izquierda = dist_real_val["evento_izquierda_estacion"].values
    labels_val_derecha = dist_real_val["evento_derecha_estacion"].values

    labels_val = np.concatenate(
        (labels_val_izquierda.reshape(-1, 1), labels_val_derecha.reshape(-1, 1)),axis=1
    )
    labels_test_izquierda = dist_real_test["evento_izquierda_estacion"].values
    labels_test_derecha = dist_real_test["evento_derecha_estacion"].values

    labels_test = np.concatenate(
        (labels_test_izquierda.reshape(-1, 1), labels_test_derecha.reshape(-1, 1)),axis=1
    )

    tests_csv.set_index("seed", inplace=True)

elif coast:
    tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
    labels_train = dist_real_train["costero"].values
    labels_val = dist_real_val["costero"].values
    labels_test = dist_real_test["costero"].values
    tests_csv.set_index("seed", inplace=True)

elif regression:
    tests_csv = pd.DataFrame(columns=["seed", "mae_train", "mae_val", "mae_test"])
    tests_csv.set_index("seed", inplace=True)
    dist_real_train = dist_real_train['distance']
    max_target = dist_real_train.max()

    labels_train = dist_real_train/max_target
    labels_val = dist_real_val['distance']/max_target
    labels_test = dist_real_test['distance']/max_target

# Save temporal and global features
os.makedirs(f'./models/distance/{prefix}/{test_name}/', exist_ok=True)
os.makedirs(f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/', exist_ok=True)
if not use_cnn_features:
    np.save(f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/min_temporal.npy',min_f_train_lstm)
    np.save(f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/max_temporal.npy',max_f_train_lstm) 
    np.save(f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/min_global.npy',min_f_train_global)
    np.save(f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/max_global.npy',max_f_train_global)
    # also save max_target for regression
    np.save(f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/max_target.npy',max_target)


for seed in seeds:
    tf.keras.utils.set_random_seed(seed)
    config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config) 
    repeats = 1
    nro_output = 1
    K.set_session(sess)

    y_estimada_repeat_train, error_repeat_train_mse,error_repeat_train_rel, loss_repeat_train= [], [], [], [] 
    y_estimada_repeat_val, error_repeat_val_mse, error_repeat_val_rel, loss_repeat_val= [], [], [], [] 
    y_estimada_repeat_test, error_repeat_test_mse,error_repeat_test_rel= [], [], []

    dist_model = DistanceModel()
    dist_model.setup_model(tam_feat_mlp=tam_feat_mlp, tam_feat_lstm=tam_feat_lstm, largo_cota=largo_cota, 
                           nro_output=nro_output, global_feat=global_features, binary=binary, lstm_units=lstm_units, use_cnn=use_cnn_model)

    os.makedirs(f"./models/distance/{prefix}/{test_name}", exist_ok=True)
    model_path = f"./models/distance/{prefix}/{test_folder_name}/{test_name}/{seed}.h5"

    if binary:
        dist_model.setup_training(model_path=model_path, loss="categorical_crossentropy", dummy=dummy, patience=patience, epochs=epochs, lr=lr)
    else:
        dist_model.setup_training(model_path=model_path, loss="mean_squared_error", dummy=dummy, patience=patience, epochs=epochs, lr=lr)
    dist_model.input_training(feat_norm_train_lstm, feat_norm_train_global, labels_train, batch_size=batch_size)
    dist_model.input_validation(feat_norm_val_lstm, feat_norm_val_global, labels_val, batch_size=batch_size)
    dist_model.train()
    model = dist_model.model

    # X_train_lstm, X_train_mlp, y_train = feat_norm_train_lstm, feat_in_train_mlp, labels_train
    # X_val_lstm, X_val_mlp, y_val = feat_norm_val_lstm, feat_in_val_mlp, labels_val
    # X_test_lstm, X_test_mlp, y_test = feat_norm_test_lstm, feat_in_test_mlp, labels_test

    # train_f = MyBatchGenerator_lstm_mlp( X_train_lstm,X_train_mlp, np.zeros(len(y_train)), batch_size=batch_size, shuffle=False, global_feat=not args.no_global_features)
    # val_f = MyBatchGenerator_lstm_mlp( X_val_lstm,X_val_mlp, np.zeros(len(y_val)), batch_size=batch_size, shuffle=False, global_feat=not args.no_global_features)
    # test_f = MyBatchGenerator_lstm_mlp( X_test_lstm,X_test_mlp, np.zeros(len(y_test)), batch_size=batch_size, shuffle=False, global_feat=not args.no_global_features)

    # train_prediction = model.predict(train_f)
    # val_prediction = model.predict(val_f)
    # test_prediction = model.predict(test_f)

    X_train_lstm, X_train_mlp = pad_and_convert_to_tensor(feat_norm_train_lstm), tf.convert_to_tensor(feat_norm_train_global, dtype=tf.float32)
    X_val_lstm, X_val_mlp = pad_and_convert_to_tensor(feat_norm_val_lstm), tf.convert_to_tensor(feat_norm_val_global, dtype=tf.float32)
    X_test_lstm, X_test_mlp = pad_and_convert_to_tensor(feat_norm_test_lstm), tf.convert_to_tensor(feat_norm_test_global, dtype=tf.float32)

    if global_features:
        train_prediction = model.predict(
            [
                X_train_lstm,
                X_train_mlp,
            ]
        )[:,0]
        val_prediction = model.predict(
            [
                X_val_lstm,
                X_val_mlp,
            ]
        )[:,0]
        test_prediction = model.predict(
            [
                X_test_lstm,
                X_test_mlp,
            ]
        )[:,0]
    else:
        train_prediction = model.predict(
            [X_train_lstm]
        )
        val_prediction = model.predict(
            [X_val_lstm]
        )
        test_prediction = model.predict(
            [X_test_lstm]
        )
    
    train_prediction = np.hstack(train_prediction)
    val_prediction = np.hstack(val_prediction)
    test_prediction = np.hstack(test_prediction)

    model.save(model_path)

    # path must be the folder where is the yaml file is (test_file) + /results/file_name.csv
    # this is for example: ./tests/mi_test.yaml -> ./tests/results/mi_test.csv
    # Construct the path for the CSV file
    test_file_dir = os.path.dirname(args.test_file)
    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    tests_csv_path = os.path.join(test_file_dir, "results", f"{test_file_name}.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(tests_csv_path), exist_ok=True)

    # Path to the directory where the binary results will be saved (same as the previous one, but one folder deeper)
    binary_results_dir = os.path.join(test_file_dir, "results", "binary_results", test_file_name)
    os.makedirs(binary_results_dir, exist_ok=True)
    # Path to the directory where the regression results will be saved (same as the previous one, but one folder deeper)
    regression_results_dir = os.path.join(test_file_dir, "results", "regression_results", test_file_name)
    os.makedirs(regression_results_dir, exist_ok=True)
    #Path to save the losses
    losses_dir = os.path.join(test_file_dir,"results","losses",test_file_name)
    os.makedirs(losses_dir, exist_ok=True)

    loss_final = dist_model.loss_final
    
    df_loss = pd.DataFrame(data=np.array([np.arange(len(loss_final['loss_train'])),loss_final['loss_train'],loss_final['loss_val']]).T, columns = ['epoch','loss_train','loss_val'])


    df_loss.to_csv(f'{losses_dir}/{seed}.csv',index=False)

    if binary:
        conf_matrix_train, accuracy_train, f1_train, precision_train, recall_train = compute_binary_metrics(train_prediction, labels_train[:,0], "train")
        conf_matrix_val, accuracy_val, f1_val, precision_val, recall_val = compute_binary_metrics(val_prediction, labels_val[:,0], "val")
        conf_matrix_test, accuracy_test, f1_test, precision_test, recall_test = compute_binary_metrics(test_prediction, labels_test[:,0], "test")

        tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_train, accuracy=accuracy_train, f1=f1_train, precision=precision_train, recall=recall_train, label="train")
        tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_val, accuracy=accuracy_val, f1=f1_val, precision=precision_val, recall=recall_val, label="val")
        tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_test, accuracy=accuracy_test, f1=f1_test, precision=precision_test, recall=recall_test, label="test")

        tests_csv.to_csv(tests_csv_path, index=True)

        # Save binary results
        df_train_binary = save_results(id_train, stations_train, labels_train[:,0], train_prediction, prefix, test_name, seed, "train", "binary_classifier")
        df_val_binary = save_results(id_val, stations_val, labels_val[:,0], val_prediction, prefix, test_name, seed, "val", "binary_classifier")
        df_test_binary = save_results(id_test, stations_test, labels_test[:,0], test_prediction, prefix, test_name, seed, "test", "binary_classifier")

        df_train_binary_path = os.path.join(binary_results_dir, f"{seed}_train_binary.csv")
        df_val_binary_path = os.path.join(binary_results_dir, f"{seed}_val_binary.csv")
        df_test_binary_path = os.path.join(binary_results_dir, f"{seed}_test_binary.csv")

        df_train_binary.to_csv(df_train_binary_path, index=False)
        df_val_binary.to_csv(df_val_binary_path, index=False)
        df_test_binary.to_csv(df_test_binary_path, index=False)


    else:
        test_target = labels_test*max_target
        val_target = labels_val*max_target
        train_target = labels_train*max_target

        test_prediction = test_prediction*max_target
        val_prediction = val_prediction*max_target
        train_prediction = train_prediction*max_target

        mape_val = np.mean(np.abs(val_target-val_prediction)/val_target)
        mape_test = np.mean(np.abs(test_target-test_prediction)/test_target)
        mape_train = np.mean(np.abs(train_target-train_prediction)/train_target)

        mae_val = np.mean(np.abs(val_target-val_prediction))
        mae_test = np.mean(np.abs(test_target-test_prediction))
        mae_train = np.mean(np.abs(train_target-train_prediction))
        
        std_val = np.std(np.abs(val_target-val_prediction))
        std_test = np.std(np.abs(test_target-test_prediction))
        std_train = np.std(np.abs(train_target-train_prediction))
        
        confidence_interval = 1.645 # 90%
        
        events_with_err_above_val_std = np.sum(np.abs(val_target-val_prediction) >  std_val*confidence_interval)
        events_with_err_above_test_std = np.sum(np.abs(test_target-test_prediction) > std_test*confidence_interval)
        events_with_err_above_train_std = np.sum(np.abs(train_target-train_prediction) > std_train*confidence_interval)
        
        max_error_val = np.max(np.abs(val_target-val_prediction))
        max_error_test = np.max(np.abs(test_target-test_prediction))
        max_error_train = np.max(np.abs(train_target-train_prediction))
        
        print('Error MAPE sobre validacion: ',mape_val)
        print('Error MAPE corregido sobre test: ',mape_test)
        print('Error MAPE corregido sobre train: ',mape_train)

        tests_csv = save_target_metrics(tests_csv = tests_csv, seed = seed, mape = mape_train, mae = mae_train, std= std_train, events_with_err_above_std = events_with_err_above_train_std, max_error = max_error_train, label = "train")
        tests_csv = save_target_metrics(tests_csv = tests_csv, seed = seed, mape = mape_val,   mae = mae_val,   std = std_val,  events_with_err_above_std = events_with_err_above_val_std,   max_error = max_error_val, label = "val")
        tests_csv = save_target_metrics(tests_csv = tests_csv, seed = seed, mape = mape_test,  mae = mae_test,  std = std_test, events_with_err_above_std = events_with_err_above_test_std,  max_error = max_error_test, label = "test")

        tests_csv.to_csv(tests_csv_path, index=True)

        df_train_distance = save_results(id_train, stations_train, train_target, train_prediction, prefix, test_name, seed, "train", "distance")
        df_val_distance = save_results(id_val, stations_val, val_target, val_prediction, prefix, test_name, seed, "val", "distance")
        df_test_distance = save_results(id_test, stations_test, test_target, test_prediction, prefix, test_name, seed, "test", "distance")

        df_train_distance_path = os.path.join(regression_results_dir, f"{seed}_train_distance.csv")
        df_val_distance_path = os.path.join(regression_results_dir, f"{seed}_val_distance.csv")
        df_test_distance_path = os.path.join(regression_results_dir, f"{seed}_test_distance.csv")

        df_train_distance.to_csv(df_train_distance_path, index=False)
        df_val_distance.to_csv(df_val_distance_path, index=False)
        df_test_distance.to_csv(df_test_distance_path, index=False)



