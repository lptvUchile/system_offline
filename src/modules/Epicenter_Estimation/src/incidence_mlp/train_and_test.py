import numpy as np
import time
from tensorflow.python.keras import backend as K
import tensorflow as tf
import os
import pandas as pd
from src.utils import (
                       save_results, save_target_metrics, sec_div_max)
from src.incidence_mlp import IncidenceMLPModel
from src.incidence_mlp.train_features_extraction import function_features_extraction


import argparse

parser = argparse.ArgumentParser(prog='Incidence Training')
parser.add_argument('--test_file', type=str)
parser.add_argument('--feat_type', type=str)
parser.add_argument('--use_accel', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--response', type=str, default='VEL')
parser.add_argument('--extract_features', action=argparse.BooleanOptionalAction)
parser.add_argument("--window_limit", type=list, default=[20, 120])


args = parser.parse_args()
test_file = args.test_file

# load yaml
import yaml

with open(test_file, 'r') as file:
    test_config = yaml.safe_load(file)

# Extract values from the loaded YAML
feat_type = test_config.get('feature', args.feat_type)
window_limit = test_config.get('window_limit', args.window_limit)
seeds = test_config.get('seeds', [args.seed])
use_accel = test_config.get('accel', args.use_accel)
test = args.test
response = test_config.get('response', args.response)
filter = test_config.get('filter', None)
dummy = test_config.get('dummy', False)
lr = test_config.get('lr', 0.00015)
epochs = test_config.get('epochs', 1000)
patience = test_config.get('patience', 50)
batch_size = test_config.get('batch_size', 16)
use_horizontal_mean = test_config.get('use_horizontal_mean', False)


print("--- PARÁMETROS ---")
print(f"feat_type: {feat_type}")
print(f"window_limit: {window_limit}")
print(f"seeds: {seeds}")
print(f"use_accel: {use_accel}")
print(f"test: {test}")
print(f"response: {response}")
print(f"filter: {filter}")
print(f"dummy: {dummy}")
print(f"lr: {lr}")
print(f"epochs: {epochs}")
print(f"patience: {patience}")
print(f"batch_size: {batch_size}")
print(f"use_horizontal_mean: {use_horizontal_mean}")

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
        filter = filter, 
        test_name = test_name, 
        test_folder_name = test_folder_name,
        use_horizontal_mean = use_horizontal_mean
        )
    

start_time = time.time()
path_root =  os.getcwd()

path_feat_in_mlp_train = f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_train_'+feat_type+'.npy'.replace('\\', '/')
label_real_train      = f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/incidence_raw_train_'+feat_type+'.npy'.replace('\\', '/')

path_feat_in_mlp_val = f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_val_'+feat_type+'.npy'.replace('\\', '/')
label_real_val      = f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/incidence_raw_val_'+feat_type+'.npy'.replace('\\', '/')

path_feat_in_mlp_test = f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_test_'+feat_type+'.npy'.replace('\\', '/')
label_real_test      = f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/incidence_raw_test_'+feat_type+'.npy'.replace('\\', '/')


feat_in_mlp_train = np.load(path_feat_in_mlp_train, allow_pickle=True)
feat_in_mlp_train = feat_in_mlp_train.astype("float")
feat_norm_mlp_train = sec_div_max(feat_in_mlp_train)
label_train = np.load(label_real_train, allow_pickle=True)
label_train = pd.DataFrame(label_train[()])
print(label_train.head())
id_train = label_train['event']
stations_train = label_train['station']


feat_in_mlp_val = np.load(path_feat_in_mlp_val, allow_pickle=True)
feat_in_mlp_val = feat_in_mlp_val.astype("float")
feat_norm_mlp_val = sec_div_max(feat_in_mlp_val)
label_val = np.load(label_real_val, allow_pickle=True)
label_val =  pd.DataFrame(label_val[()])
id_val = label_val['event']
stations_val = label_val['station']

feat_in_mlp_test = np.load(path_feat_in_mlp_test, allow_pickle=True)
feat_in_mlp_test = feat_in_mlp_test.astype("float")
feat_norm_mlp_test = sec_div_max(feat_in_mlp_test)
label_test = np.load(label_real_test, allow_pickle=True)
label_test =  pd.DataFrame(label_test[()])
id_test = label_test['event']
stations_test = label_test['station']



feat_norm_mlp_train = np.array(
    [feat_norm_mlp_train[i] for i in range(len(feat_norm_mlp_train))], dtype=object
)
feat_norm_mlp_val = np.array(
    [feat_norm_mlp_val[i] for i in range(len(feat_norm_mlp_val))], dtype=object
)
feat_norm_mlp_test = np.array(
    [feat_norm_mlp_test[i] for i in range(len(feat_norm_mlp_test))], dtype=object
)



tam_feat_mlp = feat_norm_mlp_train.shape[1] # shape (nro_eventos, nro_features) deberia dar 480
print("TAM_FEAT_MLP: ", tam_feat_mlp)



tests_csv = pd.DataFrame(columns=["seed", "mae_train", "mae_val", "mae_test"])
tests_csv.set_index("seed", inplace=True)

label_real_train = label_train['incidence']
label_real_val = label_val['incidence']
label_real_test = label_test['incidence']

labels_train = label_real_train
labels_val = label_real_val
labels_test = label_real_test

# Save temporal and global features
os.makedirs(f'./models/incidence_mlp/{prefix}/{test_name}/', exist_ok=True)
os.makedirs(f'./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/', exist_ok=True)



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

    incidence_mlp_model = IncidenceMLPModel()
    incidence_mlp_model.setup_model(tam_feat_mlp=tam_feat_mlp, nro_output=nro_output)

    os.makedirs(f"./models/incidence_mlp/{prefix}/{test_name}", exist_ok=True)
    model_path = f"./models/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/{seed}.h5"


    incidence_mlp_model.setup_training(model_path=model_path, loss="mean_squared_error", dummy=dummy, patience=patience, epochs=epochs, lr=lr)
    
    incidence_mlp_model.input_training(feat_norm_mlp_train, labels_train)
    incidence_mlp_model.input_validation(feat_norm_mlp_val, labels_val)
    incidence_mlp_model.train()
    model = incidence_mlp_model.model


    X_train_mlp = tf.convert_to_tensor(feat_norm_mlp_train, dtype=tf.float32)
    X_val_mlp = tf.convert_to_tensor(feat_norm_mlp_val, dtype=tf.float32)
    X_test_mlp = tf.convert_to_tensor(feat_norm_mlp_test, dtype=tf.float32)


    train_prediction = model.predict([X_train_mlp])
    val_prediction = model.predict([X_val_mlp])
    test_prediction = model.predict([X_test_mlp])
        
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

    loss_final = incidence_mlp_model.loss_final
    
    df_loss = pd.DataFrame(data=np.array([np.arange(len(loss_final['loss_train'])),loss_final['loss_train'],loss_final['loss_val']]).T, columns = ['epoch','loss_train','loss_val'])


    df_loss.to_csv(f'{losses_dir}/{seed}.csv',index=False)




    test_target = labels_test
    val_target = labels_val
    train_target = labels_train
    
    print(X_test_mlp)
    print(test_prediction)

    test_prediction = np.squeeze(test_prediction)
    val_prediction = np.squeeze(val_prediction)
    train_prediction = np.squeeze(train_prediction)
    
    test_prediction_angle = np.arcsin(test_prediction)*180/np.pi
    val_prediction_angle = np.arcsin(val_prediction)*180/np.pi
    train_prediction_angle = np.arcsin(train_prediction)*180/np.pi
    
    test_target_angle = np.arcsin(test_target)*180/np.pi
    val_target_angle = np.arcsin(val_target)*180/np.pi
    train_target_angle = np.arcsin(train_target)*180/np.pi
    

    mape_val = np.mean(np.abs(val_target_angle-val_prediction_angle)/val_target_angle)
    mape_test = np.mean(np.abs(test_target_angle-test_prediction_angle)/test_target_angle)
    mape_train = np.mean(np.abs(train_target_angle-train_prediction_angle)/train_target_angle)

    mae_val = np.mean(np.abs(val_target_angle-val_prediction_angle))
    mae_test = np.mean(np.abs(test_target_angle-test_prediction_angle))
    mae_train = np.mean(np.abs(train_target_angle-train_prediction_angle))
    
    std_val = np.std(np.abs(val_target_angle-val_prediction_angle))
    std_test = np.std(np.abs(test_target_angle-test_prediction_angle))
    std_train = np.std(np.abs(train_target_angle-train_prediction_angle))
    
    confidence_interval = 1.645 # 90%
    
    events_with_err_above_val_std = np.sum(np.abs(val_target_angle-val_prediction_angle) >  std_val*confidence_interval)
    events_with_err_above_test_std = np.sum(np.abs(test_target_angle-test_prediction_angle) > std_test*confidence_interval)
    events_with_err_above_train_std = np.sum(np.abs(train_target_angle-train_prediction_angle) > std_train*confidence_interval)
    
    max_error_val = np.max(np.abs(val_target_angle-val_prediction_angle))
    max_error_test = np.max(np.abs(test_target_angle-test_prediction_angle))
    max_error_train = np.max(np.abs(train_target_angle-train_prediction_angle))
    
    print('Error MAPE sobre validacion: ',mape_val)
    print('Error MAPE corregido sobre test: ',mape_test)
    print('Error MAPE corregido sobre train: ',mape_train)

    tests_csv = save_target_metrics(tests_csv = tests_csv, seed = seed, mape = mape_train, mae = mae_train, std= std_train, events_with_err_above_std = events_with_err_above_train_std, max_error = max_error_train, label = "train")
    tests_csv = save_target_metrics(tests_csv = tests_csv, seed = seed, mape = mape_val,   mae = mae_val,   std = std_val,  events_with_err_above_std = events_with_err_above_val_std,   max_error = max_error_val, label = "val")
    tests_csv = save_target_metrics(tests_csv = tests_csv, seed = seed, mape = mape_test,  mae = mae_test,  std = std_test, events_with_err_above_std = events_with_err_above_test_std,  max_error = max_error_test, label = "test")

    tests_csv.to_csv(tests_csv_path, index=True)

    df_train_incidence = save_results(id_train, stations_train, train_target, train_prediction, prefix, test_name, seed, "train", "incidence")
    df_val_incidence = save_results(id_val, stations_val, val_target, val_prediction, prefix, test_name, seed, "val", "incidence")
    df_test_incidence = save_results(id_test, stations_test, test_target, test_prediction, prefix, test_name, seed, "test", "incidence")

    df_train_incidence_path = os.path.join(regression_results_dir, f"{seed}_train_incidence.csv")
    df_val_incidence_path = os.path.join(regression_results_dir, f"{seed}_val_incidence.csv")
    df_test_incidence_path = os.path.join(regression_results_dir, f"{seed}_test_incidence.csv")

    df_train_incidence.to_csv(df_train_incidence_path, index=False)
    df_val_incidence.to_csv(df_val_incidence_path, index=False)
    df_test_incidence.to_csv(df_test_incidence_path, index=False)



