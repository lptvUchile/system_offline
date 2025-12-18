from src.binary_classifier import BinaryClassModel
from src.back_azimuth.train_features_extraction import (
    function_features_extraction as function_features_extraction_baz,
)

import numpy as np
import time
from tensorflow.python.keras import backend as K
import tensorflow as tf
import os
import pandas as pd
from src.utils import to_angle, estimar_error_abs, sec_div_max
from sklearn.metrics import confusion_matrix

from src.utils import MyBatchGenerator_lstm_mlp, compute_binary_metrics, save_binary_metrics, save_results, pad_and_convert_to_tensor
from src.distance.train_features_extraction import (
    function_features_extraction as function_features_extraction_dist,
)
from src.multitask_learning.train_features_extraction import (
    function_features_extraction as function_features_extraction_multitask,
)

import argparse


parser = argparse.ArgumentParser(prog="Distance Training")
parser.add_argument('--test_file', type=str)
parser.add_argument('--feat_type', type=str)
parser.add_argument('--use_accel', action=argparse.BooleanOptionalAction)
parser.add_argument('--global_features', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--response', type=str, default='VEL')
parser.add_argument('--extract_features', action=argparse.BooleanOptionalAction)
parser.add_argument("--binary", action=argparse.BooleanOptionalAction)
parser.add_argument("--window_limit_dist", type=list, default=[20, 120])
parser.add_argument("--window_limit_baz", type=list, default=[1, 3])
parser.add_argument("--umbral_corte", type=float, default=0.03)
parser.add_argument("--one_hot_encoding", action=argparse.BooleanOptionalAction)

args = parser.parse_args()
test_file = args.test_file

# tests_csv_path = args.tests_csv
# feat_type = args.feat_type
# test = args.test
# tests_csv = pd.read_csv(tests_csv_path)

# load yaml
import yaml

with open(test_file, 'r') as file:
    test_config = yaml.safe_load(file)

# Extract values from the loaded YAML
feat_type = test_config.get('feature', args.feat_type)
#binary = test_config.get('binary', args.binary)
window_limit_dist = test_config.get('window_limit_dist', args.window_limit_dist)
window_limit_baz = test_config.get('window_limit_baz', args.window_limit_baz)
seeds = test_config.get('seeds', [args.seed])
use_accel = test_config.get('accel', args.use_accel)
global_features = test_config.get('global_feat', args.global_features)
test = args.test
response = test_config.get('response', args.response)
filter_baz = test_config.get('filter_baz', None)
filter_dist = test_config.get('filter_dist', None)
umbral_corte = test_config.get('umbral_corte', args.umbral_corte)
dummy = test_config.get('dummy', False)
lr = test_config.get('lr', 0.00015)
epochs = test_config.get('epochs', 1000)
patience = test_config.get('patience', 50)
batch_size = test_config.get('batch_size', 1)
use_cnn_model = test_config.get('use_cnn_model', True)
use_cnn_features = test_config.get('use_cnn_features', False)
regression = test_config.get('regression', False)
coast = test_config.get('coast', False)
left_right = test_config.get('left_right', False)
softmax = test_config.get('softmax', False)
one_hot_encoding = test_config.get('one_hot_encoding', False)

print("--- PARÁMETROS ---")
print(f"feat_type: {feat_type}")
#print(f"binary: {binary}")
print(f"window_limit_dist: {window_limit_dist}")
print(f"window_limit_baz: {window_limit_baz}")
print(f"seeds: {seeds}")
print(f"use_accel: {use_accel}")
print(f"global_features: {global_features}")
print(f"test: {test}")
print(f"response: {response}")
print(f"filter_baz: {filter_baz}")
print(f"filter_dist: {filter_dist}")
print(f"umbral_corte: {umbral_corte}")
print(f"dummy: {dummy}")
print(f"lr: {lr}")
print(f"epochs: {epochs}")
print(f"patience: {patience}")
print(f"batch_size: {batch_size}")
print(f'use_cnn_model: {use_cnn_model}')
print(f'use_cnn_features: {use_cnn_features}')
print(f"regression: {regression}")
print(f"coast: {coast}")
print(f"left_right: {left_right}")
print(f"one_hot_encoding: {one_hot_encoding}")

prefix = "acc" if use_accel else "vel"
test_name = test_file.split("/")[-1].split(".")[
    0
]
test_folder_name = test_file.split("/")[-2]
print(test_file)
print(test_name)
time.sleep(2)

options = {
    "default": [1, 3],
    "v2": [1, 4],
    "v3": [1, 5],
    "v4": [1, 6],
    "v5": [1, 7],
}
path_ind_random_train = './data/set/acc/train_20220927_mayores4M.csv'
# load the dataframe
df_train = pd.read_csv(path_ind_random_train, delimiter=",", index_col=False)
len_train = len(df_train['Evento'].values)
ind_random_train = np.random.permutation(np.arange(0, len_train))

path_ind_random_val = './data/set/acc/val_20220927_mayores4M.csv'
# load the dataframe
df_val = pd.read_csv(path_ind_random_val, delimiter=",", index_col=False)
len_val = len(df_val['Evento'].values)
ind_random_val = np.random.permutation(np.arange(0, len_val))

path_ind_random_test = './data/set/acc/test_20220927_mayores4M.csv'
# load the dataframe
df_test = pd.read_csv(path_ind_random_test, delimiter=",", index_col=False)
len_test = len(df_test['Evento'].values)
ind_random_test = np.random.permutation(np.arange(0, len_test))
time.sleep(2)

### Extract feature
if args.extract_features:
    tf.keras.utils.set_random_seed(1)
    function_features_extraction_baz(BDtype=feat_type, test = test, use_accel = use_accel, response = response, 
                                     test_name = test_name, test_folder_name = test_folder_name, options = window_limit_baz, 
                                     filter = filter_baz, ind_random_train = ind_random_train, ind_random_val = ind_random_val, 
                                     ind_random_test = ind_random_test, one_hot_encoding=one_hot_encoding)
    function_features_extraction_dist(BDtype = feat_type, test = test, use_accel = use_accel, response = response, 
                                      window_limit = window_limit_dist, umbral_corte = umbral_corte, filter= filter_dist,
                                      test_name = test_name, test_folder_name = test_folder_name,
                                      ind_random_train = ind_random_train, ind_random_val = ind_random_val, 
                                      ind_random_test = ind_random_test, use_cnn = use_cnn_features, one_hot_encoding=one_hot_encoding)


##############################################
## Load features for DISTANCE
##############################################
# define paths
path_feat_in_train_lstm_dist = (
    f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_train_mlp_dist = (
    f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_lstm_dist = (
    f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_mlp_dist = (
    f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_lstm_dist = (
    f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_lstm_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_mlp_dist = (
    f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)

# load files
feat_in_train_lstm_dist = np.load(path_feat_in_train_lstm_dist, allow_pickle=True)
feat_in_train_mlp_dist = np.load(path_feat_in_train_mlp_dist, allow_pickle=True)


feat_in_val_lstm_dist = np.load(path_feat_in_val_lstm_dist, allow_pickle=True)
feat_in_val_mlp_dist = np.load(path_feat_in_val_mlp_dist, allow_pickle=True)


feat_in_test_lstm_dist = np.load(path_feat_in_test_lstm_dist, allow_pickle=True)
feat_in_test_mlp_dist = np.load(path_feat_in_test_mlp_dist, allow_pickle=True)

feat_in_train_mlp_dist = np.array([feat_in_train_mlp_dist[i] for i in range(len(feat_in_train_mlp_dist))],dtype='float32')
feat_in_val_mlp_dist = np.array([feat_in_val_mlp_dist[i] for i in range(len(feat_in_val_mlp_dist))],dtype='float32')
feat_in_test_mlp_dist = np.array([feat_in_test_mlp_dist[i] for i in range(len(feat_in_test_mlp_dist))],dtype='float32')


# use_cnn_features = True implica que se pasa la señal al modelo, no los features FFT y energía. 
if use_cnn_features:
    feat_norm_train_lstm_dist = sec_div_max(feat_in_train_lstm_dist)
    feat_norm_val_lstm_dist = sec_div_max(feat_in_val_lstm_dist)
    feat_norm_test_lstm_dist = sec_div_max(feat_in_test_lstm_dist)


else:    
    # spectral features are normalized through min-max normalization
    min_f_train_lstm_dist = np.min([np.min(x, 0) for x in feat_in_train_lstm_dist], 0)
    max_f_train_lstm_dist = np.max([np.max(x, 0) for x in feat_in_train_lstm_dist], 0)

    feat_norm_train_lstm_dist = np.array([(feat_in_train_lstm_dist[i]-min_f_train_lstm_dist)/(max_f_train_lstm_dist-min_f_train_lstm_dist)
                                    for i in range(len(feat_in_train_lstm_dist))],dtype=object)

    feat_norm_val_lstm_dist = np.array([(feat_in_val_lstm_dist[i]-min_f_train_lstm_dist)/(max_f_train_lstm_dist-min_f_train_lstm_dist)
                                    for i in range(len(feat_in_val_lstm_dist))],dtype=object)
    feat_norm_test_lstm_dist = np.array([(feat_in_test_lstm_dist[i]-min_f_train_lstm_dist)/(max_f_train_lstm_dist-min_f_train_lstm_dist)
                                    for i in range(len(feat_in_test_lstm_dist))],dtype=object)

    
    min_f_train_global_dist = np.min(feat_in_train_mlp_dist,0)
    max_f_train_global_dist =  np.max(feat_in_train_mlp_dist,0)

    if not one_hot_encoding:
        feat_norm_train_global_dist = np.array([(feat_in_train_mlp_dist[i]-min_f_train_global_dist)/(max_f_train_global_dist-min_f_train_global_dist)
                                    for i in range(len(feat_in_train_mlp_dist))],dtype=object)
        feat_norm_val_global_dist = np.array([(feat_in_val_mlp_dist[i]-min_f_train_global_dist)/(max_f_train_global_dist-min_f_train_global_dist)
                                    for i in range(len(feat_in_val_mlp_dist))],dtype=object)
        feat_norm_test_global_dist = np.array([(feat_in_test_mlp_dist[i]-min_f_train_global_dist)/(max_f_train_global_dist-min_f_train_global_dist)
                                    for i in range(len(feat_in_test_mlp_dist))],dtype=object)


    os.makedirs(f'./models/binary_classifier/{prefix}/{test_name}/', exist_ok=True)
    os.makedirs(f'./data/features/binary_classifier/{prefix}/{test_folder_name}/{test_name}/', exist_ok=True)

    np.save(f'./data/features/binary_classifier/{prefix}/{test_folder_name}/{test_name}/min_temporal_dist.npy',min_f_train_lstm_dist)
    np.save(f'./data/features/binary_classifier/{prefix}/{test_folder_name}/{test_name}/max_temporal_dist.npy',max_f_train_lstm_dist)
    np.save(f'./data/features/binary_classifier/{prefix}/{test_folder_name}/{test_name}/min_global_dist.npy',min_f_train_global_dist)
    np.save(f'./data/features/binary_classifier/{prefix}/{test_folder_name}/{test_name}/max_global.npy',max_f_train_global_dist)

largo_cota_dist = None  # feat_in_train_lstm.shape[1]
tam_feat_mlp_dist = feat_in_train_mlp_dist.shape[1]  #### 49
tam_feat_lstm_dist = feat_norm_train_lstm_dist[0].shape[1]


##############################################
## Load features for BACK AZIMUTH
##############################################
path_feat_in_train_lstm_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/feat_cnn_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_train_mlp_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)

path_feat_in_val_lstm_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/feat_cnn_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_mlp_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)

path_feat_in_test_lstm_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/feat_cnn_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_mlp_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/feat_mlp_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)


feat_in_train_lstm_baz = np.load(path_feat_in_train_lstm_baz, allow_pickle=True)
feat_in_train_lstm_baz = feat_in_train_lstm_baz.astype("float")
feat_norm_train_lstm_baz = sec_div_max(feat_in_train_lstm_baz)
feat_in_train_mlp_baz = np.load(path_feat_in_train_mlp_baz, allow_pickle=True)


feat_in_val_lstm_baz = np.load(path_feat_in_val_lstm_baz, allow_pickle=True)
feat_in_val_lstm_baz = feat_in_val_lstm_baz.astype("float")
feat_norm_val_lstm_baz = sec_div_max(feat_in_val_lstm_baz)
feat_in_val_mlp_baz = np.load(path_feat_in_val_mlp_baz, allow_pickle=True)
feat_in_val_mlp_baz = feat_in_val_mlp_baz

feat_in_test_lstm_baz = np.load(path_feat_in_test_lstm_baz, allow_pickle=True)
feat_in_test_lstm_baz = feat_in_test_lstm_baz.astype("float")
feat_norm_test_lstm_baz = sec_div_max(feat_in_test_lstm_baz)
feat_in_test_mlp_baz = np.load(path_feat_in_test_mlp_baz, allow_pickle=True)
feat_in_test_mlp_baz = feat_in_test_mlp_baz


feat_norm_train_mlp_baz = np.array(
    [feat_in_train_mlp_baz[i] for i in range(len(feat_in_train_mlp_baz))], dtype=object
)
feat_norm_val_mlp = np.array(
    [feat_in_val_mlp_baz[i] for i in range(len(feat_in_val_mlp_baz))], dtype=object
)
feat_norm_test_mlp = np.array(
    [feat_in_test_mlp_baz[i] for i in range(len(feat_in_test_mlp_baz))], dtype=object
)

if not one_hot_encoding:
    min_f_train_global_baz = np.min(feat_in_train_mlp_baz, 0)
    max_f_train_global_baz = np.max(feat_in_train_mlp_baz, 0)

    feat_norm_train_mlp_baz = np.array(
        [(feat_in_train_mlp_baz[i] - min_f_train_global_baz) / (max_f_train_global_baz - min_f_train_global_baz)
         for i in range(len(feat_in_train_mlp_baz))], dtype=object
    )
    feat_norm_val_mlp_baz = np.array(
        [(feat_in_val_mlp_baz[i] - min_f_train_global_baz) / (max_f_train_global_baz - min_f_train_global_baz)
         for i in range(len(feat_in_val_mlp_baz))], dtype=object
    )
    feat_norm_test_mlp_baz = np.array(
        [(feat_in_test_mlp_baz[i] - min_f_train_global_baz) / (max_f_train_global_baz - min_f_train_global_baz)
         for i in range(len(feat_in_test_mlp_baz))], dtype=object
    )

largo_cota_baz = feat_in_train_lstm_baz.shape[1]
tam_feat_mlp_baz = feat_norm_train_mlp_baz.shape[-1]
tam_feat_lstm_baz = feat_norm_test_lstm_baz.shape[-1]


##############################################
## Load labels
##############################################
distth_az_real_train = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/distance_raw_train_'+feat_type+'.npy'.replace('\\', '/')
dist_real_train = np.load(distth_az_real_train, allow_pickle=True)
dist_real_train = pd.DataFrame(dist_real_train[()])
id_train = dist_real_train['Evento']
stations_train = dist_real_train['Estacion']

path_dist_real_val = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/distance_raw_val_'+feat_type+'.npy'.replace('\\', '/')
dist_real_val = np.load(path_dist_real_val, allow_pickle=True)
dist_real_val =  pd.DataFrame(dist_real_val[()])
id_val = dist_real_val['Evento']
stations_val = dist_real_val['Estacion']

path_dist_real_test = f'./data/features/distance/{prefix}/{test_folder_name}/{test_name}/distance_raw_test_'+feat_type+'.npy'.replace('\\', '/')
dist_real_test = np.load(path_dist_real_test, allow_pickle=True)
dist_real_test =  pd.DataFrame(dist_real_test[()])
id_test = dist_real_test['Evento']
stations_test = dist_real_test['Estacion']

##############################################
## Define type of classification
##############################################
if coast:
    tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
    labels_train = dist_real_train["costero"].values
    labels_val = dist_real_val["costero"].values
    labels_test = dist_real_test["costero"].values

elif left_right:
    if softmax:
        tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
        labels_train_izquierda = dist_real_train["evento_izquierda_estacion"].values
        labels_train_derecha = dist_real_train["evento_derecha_estacion"].values
        labels_train = np.concatenate(
            (labels_train_izquierda.reshape(-1, 1), labels_train_derecha.reshape(-1, 1)),axis=1
        )
        #labels_train = np.array(list(zip(labels_train_izquierda, labels_train_derecha)))
        labels_val_izquierda = dist_real_val["evento_izquierda_estacion"].values
        labels_val_derecha = dist_real_val["evento_derecha_estacion"].values
        #labels_val = np.array(list(zip(labels_val_izquierda, labels_val_derecha)))
        labels_val = np.concatenate(
            (labels_val_izquierda.reshape(-1, 1), labels_val_derecha.reshape(-1, 1)),axis=1
        )
        labels_test_izquierda = dist_real_test["evento_izquierda_estacion"].values
        labels_test_derecha = dist_real_test["evento_derecha_estacion"].values
        #labels_test = np.array(list(zip(labels_test_izquierda, labels_test_derecha)))
        labels_test = np.concatenate(
            (labels_test_izquierda.reshape(-1, 1), labels_test_derecha.reshape(-1, 1)),axis=1
        )
        tests_csv.set_index("seed", inplace=True)
    else: 
        tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
        labels_train_izquierda = dist_real_train["evento_izquierda_estacion"].values
        labels_train = labels_train_izquierda.reshape(-1, 1)

        labels_val_izquierda = dist_real_val["evento_izquierda_estacion"].values
        labels_val = labels_val_izquierda.reshape(-1, 1)

        labels_test_izquierda = dist_real_test["evento_izquierda_estacion"].values
        labels_test = labels_test_izquierda.reshape(-1, 1)

##############################################
## Train and test
##############################################

for seed in seeds:    
    tf.keras.utils.set_random_seed(seed)
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 1, "CPU": 1})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    repeats = 1  # number of repetitions
    nro_output = 1
    K.set_session(sess)
    path_root = os.getcwd()

    (
        y_estimada_repeat_train,
        error_repeat_train_mse,
        error_repeat_train_rel,
        loss_repeat_train,
    ) = ([], [], [], [])
    (
        y_estimada_repeat_val,
        error_repeat_val_mse,
        error_repeat_val_rel,
        loss_repeat_val,
    ) = ([], [], [], [])
    y_estimada_repeat_test, error_repeat_test_mse, error_repeat_test_rel = [], [], []

    binary_class_model = BinaryClassModel(
        tam_feat_mlp_dist = tam_feat_mlp_dist,
        tam_feat_lstm_dist = tam_feat_lstm_dist,
        largo_cota_dist = largo_cota_dist,
        tam_feat_mlp_baz = tam_feat_mlp_baz,
        tam_feat_lstm_baz = tam_feat_lstm_baz,
        largo_cota_baz = largo_cota_baz,
        batch_size=batch_size,
        global_feat=global_features,
        use_cnn=use_cnn_model,
        coast=coast,
        softmax=softmax
    )

    binary_class_model.setup_model()

    model_path = f"./models/binary_classifier/{prefix}/{test_folder_name}/{test_name}/{seed}.h5"

    if softmax:
        binary_class_model.setup_training(model_path = model_path, dummy=dummy, loss="categorical_crossentropy", lr=lr, patience=patience)
    else:
        binary_class_model.setup_training(model_path = model_path, dummy=dummy, loss="binary_crossentropy", lr=lr, patience=patience)

    ## Two test to do. Using DIST global features or BAZ global features.
    binary_class_model.input_training(
        feat_norm_train_lstm_dist = feat_norm_train_lstm_dist,
        feat_in_train_mlp_dist = feat_in_train_mlp_dist if one_hot_encoding else feat_norm_train_global_dist,
        feat_norm_train_lstm_baz = feat_norm_train_lstm_baz,
        labels_train = labels_train,
    )

    binary_class_model.input_validation(
        feat_norm_val_lstm_dist = feat_norm_val_lstm_dist,
        feat_in_val_mlp_dist = feat_in_val_mlp_dist if one_hot_encoding else feat_norm_val_global_dist,
        feat_norm_val_lstm_baz = feat_norm_val_lstm_baz,
        labels_val = labels_val,
    )

    binary_class_model.train()
    model = binary_class_model.model

    # Predict
    y_train, y_val, y_test = labels_train, labels_val, labels_test

    X_train_lstm_dist, X_train_mlp_dist = (
        feat_norm_train_lstm_dist,
        feat_in_train_mlp_dist if one_hot_encoding else feat_norm_train_global_dist,
    )
    X_val_lstm_dist, X_val_mlp_dist = feat_norm_val_lstm_dist, feat_in_val_mlp_dist if one_hot_encoding else feat_norm_val_global_dist
    X_test_lstm_dist, X_test_mlp_dist = feat_norm_test_lstm_dist, feat_in_test_mlp_dist if one_hot_encoding else feat_norm_test_global_dist

    X_train_lstm_dist = pad_and_convert_to_tensor(feat_norm_train_lstm_dist)
    X_val_lstm_dist = pad_and_convert_to_tensor(feat_norm_val_lstm_dist)
    X_test_lstm_dist = pad_and_convert_to_tensor(feat_norm_test_lstm_dist)



    if global_features:
        train_prediction = model.predict(
            [   X_train_lstm_dist,
                #tf.convert_to_tensor(feat_norm_train_lstm_dist.astype("float32")),
                tf.convert_to_tensor(feat_norm_train_lstm_baz.astype("float32")),
                tf.convert_to_tensor(X_train_mlp_dist.astype("float32")),
            ]
        )[:,0]
        val_prediction = model.predict(
            [   X_val_lstm_dist,
                #tf.convert_to_tensor(feat_norm_val_lstm_dist.astype("float32")),
                tf.convert_to_tensor(feat_norm_val_lstm_baz.astype("float32")),
                tf.convert_to_tensor(X_val_mlp_dist.astype("float32")),
            ]
        )[:,0]
        test_prediction = model.predict(
            [   X_test_lstm_dist,
                #tf.convert_to_tensor(feat_norm_test_lstm_dist.astype("float32")),
                tf.convert_to_tensor(feat_norm_test_lstm_baz.astype("float32")),
                tf.convert_to_tensor(X_test_mlp_dist.astype("float32")),
            ]
        )[:,0]
    else:
        train_prediction = model.predict(
            X_train_lstm_dist,
            [tf.convert_to_tensor(feat_norm_train_lstm_baz.astype("float32"))],
        )
        val_prediction = model.predict(
            X_val_lstm_dist,
            [tf.convert_to_tensor(feat_norm_val_lstm_baz.astype("float32"))],
        )
        test_prediction = model.predict(
            X_test_lstm_dist[
                tf.convert_to_tensor(feat_norm_test_lstm_baz.astype("float32"))
            ]
        )
    # do a binary matrix with the predictions
    train_prediction = np.hstack(train_prediction)
    val_prediction = np.hstack(val_prediction)
    test_prediction = np.hstack(test_prediction)

    train_target = y_train
    val_target = y_val
    test_target = y_test

    conf_matrix_train, accuracy_train, f1_train, precision_train, recall_train = compute_binary_metrics(train_prediction, train_target[:,0], "train")
    conf_matrix_val, accuracy_val, f1_val, precision_val, recall_val = compute_binary_metrics(val_prediction, val_target[:,0], "val")
    conf_matrix_test, accuracy_test, f1_test, precision_test, recall_test = compute_binary_metrics(test_prediction, test_target[:,0], "test")

    tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_train, accuracy=accuracy_train, f1=f1_train, precision=precision_train, recall=recall_train, label="train")
    tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_val, accuracy=accuracy_val, f1=f1_val, precision=precision_val, recall=recall_val, label="val")
    tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_test, accuracy=accuracy_test, f1=f1_test, precision=precision_test, recall=recall_test, label="test")

   
   # path must be the folder where is the yaml file is (test_file) + /results/file_name.csv
    # this is for example: ./tests/mi_test.yaml -> ./tests/results/mi_test.csv
    # Construct the path for the CSV file
    test_file_dir = os.path.dirname(args.test_file)
    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    tests_csv_path = os.path.join(test_file_dir, "results", f"{test_file_name}.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(tests_csv_path), exist_ok=True)
    tests_csv.to_csv(tests_csv_path, index=True)

    # save the model
    model.save(model_path)


    df_train_binary = save_results(id_train, stations_train, train_target[:,0], train_prediction, prefix, test_name, seed, "train", "binary_classifier")
    df_val_binary = save_results(id_val, stations_val, val_target[:,0], val_prediction, prefix, test_name, seed, "val", "binary_classifier")
    df_test_binary = save_results(id_test, stations_test, test_target[:,0], test_prediction, prefix, test_name, seed, "test", "binary_classifier")

    # Path to the directory where the binary results will be saved (same as the previous one, but one folder deeper)
    binary_results_dir = os.path.join(test_file_dir, "results", "binary_results", test_file_name)
    os.makedirs(binary_results_dir, exist_ok=True)

    df_train_binary_path = os.path.join(binary_results_dir, f"{seed}_train_binary.csv")
    df_val_binary_path = os.path.join(binary_results_dir, f"{seed}_val_binary.csv")
    df_test_binary_path = os.path.join(binary_results_dir, f"{seed}_test_binary.csv")

    df_train_binary.to_csv(df_train_binary_path, index=False)
    df_val_binary.to_csv(df_val_binary_path, index=False)
    df_test_binary.to_csv(df_test_binary_path, index=False)
