from src.multitask_learning import MultiTaskModel
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

from src.utils import MyBatchGenerator_lstm_mlp, compute_binary_metrics, save_binary_metrics, save_results, pad_and_convert_to_tensor, save_regression_metrics
from src.distance.train_features_extraction import (
    function_features_extraction as function_features_extraction_dist,
)

from src.multitask_learning.train_features_extraction import (
    function_features_extraction as function_features_extraction_multitask,
)
from src.multitask_learning.dynamic_weigth import DynamicWeightedLoss

from sklearn.metrics import mean_squared_error, mean_absolute_error

import argparse


parser = argparse.ArgumentParser(prog="Distance Training")
parser.add_argument('--test_file', type=str)
parser.add_argument('--feat_type', type=str)
parser.add_argument('--use_accel', action=argparse.BooleanOptionalAction)
parser.add_argument('--no_global_features', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--response', type=str, default='VEL')
parser.add_argument('--extract_features', action=argparse.BooleanOptionalAction)
#parser.add_argument("--binary", action=argparse.BooleanOptionalAction)
parser.add_argument("--window_limit_dist", type=list, default=[20, 120])
parser.add_argument("--window_limit_baz", type=list, default=[1, 3])
parser.add_argument("--umbral_corte", type=float, default=0.03)

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
no_global_features = not test_config.get('global_feat', not args.no_global_features)
test = args.test
response = test_config.get('response', args.response)
filter_baz = test_config.get('filter_baz', None)
filter_dist = test_config.get('filter_dist', None)
umbral_corte = test_config.get('umbral_corte', args.umbral_corte)
dummy = test_config.get('dummy', False)
lr = test_config.get('lr', 0.00015)
epochs = test_config.get('epochs', 1000)
patience = test_config.get('patience', 50)

print("--- PARÁMETROS ---")
print(f"feat_type: {feat_type}")
#print(f"binary: {binary}")
print(f"window_limit_dist: {window_limit_dist}")
print(f"window_limit_baz: {window_limit_baz}")
print(f"seeds: {seeds}")
print(f"use_accel: {use_accel}")
print(f"no_global_features: {no_global_features}")
print(f"test: {test}")
print(f"response: {response}")
print(f"filter_baz: {filter_baz}")
print(f"filter_dist: {filter_dist}")
print(f"umbral_corte: {umbral_corte}")
print(f"dummy: {dummy}")
print(f"lr: {lr}")
print(f"epochs: {epochs}")
print(f"patience: {patience}")

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
print(len_train, len_val, len_test)
time.sleep(5)

### Extract feature
if args.extract_features:
    tf.keras.utils.set_random_seed(1)
    function_features_extraction_baz(feat_type, test, use_accel, response, 
                                      test_name, window_limit_baz, filter_baz, ind_random_train, ind_random_val, ind_random_test)
    function_features_extraction_dist(feat_type, test, use_accel, response, 
                                       window_limit_dist, umbral_corte, filter_dist, test_name, ind_random_train, ind_random_val, ind_random_test)
    #function_features_extraction_multitask(feat_type, test, use_accel, response,
    #                                       window_limit_dist, window_limit_baz,
    #                                       umbral_corte, filter_dist, filter_baz,
    #                                       test_name, ind_random_train, ind_random_val, ind_random_test)
## Load features for DISTANCE
# define paths
path_feat_in_train_lstm_dist = (
    f"./data/features/distance/{prefix}/{test_name}/feat_lstm_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_train_mlp_dist = (
    f"./data/features/distance/{prefix}/{test_name}/feat_mlp_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_lstm_dist = (
    f"./data/features/distance/{prefix}/{test_name}/feat_lstm_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_mlp_dist = (
    f"./data/features/distance/{prefix}/{test_name}/feat_mlp_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_lstm_dist = (
    f"./data/features/distance/{prefix}/{test_name}/feat_lstm_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_mlp_dist = (
    f"./data/features/distance/{prefix}/{test_name}/feat_mlp_raw_test_"
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


feat_in_train_mlp_dist = np.array(
    [feat_in_train_mlp_dist[i] for i in range(len(feat_in_train_mlp_dist))],
    dtype="float32",
)
feat_in_val_mlp_dist = np.array(
    [feat_in_val_mlp_dist[i] for i in range(len(feat_in_val_mlp_dist))], dtype="float32"
)
feat_in_test_mlp_dist = np.array(
    [feat_in_test_mlp_dist[i] for i in range(len(feat_in_test_mlp_dist))],
    dtype="float32",
)

# spectral features are normalized through min-max normalization
min_f_train_lstm_dist = np.min([np.min(x, 0) for x in feat_in_train_lstm_dist], 0)
max_f_train_lstm_dist = np.max([np.max(x, 0) for x in feat_in_train_lstm_dist], 0)
feat_norm_train_lstm_dist = np.array(
    [
        (feat_in_train_lstm_dist[i] - min_f_train_lstm_dist)
        / (max_f_train_lstm_dist - min_f_train_lstm_dist)
        for i in range(len(feat_in_train_lstm_dist))
    ],
    dtype=object,
)
feat_norm_val_lstm_dist = np.array(
    [
        (feat_in_val_lstm_dist[i] - min_f_train_lstm_dist)
        / (max_f_train_lstm_dist - min_f_train_lstm_dist)
        for i in range(len(feat_in_val_lstm_dist))
    ],
    dtype=object,
)
feat_norm_test_lstm_dist = np.array(
    [
        (feat_in_test_lstm_dist[i] - min_f_train_lstm_dist)
        / (max_f_train_lstm_dist - min_f_train_lstm_dist)
        for i in range(len(feat_in_test_lstm_dist))
    ],
    dtype=object,
)

largo_cota_dist = None  # feat_in_train_lstm.shape[1]
tam_feat_mlp_dist = feat_in_train_mlp_dist.shape[1]  #### 49
tam_feat_lstm_dist = feat_norm_train_lstm_dist[0].shape[1]


## Load features for BACK AZIMUTH
path_feat_in_train_lstm_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_cnn_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_train_mlp_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_mlp_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)

path_feat_in_val_lstm_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_cnn_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_mlp_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_mlp_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)

path_feat_in_test_lstm_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_cnn_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_mlp_baz = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_mlp_raw_test_"
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


largo_cota_baz = feat_in_train_lstm_baz.shape[1]
tam_feat_mlp_baz = feat_norm_train_mlp_baz.shape[-1]
tam_feat_lstm_baz = feat_norm_test_lstm_baz.shape[-1]


distth_az_real_train = f'./data/features/distance/{prefix}/{test_name}/distance_raw_train_'+feat_type+'.npy'.replace('\\', '/')
dist_real_train = np.load(distth_az_real_train, allow_pickle=True)
dist_real_train = pd.DataFrame(dist_real_train[()])
id_train = dist_real_train['Evento']
stations_train = dist_real_train['Estacion']

path_dist_real_val = f'./data/features/distance/{prefix}/{test_name}/distance_raw_val_'+feat_type+'.npy'.replace('\\', '/')
dist_real_val = np.load(path_dist_real_val, allow_pickle=True)
dist_real_val =  pd.DataFrame(dist_real_val[()])
id_val = dist_real_val['Evento']
stations_val = dist_real_val['Estacion']

path_dist_real_test = f'./data/features/distance/{prefix}/{test_name}/distance_raw_test_'+feat_type+'.npy'.replace('\\', '/')
dist_real_test = np.load(path_dist_real_test, allow_pickle=True)
dist_real_test =  pd.DataFrame(dist_real_test[()])
id_test = dist_real_test['Evento']
stations_test = dist_real_test['Estacion']

#para multitask learning
labels_baz_train_path = f'./data/features/back_azimuth/{prefix}/{test_name}/angulo_raw_train_'+feat_type+'.npy'.replace('\\', '/')
labels_baz_train = np.load(labels_baz_train_path, allow_pickle=True)
baz_real_train = pd.DataFrame(labels_baz_train[()])

labels_baz_val_path = f'./data/features/back_azimuth/{prefix}/{test_name}/angulo_raw_val_'+feat_type+'.npy'.replace('\\', '/')
labels_baz_val = np.load(labels_baz_val_path, allow_pickle=True)
baz_real_val = pd.DataFrame(labels_baz_val[()])

labels_baz_test_path = f'./data/features/back_azimuth/{prefix}/{test_name}/angulo_raw_test_'+feat_type+'.npy'.replace('\\', '/')
labels_baz_test = np.load(labels_baz_test_path, allow_pickle=True)
baz_real_test = pd.DataFrame(labels_baz_test[()])

# Normalizacion de la etiqueta de distancia
dist_min_train = dist_real_train['distancia'].min()
dist_max_train  = dist_real_train['distancia'].max()
dist_abs_max_train = max(abs(dist_min_train), abs(dist_max_train))
labels_train_dist = dist_real_train['distancia'].copy().values/dist_abs_max_train 

dist_min_val = dist_real_val['distancia'].min()
dist_max_val  = dist_real_val['distancia'].max()
dist_abs_max_val = max(abs(dist_min_val), abs(dist_max_val))
labels_val_dist = dist_real_val['distancia'].copy().values/dist_abs_max_val

dist_min_test = dist_real_test['distancia'].min()
dist_max_test  = dist_real_test['distancia'].max()
dist_abs_max_test = max(abs(dist_min_test), abs(dist_max_test))
labels_test_dist = dist_real_test['distancia'].copy().values/dist_abs_max_test

labels_train = np.concatenate(
    (
    dist_real_train["costero"].values.reshape(-1, 1),
    #dist_real_train["distancia"].values.reshape(-1, 1),
    labels_train_dist.reshape(-1, 1),
    baz_real_train["Coseno"].values.reshape(-1, 1),
    baz_real_train["Seno"].values.reshape(-1, 1),
    ),
    axis=1
)


labels_val = np.concatenate(
    (
    dist_real_val["costero"].values.reshape(-1, 1),
    #dist_real_val["distancia"].values.reshape(-1, 1),
    labels_val_dist.reshape(-1, 1),
    baz_real_val["Coseno"].values.reshape(-1, 1),
    baz_real_val["Seno"].values.reshape(-1, 1),
    ),
    axis=1
)

labels_test = np.concatenate(
    (
    dist_real_test["costero"].values.reshape(-1, 1),
    #dist_real_test["distancia"].values.reshape(-1, 1),
    labels_test_dist.reshape(-1, 1),
    baz_real_test["Coseno"].values.reshape(-1, 1),
    baz_real_test["Seno"].values.reshape(-1, 1),
    ),
    axis=1
)


#if binary:
tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
#else:
# raise Exception("Binary is not implemented")
tests_csv.set_index("seed", inplace=True)

for seed in seeds:    
    tf.keras.utils.set_random_seed(seed)
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 1, "CPU": 1})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    repeats = 1  # number of repetitions
    nro_output = 1
    K.set_session(sess)
    path_root = os.getcwd()

    os.makedirs(f'./models/binary_classifier/{prefix}/{test_name}/', exist_ok=True)
    np.save(f'./models/binary_classifier/{prefix}/{test_name}/{seed}_min_params.npy',min_f_train_lstm_dist)
    np.save(f'./models/binary_classifier/{prefix}/{test_name}/{seed}_max_params.npy',max_f_train_lstm_dist)       

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

    binary_class_model = MultiTaskModel(
        tam_feat_mlp_dist,
        tam_feat_lstm_dist,
        largo_cota_dist,
        tam_feat_mlp_baz,
        tam_feat_lstm_baz,
        largo_cota_baz,
        global_feat=not args.no_global_features,
    )

    binary_class_model.setup_model()

    model_path = f"./models/binary_classifier/{prefix}/{test_folder_name}/{test_name}/{seed}.h5"

    binary_class_model.setup_training(model_path = model_path, dummy=dummy, loss = None, lr=lr, patience=patience, epochs = epochs)
    binary_class_model.input_training(
        feat_norm_train_lstm_dist,
        feat_in_train_mlp_dist,
        feat_norm_train_lstm_baz,
        labels_train,
    )
    binary_class_model.input_validation(
        feat_norm_val_lstm_dist,
        feat_in_val_mlp_dist,
        feat_norm_val_lstm_baz,
        labels_val,
    )
    binary_class_model.train()
    model = binary_class_model.model

    # Predict
    y_train, y_val, y_test = labels_train, labels_val, labels_test

    X_train_lstm_dist, X_train_mlp_dist = (
        feat_norm_train_lstm_dist,
        feat_in_train_mlp_dist,
    )
    X_val_lstm_dist, X_val_mlp_dist = feat_norm_val_lstm_dist, feat_in_val_mlp_dist
    X_test_lstm_dist, X_test_mlp_dist = feat_norm_test_lstm_dist, feat_in_test_mlp_dist

    X_train_lstm_dist = pad_and_convert_to_tensor(feat_norm_train_lstm_dist)
    X_val_lstm_dist = pad_and_convert_to_tensor(feat_norm_val_lstm_dist)
    X_test_lstm_dist = pad_and_convert_to_tensor(feat_norm_test_lstm_dist)



    if not args.no_global_features:
        train_prediction = model.predict(
            [   X_train_lstm_dist,
                tf.convert_to_tensor(feat_norm_train_lstm_baz.astype("float32")),
                tf.convert_to_tensor(X_train_mlp_dist.astype("float32")),
            ]
        )
        val_prediction = model.predict(
            [   X_val_lstm_dist,
                tf.convert_to_tensor(feat_norm_val_lstm_baz.astype("float32")),
                tf.convert_to_tensor(X_val_mlp_dist.astype("float32")),
            ]
        )
        test_prediction = model.predict(
            [   X_test_lstm_dist,
                tf.convert_to_tensor(feat_norm_test_lstm_baz.astype("float32")),
                tf.convert_to_tensor(X_test_mlp_dist.astype("float32")),
            ]
        )
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
            X_test_lstm_dist,
            tf.convert_to_tensor(feat_norm_test_lstm_baz.astype("float32"))
        )
    
    # Extract the predictions made for all the tasks
    train_binary_pred = np.hstack(train_prediction[0])
    val_binary_pred = np.hstack(val_prediction[0])
    test_binary_pred = np.hstack(test_prediction[0])

    train_distances_pred = train_prediction[1]
    val_distances_pred = val_prediction[1]
    test_distances_pred = test_prediction[1]

    train_cos_pred = train_prediction[2]
    val_cos_pred = val_prediction[2]
    test_cos_pred = test_prediction[2]
    print(train_cos_pred.shape)
    print(val_cos_pred.shape)
    print(test_cos_pred.shape)

    train_sin_pred = train_prediction[3]
    val_sin_pred = val_prediction[3]
    test_sin_pred = test_prediction[3]

    conf_matrix_train, accuracy_train, f1_train, precision_train, recall_train = compute_binary_metrics(train_binary_pred, y_train[:,0], "train")
    conf_matrix_val, accuracy_val, f1_val, precision_val, recall_val = compute_binary_metrics(val_binary_pred, y_val[:,0], "val")
    conf_matrix_test, accuracy_test, f1_test, precision_test, recall_test = compute_binary_metrics(test_binary_pred, y_test[:,0], "test")

    # MAE PARA DISTANCIA
    mae_train_dist = mean_absolute_error(y_train[:,1], train_distances_pred)
    mae_val_dist = mean_absolute_error(y_val[:,1], val_distances_pred)
    mae_test_dist = mean_absolute_error(y_test[:,1], test_distances_pred)

    # MAE PARA COS Y SIN 
    train_prediction_baz = to_angle(train_sin_pred, train_cos_pred)
    val_prediction_baz = to_angle(val_sin_pred, val_cos_pred)
    test_prediction_baz = to_angle(test_sin_pred, test_sin_pred)

    train_target_baz = to_angle(labels_train[:, 3], labels_train[:, 2])
    val_target_baz = to_angle(labels_val[:, 3], labels_val[:, 2])
    test_target_baz = to_angle(labels_test[:, 3], labels_test[:, 2])


    mae_val_baz = estimar_error_abs(val_prediction_baz, val_target_baz)
    mae_test_baz = estimar_error_abs(test_prediction_baz, test_target_baz)
    mae_train_baz = estimar_error_abs(train_prediction_baz, train_target_baz)


    tests_csv = save_regression_metrics(tests_csv, seed, mae_train_dist, mae_train_baz ,"train")
    tests_csv = save_regression_metrics(tests_csv, seed, mae_val_dist, mae_val_baz, "val")
    tests_csv = save_regression_metrics(tests_csv, seed, mae_test_dist, mae_test_baz, "test")


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

    train_targets = [y_train[:,0], y_train[:,1], y_train[:,2], y_train[:,3]]
    val_targets = [y_val[:,0], y_val[:,1], y_val[:,2], y_val[:,3]]
    test_targets = [y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3]]


    df_train_binary = save_results(id_train, stations_train, train_targets, train_prediction, prefix, test_name, seed, "train", "binary_classifier")
    df_val_binary = save_results(id_val, stations_val, val_targets, val_prediction, prefix, test_name, seed, "val", "binary_classifier")
    df_test_binary = save_results(id_test, stations_test, test_targets, test_prediction, prefix, test_name, seed, "test", "binary_classifier")

    # Path to the directory where the binary results will be saved (same as the previous one, but one folder deeper)
    binary_results_dir = os.path.join(test_file_dir, "results", "binary_results", test_file_name)
    os.makedirs(binary_results_dir, exist_ok=True)

    df_train_binary_path = os.path.join(binary_results_dir, f"{seed}_train_binary.csv")
    df_val_binary_path = os.path.join(binary_results_dir, f"{seed}_val_binary.csv")
    df_test_binary_path = os.path.join(binary_results_dir, f"{seed}_test_binary.csv")

    df_train_binary.to_csv(df_train_binary_path, index=False)
    df_val_binary.to_csv(df_val_binary_path, index=False)
    df_test_binary.to_csv(df_test_binary_path, index=False)
