#!/usr/bin/env python
# coding: utf-8


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from src.back_azimuth import BackAzimuthModel
from src.back_azimuth.train_features_extraction import function_features_extraction

import numpy as np
import time
from tensorflow.python.keras import backend as K
import tensorflow as tf
import os
import pandas as pd
import argparse
from src.utils import compute_binary_metrics, save_binary_metrics, save_results, to_angle, estimar_error_abs, sec_div_max

parser = argparse.ArgumentParser(prog="Distance Training")
parser.add_argument("--test_file", type=str)
parser.add_argument("--feat_type", type=str)
parser.add_argument("--use_accel", action=argparse.BooleanOptionalAction)
parser.add_argument("--no_global_features", action=argparse.BooleanOptionalAction)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--extract_features", action=argparse.BooleanOptionalAction)
parser.add_argument("--response", type=str, default="VEL")
parser.add_argument("--binary", action=argparse.BooleanOptionalAction)
parser.add_argument("--one_hot_encoding", action=argparse.BooleanOptionalAction)
# window limit is an array of [min, max]
parser.add_argument("--window_limit", type=list, default=[1, 3])
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
no_global_features = not test_config.get('global_feat', not args.no_global_features)
test = args.test
response = test_config.get('response', args.response)
filter = test_config.get('filter', None)
dummy = test_config.get('dummy', False)
lr = test_config.get('lr', 0.00015)
epochs = test_config.get('epochs', 1000)
patience = test_config.get('patience', 50)
ome_hot_encoding = test_config.get('one_hot_encoding', args.one_hot_encoding)

print("--- PARÁMETROS ---")
print(f"feat_type: {feat_type}")
print(f"binary: {binary}")
print(f"window_limit: {window_limit}")
print(f"seeds: {seeds}")
print(f"use_accel: {use_accel}")
print(f"no_global_features: {no_global_features}")
print(f"test: {test}")
print(f"response: {response}")
print(f"filter: {filter}")
print(f"dummy: {dummy}")
print(f"lr: {lr}")
print(f"epochs: {epochs}")
print(f"patience: {patience}")
print(f"one_hot_encoding: {ome_hot_encoding}")

time.sleep(5)



prefix = "acc" if use_accel else "vel"
test_name = test_file.split("/")[-1].split(".")[
    0
]  # Get the name of the test from the path


### Extract feature
if args.extract_features:
    tf.keras.utils.set_random_seed(1)
    function_features_extraction(
        feat_type, test, use_accel, response, test_name, window_limit, filter, ome_hot_encoding = ome_hot_encoding
    )

### Load features


path_feat_in_train_lstm = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_cnn_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_train_mlp = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_mlp_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_az_real_train = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/angulo_raw_train_"
    + feat_type
    + ".npy".replace("\\", "/")
)

path_feat_in_val_lstm = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_cnn_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_val_mlp = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_mlp_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_az_real_val = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/angulo_raw_val_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_lstm = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_cnn_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_feat_in_test_mlp = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/feat_mlp_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)
path_az_real_test = (
    f"./data/features/back_azimuth/{prefix}/{test_name}/angulo_raw_test_"
    + feat_type
    + ".npy".replace("\\", "/")
)


feat_in_train_lstm = np.load(path_feat_in_train_lstm, allow_pickle=True)
feat_in_train_lstm = feat_in_train_lstm.astype("float")
feat_norm_train_lstm = sec_div_max(feat_in_train_lstm)
feat_in_train_mlp = np.load(path_feat_in_train_mlp, allow_pickle=True)
feat_in_train_mlp = feat_in_train_mlp
az_real_train = np.load(path_az_real_train, allow_pickle=True)
az_real_train = pd.DataFrame(az_real_train[()])
id_train = az_real_train["Evento"]
stations_train = az_real_train["Estacion"]


feat_in_val_lstm = np.load(path_feat_in_val_lstm, allow_pickle=True)
feat_in_val_lstm = feat_in_val_lstm.astype("float")
feat_norm_val_lstm = sec_div_max(feat_in_val_lstm)
feat_in_val_mlp = np.load(path_feat_in_val_mlp, allow_pickle=True)
feat_in_val_mlp = feat_in_val_mlp
az_real_val = np.load(path_az_real_val, allow_pickle=True)
az_real_val = pd.DataFrame(az_real_val[()])
id_val = az_real_val["Evento"]
stations_val = az_real_val["Estacion"]

feat_in_test_lstm = np.load(path_feat_in_test_lstm, allow_pickle=True)
feat_in_test_lstm = feat_in_test_lstm.astype("float")
feat_norm_test_lstm = sec_div_max(feat_in_test_lstm)
feat_in_test_mlp = np.load(path_feat_in_test_mlp, allow_pickle=True)
feat_in_test_mlp = feat_in_test_mlp
az_real_test = np.load(path_az_real_test, allow_pickle=True)
az_real_test = pd.DataFrame(az_real_test[()])
id_test = az_real_test["Evento"]
stations_test = az_real_test["Estacion"]


feat_norm_train_mlp = np.array(
    [feat_in_train_mlp[i] for i in range(len(feat_in_train_mlp))], dtype=object
)
feat_norm_val_mlp = np.array(
    [feat_in_val_mlp[i] for i in range(len(feat_in_val_mlp))], dtype=object
)
feat_norm_test_mlp = np.array(
    [feat_in_test_mlp[i] for i in range(len(feat_in_test_mlp))], dtype=object
)


# LABELS
if binary:
    labels_train = az_real_train["costero"].values
    labels_val = az_real_val["costero"].values
    labels_test = az_real_test["costero"].values
else:
    labels_train = np.concatenate(
    (
        az_real_train["Coseno"].values.reshape(-1, 1),
        az_real_train["Seno"].values.reshape(-1, 1),
    ),
    axis=1,
    )
    labels_test = np.concatenate(
        (
            az_real_test["Coseno"].values.reshape(-1, 1),
            az_real_test["Seno"].values.reshape(-1, 1),
        ),
        axis=1,
    )
    labels_val = np.concatenate(
        (
            az_real_val["Coseno"].values.reshape(-1, 1),
            az_real_val["Seno"].values.reshape(-1, 1),
        ),
        axis=1,
    )

az_real_train = labels_train
az_real_val = labels_val
az_real_test = labels_test

largo_cota = feat_in_train_lstm.shape[1]
tam_feat_mlp = feat_norm_train_mlp.shape[-1]
tam_feat_lstm = feat_norm_test_lstm.shape[-1]

if binary:
    tests_csv = pd.DataFrame(columns=["seed", "conf_matrix_train", "accuracy_train", "f1_train", "precision_train", "recall_train", "conf_matrix_val", "accuracy_val", "f1_val", "precision_val", "recall_val", "conf_matrix_test", "accuracy_test", "f1_test", "precision_test", "recall_test"])
else:
    raise Exception("Binary is not implemented")
tests_csv.set_index("seed", inplace=True)
for seed in seeds:
    tf.keras.utils.set_random_seed(seed)
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 1, "CPU": 1})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    repeats = 1  # number of repetitions
    nro_output = 2
    K.set_session(sess)
    path_root = os.getcwd()

    start_time = time.time()

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

    ba_model = BackAzimuthModel()
    ba_model.setup_model(
        tam_feat_mlp=tam_feat_mlp,
        tam_feat_lstm=tam_feat_lstm,
        largo_cota=largo_cota,
        nro_output=nro_output,
        global_feat=not args.no_global_features,
        binary=binary
    )
    
    model_path = f"./models/back_azimuth/{prefix}/{test_name}/{seed}.h5"
    os.makedirs(f"./models/back_azimuth/{prefix}/{test_name}", exist_ok=True)
    if binary:
        ba_model.setup_training(model_path=model_path, loss="binary_crossentropy", dummy=dummy, patience=patience, epochs=epochs, lr=lr)
    else:
        raise Exception("Binary is not implemented")

    ba_model.input_training(feat_norm_train_lstm, feat_in_train_mlp, az_real_train)
    ba_model.input_validation(feat_norm_val_lstm, feat_in_val_mlp, az_real_val)
    ba_model.train()
    model = ba_model.model

    if not args.no_global_features:
        train_prediction = model.predict(
            [
                tf.convert_to_tensor(feat_norm_train_lstm.astype("float32")),
                tf.convert_to_tensor(feat_in_train_mlp),
            ]
        )
        val_prediction = model.predict(
            [
                tf.convert_to_tensor(feat_norm_val_lstm.astype("float32")),
                tf.convert_to_tensor(feat_in_val_mlp),
            ]
        )
        test_prediction = model.predict(
            [
                tf.convert_to_tensor(feat_norm_test_lstm.astype("float32")),
                tf.convert_to_tensor(feat_in_test_mlp),
            ]
        )
    else:
        train_prediction = model.predict(
            [tf.convert_to_tensor(feat_norm_train_lstm.astype("float32"))]
        )
        val_prediction = model.predict(
            [tf.convert_to_tensor(feat_norm_val_lstm.astype("float32"))]
        )
        test_prediction = model.predict(
            [tf.convert_to_tensor(feat_norm_test_lstm.astype("float32"))]
        )
    if binary:
        conf_matrix_train, accuracy_train, f1_train, precision_train, recall_train = compute_binary_metrics(train_prediction, az_real_train, "train")
        conf_matrix_val, accuracy_val, f1_val, precision_val, recall_val = compute_binary_metrics(val_prediction, az_real_val, "val")
        conf_matrix_test, accuracy_test, f1_test, precision_test, recall_test = compute_binary_metrics(test_prediction, az_real_test, "test")

        tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_train, accuracy=accuracy_train, f1=f1_train, precision=precision_train, recall=recall_train, label="train")
        tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_val, accuracy=accuracy_val, f1=f1_val, precision=precision_val, recall=recall_val, label="val")
        tests_csv = save_binary_metrics(tests_csv=tests_csv, seed=seed, conf_matrix=conf_matrix_test, accuracy=accuracy_test, f1=f1_test, precision=precision_test, recall=recall_test, label="test")
    else:
        pass
    # path must be the folder where is the yaml file is (test_file) + /results/file_name.csv
    # this is for example: ./tests/mi_test.yaml -> ./tests/results/mi_test.csv
    # Construct the path for the CSV file
    test_file_dir = os.path.dirname(args.test_file)
    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    tests_csv_path = os.path.join(test_file_dir, "results", f"{test_file_name}.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(tests_csv_path), exist_ok=True)
    tests_csv.to_csv(tests_csv_path, index=True)

    
   

    save_results(id_train, stations_train, az_real_train, train_prediction, prefix, test_name, seed, "train", "back_azimuth")
    save_results(id_val, stations_val, az_real_val, val_prediction, prefix, test_name, seed, "val", "back_azimuth")
    save_results(id_test, stations_test, az_real_test, test_prediction, prefix, test_name, seed, "test", "back_azimuth")

