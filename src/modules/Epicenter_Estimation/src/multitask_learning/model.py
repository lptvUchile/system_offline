from tensorflow import keras
from keras_multi_head import MultiHeadAttention
from keras.models import Model
from keras.layers import (
    Dense,
    Concatenate,
    Input,
    Dropout,
    LayerNormalization,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Bidirectional,
)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Masking
from tensorflow.keras.optimizers import Adam

from src.utils import VanillaPositionalEncoding, to_angle, sec_div_max
from src.utils import MyBatchGenerator_lstm_mlp

import numpy as np
import time


class MultiTaskModel():
    def __init__(self, tam_feat_mlp_dist, tam_feat_lstm_dist, largo_cota_dist,
                 tam_feat_mlp_baz, tam_feat_lstm_baz, largo_cota_baz,
                global_feat=True): 

        self.global_feat = global_feat
        self.tam_feat_mlp_dist = tam_feat_mlp_dist
        self.tam_feat_lstm_dist = tam_feat_lstm_dist
        self.largo_cota_dist = largo_cota_dist
        self.tam_feat_mlp_baz = tam_feat_mlp_baz
        self.tam_feat_lstm_baz = tam_feat_lstm_baz
        self.largo_cota_baz = largo_cota_baz

    def setup_model(self):

        # BackAzimuthModel LSTM Input and CNN Layers
        lstm_input_azimuth = Input(shape=(self.largo_cota_baz, self.tam_feat_lstm_baz))
        masked_azimuth = Masking(mask_value=0.0)(lstm_input_azimuth)  # Mask zero-padded values

        CNN1 = Conv1D(24, 3, activation="relu")(masked_azimuth)
        CNN1_drop = Dropout(0.2)(CNN1)
        pool1 = MaxPooling1D(pool_size=2)(CNN1_drop)

        CNN2 = Conv1D(16, 3, activation="relu")(pool1)
        CNN2_drop = Dropout(0.2)(CNN2)
        pool2 = MaxPooling1D(pool_size=2)(CNN2_drop)

        CNN3 = Conv1D(16, 3, activation="relu")(pool2)
        CNN3_drop = Dropout(0.2)(CNN3)
        pool3 = MaxPooling1D(pool_size=2)(CNN3_drop)

        CNN4 = Conv1D(16, 3, activation="relu")(pool3)
        CNN4_drop = Dropout(0.2)(CNN4) 
        pool4 = MaxPooling1D(pool_size=2)(CNN4_drop)

        hidden1 = Bidirectional(LSTM(10, activation="relu", return_sequences=True))(
            pool4
        )
        hidden2 = Bidirectional(LSTM(10, activation="relu", return_sequences=True))(
            hidden1
        )
        hidden3 = Bidirectional(LSTM(10, activation="relu", return_sequences=False))(
            hidden2
        )
        norm = LayerNormalization()(hidden3)

        # Cambiamos la capa de atención y el orden de la capa de layernormalization y se saca el vanilla encoding
        # pool4_norm = LayerNormalization()(pool4)  
        # inputs_pos_enc = VanillaPositionalEncoding()(pool4_norm)
        # hidden_azimuth = MultiHeadAttention(
        #     head_num=2, activation=None, use_bias=False, name="Multi-Head_1"
        # )(inputs_pos_enc)

        flat_azimuth = keras.layers.Flatten()(norm)

        # DistanceModel LSTM Input and Bi-LSTM Layer
        lstm_input_distance = Input(shape=(None, self.tam_feat_lstm_dist))
        masked_distance = Masking(mask_value=0.0)(lstm_input_distance)  # Mask zero-padded values
        hidden_distance = Bidirectional(LSTM(10, activation='relu', return_sequences=False))(masked_distance)

        combined_output = Concatenate()([flat_azimuth, hidden_distance])

        if self.global_feat:
            input_mlp = Input(shape=(self.tam_feat_mlp_dist,))
            combined_output = Concatenate()([combined_output, input_mlp])

        # shared layer
        mlp_hidden = Dense(64, activation="relu")(combined_output)
        mlp_hidden_drop = Dropout(0.2)(mlp_hidden)
        mlp_hidden = Dense(32, activation="relu")(mlp_hidden_drop)
        mlp_hidden_drop = Dropout(0.2)(mlp_hidden)

        output_costero = Dense(1, activation='sigmoid', name='costero')(mlp_hidden_drop)
        output_distancia = Dense(1, name='distancia')(mlp_hidden_drop)
        output_coseno = Dense(1, activation='tanh', name='coseno')(mlp_hidden_drop)
        output_seno = Dense(1, activation='tanh', name='seno')(mlp_hidden_drop)


        # Model with or without global features
        if self.global_feat:
            self.model = Model(
                inputs=[lstm_input_distance, lstm_input_azimuth, input_mlp], 
                outputs=[output_costero, output_distancia, output_coseno, output_seno]  # Cada taras tiene su propia salida
            )
        else:
            self.model = Model(
                inputs=[lstm_input_distance, lstm_input_azimuth], 
                outputs=[output_costero, output_distancia, output_coseno, output_seno] 
            )


    def dynamic_loss(self, y_true, y_pred):
        # Calcular el error de cada objetivo
        costero_loss = tf.keras.losses.binary_crossentropy(y_true['costero'], y_pred['costero'])
        distancia_loss = tf.keras.losses.mean_squared_error(y_true['distancia'], y_pred['distancia'])
        coseno_loss = tf.keras.losses.mean_squared_error(y_true['coseno'], y_pred['coseno'])
        seno_loss = tf.keras.losses.mean_squared_error(y_true['seno'], y_pred['seno'])

        # Normalizar las pérdidas
        total_loss = costero_loss + distancia_loss + coseno_loss + seno_loss
        loss_weights = {
            'costero': costero_loss / total_loss,
            'distancia': distancia_loss / total_loss,
            'coseno': coseno_loss / total_loss,
            'seno': seno_loss / total_loss
        }

        return loss_weights

    def setup_training(self, model_path, dummy=False, loss=None, lr=0.00015, patience=50, epochs=1000):
        self.model_path = model_path
        self.epochs = epochs
        opt = Adam(learning_rate=lr)
        
        if loss is None:
            loss = {
                'costero': 'binary_crossentropy', 
                'distancia': 'mean_squared_error',
                'coseno': 'mean_squared_error',    
                'seno': 'mean_squared_error'                    
            }

        def normalize_losses(losses):
            mean = tf.reduce_mean(losses)
            std = tf.math.reduce_std(losses)
            return (losses - mean) / (std + 1e-5)

        def dynamic_loss(y_true, y_pred, task_loss):
            loss_value = task_loss(y_true, y_pred)
            normalized_loss = normalize_losses(loss_value)
            
            if tf.reduce_mean(loss_value) > 0:
                sigma_sq = tf.math.reduce_variance(normalized_loss)
                epsilon = 1e-5
                loss_weight = 1.0 / (2.0 * (sigma_sq + epsilon))
                loss_weight = tf.clip_by_value(loss_weight, 0.1, 10.0)  # Limitar el rango de los pesos
                loss_value *= loss_weight
            
            return loss_value
        
        loss = {
            'costero': lambda y_true, y_pred: dynamic_loss(y_true, y_pred, tf.keras.losses.binary_crossentropy),
            'distancia': lambda y_true, y_pred: dynamic_loss(y_true, y_pred, tf.keras.losses.mean_squared_error),
            'coseno': lambda y_true, y_pred: dynamic_loss(y_true, y_pred, tf.keras.losses.mean_squared_error),
            'seno': lambda y_true, y_pred: dynamic_loss(y_true, y_pred, tf.keras.losses.mean_squared_error)
        }

        # Compilar el modelo con la pérdida ajustada
        self.model.compile(loss=loss, optimizer=opt)

        # self.model.compile(loss=loss, optimizer=opt)


        # EarlyStopping and ModelCheckpoint callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            mode="min", 
            verbose=1, 
            patience=patience, 
            restore_best_weights=True
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, 
            monitor="val_loss", 
            mode="min", 
            save_best_only=True, 
            verbose=1
        )


        if dummy:
            print("Dummy model")
            self.callbacks = [model_checkpoint]
        else:
            self.callbacks = [early_stopping, model_checkpoint]

            
    def pad_and_convert_to_tensor(self, arrays):
        max_length = max(arr.shape[0] for arr in arrays)
        
        padded_arrays = [
            np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant')
            for arr in arrays
        ]
        
        return tf.convert_to_tensor(padded_arrays, dtype=tf.float32)


    def input_training(self, feat_norm_train_lstm_dist, feat_in_train_mlp_dist,
                       feat_norm_train_lstm_baz, labels_train):
        
        self.X_train_lstm_baz = (
            tf.convert_to_tensor(feat_norm_train_lstm_baz.astype("float32")),
        )
        self.X_train_lstm_dist = self.pad_and_convert_to_tensor(feat_norm_train_lstm_dist)
        self.X_train_mlp_dist = tf.convert_to_tensor(feat_in_train_mlp_dist)

        self.y_train_costero = tf.convert_to_tensor(labels_train[:,0].astype("float32"))
        self.y_train_distancia = tf.convert_to_tensor(labels_train[:,1].astype("float32"))
        self.y_train_coseno = tf.convert_to_tensor(labels_train[:,2].astype("float32"))
        self.y_train_seno = tf.convert_to_tensor(labels_train[:,3].astype("float32"))



    def input_validation(self, feat_norm_val_lstm_dist, feat_in_val_mlp_dist,
                         feat_norm_val_lstm_baz, labels_val):
        
        self.X_val_lstm_baz = (
            tf.convert_to_tensor(feat_norm_val_lstm_baz.astype("float32")),
        )

        self.X_val_lstm_dist = self.pad_and_convert_to_tensor(feat_norm_val_lstm_dist)
        self.X_val_mlp_dist = tf.convert_to_tensor(feat_in_val_mlp_dist)

        self.y_val_costero = tf.convert_to_tensor(labels_val[:,0].astype("float32"))
        self.y_val_distancia = tf.convert_to_tensor(labels_val[:,1].astype("float32"))
        self.y_val_coseno = tf.convert_to_tensor(labels_val[:,2].astype("float32"))
        self.y_val_seno = tf.convert_to_tensor(labels_val[:,3].astype("float32"))



    def train(self):
        self.model.summary()

        self.X_train_mlp_dist = tf.cast(self.X_train_mlp_dist, tf.float32)
        self.input_mlps = self.X_train_mlp_dist

        self.X_val_mlp_dist = tf.cast(self.X_val_mlp_dist, tf.float32)
        self.input_mlps_val = self.X_val_mlp_dist

        if self.global_feat:
            hist = self.model.fit(
                x=[self.X_train_lstm_dist, self.X_train_lstm_baz, self.input_mlps],
                y={
                    'costero': self.y_train_costero, 
                    'distancia': self.y_train_distancia,
                    'coseno': self.y_train_coseno, 
                    'seno': self.y_train_seno
                },
                validation_data=(
                    [self.X_val_lstm_dist, self.X_val_lstm_baz, self.input_mlps_val],
                    {
                        'costero': self.y_val_costero, 
                        'distancia': self.y_val_distancia,
                        'coseno': self.y_val_coseno, 
                        'seno': self.y_val_seno
                    }
                ),
                batch_size=32,
                epochs=self.epochs,
                callbacks=self.callbacks,  
                verbose=2,
            )
        else:
            hist = self.model.fit(
                x=[self.X_train_lstm_dist, self.X_train_lstm_baz],
                y={
                    'output_costero': self.y_train_costero, 
                    'output_distancia': self.y_train_distancia,
                    'output_coseno': self.y_train_coseno, 
                    'output_seno': self.y_train_seno
                },
                validation_data=(
                    [self.X_val_lstm_dist, self.X_val_lstm_baz],
                    {
                        'output_costero': self.y_val_costero, 
                        'output_distancia': self.y_val_distancia,
                        'output_coseno': self.y_val_coseno, 
                        'output_seno': self.y_val_seno
                    }
                ),
                batch_size=32,
                epochs=self.epochs,
                callbacks=self.callbacks, 
                verbose=2,
            )

        loss_train = hist.history["loss"]
        loss_val = hist.history["val_loss"]

        return self.model



    def predict_test(self, temporal_feat_baz, temporal_feat_dist, mlp_feat):
        temporal_feat_baz = np.array(temporal_feat_baz).astype("float32")
        temporal_feat_dist = np.array(temporal_feat_dist).astype("float32")
        if mlp_feat is not None:
            mlp_feat = np.array(mlp_feat).astype("float32")
            print(temporal_feat_baz.shape, temporal_feat_dist.shape, mlp_feat.shape)
            prediction = self.model.predict(
                [tf.convert_to_tensor(temporal_feat_baz), tf.convert_to_tensor(temporal_feat_dist), tf.convert_to_tensor(mlp_feat)],
                verbose=0,
            )
        else:
            prediction = self.model.predict(
                [tf.convert_to_tensor(temporal_feat_baz), tf.convert_to_tensor(temporal_feat_dist)],
                verbose=0,
            )
        return prediction[0]

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "MultiHeadAttention": MultiHeadAttention,
                "VanillaPositionalEncoding": VanillaPositionalEncoding,
            },
        )
