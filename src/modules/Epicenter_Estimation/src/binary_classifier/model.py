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
    Masking
)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.utils import VanillaPositionalEncoding, to_angle, sec_div_max
from src.utils import MyBatchGenerator_lstm_mlp

import numpy as np


class BinaryClassModel():
    def __init__(self, tam_feat_mlp_dist, tam_feat_lstm_dist, largo_cota_dist,
                 tam_feat_mlp_baz, tam_feat_lstm_baz, largo_cota_baz, batch_size,
                 global_feat=True, use_cnn=True, coast = False, softmax = True): 

        self.global_feat = global_feat
        self.tam_feat_mlp_dist = tam_feat_mlp_dist
        self.tam_feat_lstm_dist = tam_feat_lstm_dist
        self.largo_cota_dist = largo_cota_dist
        self.tam_feat_mlp_baz = tam_feat_mlp_baz
        self.tam_feat_lstm_baz = tam_feat_lstm_baz
        self.largo_cota_baz = largo_cota_baz
        self.use_cnn = use_cnn
        self.batch_size = batch_size
        self.coast = coast
        self.softmax = softmax

    def setup_model(self):
        # BackAzimuthModel LSTM Input and CNN Layers
        lstm_input_azimuth = Input(shape=(self.largo_cota_baz, self.tam_feat_lstm_baz))
        CNN1 = Conv1D(24, 3, activation="relu")(lstm_input_azimuth)
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

        if self.use_cnn:
            lstm_input_distance = Input(shape=(None, self.tam_feat_lstm_dist))
            
            CNN1 = Conv1D(32, 8, activation="relu")(lstm_input_distance)  # 32 8
            CNN1_drop = Dropout(0.2)(CNN1)
            pool1 = MaxPooling1D(pool_size=2)(CNN1_drop)

            CNN2 = Conv1D(24, 4, activation="relu")(pool1)  # 24
            CNN2_drop = Dropout(0.2)(CNN2)
            pool2 = MaxPooling1D(pool_size=2)(CNN2_drop)

            CNN3 = Conv1D(16, 3, activation="relu")(pool2)  # 16
            CNN3_drop = Dropout(0.2)(CNN3)
            pool3 = MaxPooling1D(pool_size=2)(CNN3_drop)

            CNN4 = Conv1D(16, 3, activation="relu")(pool3)  # 16
            CNN4_drop = Dropout(0.2)(CNN4)
            pool4 = MaxPooling1D(pool_size=2)(CNN4_drop)
                    
            hidden1 = Bidirectional(LSTM(10, activation = 'relu', return_sequences = True))(pool4)
            hidden2 = Bidirectional(LSTM(10, activation = 'relu', return_sequences = True))(hidden1)
            hidden3 = Bidirectional(LSTM(10, activation = 'relu', return_sequences = False))(hidden2)
            norm = LayerNormalization()(hidden3)
            hidden_distance = keras.layers.Flatten()(norm)

        else:
            #Modelo LSTM
            lstm_input_distance = Input(shape=(None, self.tam_feat_lstm_dist))
            masked_lstm_output = Masking(mask_value=0.0)(lstm_input_distance)  # Add Masking layer to ignore zeros
            hidden_distance = Bidirectional(LSTM(10, activation = 'relu', return_sequences = False))(masked_lstm_output)

        #lstm_input_distance = Input(shape=(None, self.tam_feat_lstm_dist))
        #hidden_distance = Bidirectional(LSTM(10, activation='relu', return_sequences=False))(lstm_input_distance)

        combined_output = Concatenate()([flat_azimuth, hidden_distance])

        if self.global_feat:
            input_mlp = Input(shape=(self.tam_feat_mlp_dist,))
            combined_output = Concatenate()([combined_output, input_mlp])

        # Single MLP for classification
        mlp_hidden = Dense(30, activation="relu")(combined_output)
        mlp_hidden_drop = Dropout(0.2)(mlp_hidden)

        if self.softmax:
            final_output = Dense(2, activation="softmax")(mlp_hidden_drop)
        else:
            final_output = Dense(1, activation="sigmoid")(mlp_hidden_drop)

        if self.global_feat:
            self.model = Model(inputs=[lstm_input_distance, lstm_input_azimuth, input_mlp], outputs=final_output)
        else:
            self.model = Model(inputs=[lstm_input_distance, lstm_input_azimuth], outputs=final_output)


    def setup_training(self, model_path, dummy = False, loss="binary_crossentropy", lr=0.0001, patience=50, epochs=1000):
        self.model_path = model_path
        self.epochs = epochs
        opt = Adam(learning_rate=lr)
        self.model.compile(loss=loss, optimizer=opt)
        self.es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=patience,
            restore_best_weights=True,
            start_from_epoch=50
        )

        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=1)
        if dummy:
            print("Dummy model")
            self.callbacks = [self.checkpoint]
        else:
            self.callbacks = [self.es, self.checkpoint]
            
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
        self.X_train_mlp_dist = tf.convert_to_tensor(feat_in_train_mlp_dist, dtype=tf.float32)
        #self.y_train_izq = tf.convert_to_tensor(labels_train[:,0].astype("float32"))
        #self.y_train_der = tf.convert_to_tensor(labels_train[:,1].astype("float32"))
        self.y_train = tf.convert_to_tensor(labels_train.astype("float32"))


    def input_validation(self, feat_norm_val_lstm_dist, feat_in_val_mlp_dist,
                         feat_norm_val_lstm_baz, labels_val):
        
        self.X_val_lstm_baz = (
            tf.convert_to_tensor(feat_norm_val_lstm_baz.astype("float32")),
        )

        self.X_val_lstm_dist = self.pad_and_convert_to_tensor(feat_norm_val_lstm_dist)
        self.X_val_mlp_dist = tf.convert_to_tensor(feat_in_val_mlp_dist, dtype=tf.float32)

        #self.y_val_izq = tf.convert_to_tensor(labels_val[:,0].astype("float32").reshape(-1, 1))
        #self.y_val_der = tf.convert_to_tensor(labels_val[:,1].astype("float32").reshape(-1, 1))
        self.y_val = tf.convert_to_tensor(labels_val.astype("float32"))

    def train(self):
        self.model.summary()
        
        self.X_train_mlp_dist = tf.cast(self.X_train_mlp_dist, tf.float32)
        self.input_mlps = self.X_train_mlp_dist

        self.X_val_mlp_dist = tf.cast(self.X_val_mlp_dist, tf.float32)
        self.input_mlps_val = self.X_val_mlp_dist

        if self.global_feat:
            hist = self.model.fit(
                x=[self.X_train_lstm_dist, self.X_train_lstm_baz, self.input_mlps],
                y=[self.y_train],
                validation_data=(
                    [self.X_val_lstm_dist, self.X_val_lstm_baz, self.input_mlps_val], 
                    [self.y_val]
                ),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.callbacks,  # Use self.callbacks instead of self.es
                verbose=2,
            )
        else:
            hist = self.model.fit(
                x=[self.X_train_lstm_dist, self.X_train_lstm_baz],
                y=self.y_train,
                validation_data=(
                    [self.X_val_lstm_dist, self.X_val_lstm_baz], 
                    self.y_val
                ),
                batch_size=self.batch_size,
                epochs=200,
                callbacks=self.callbacks,  # Use self.callbacks instead of self.es
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
