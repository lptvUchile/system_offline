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
    Bidirectional,
    LSTM,
)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from src.utils import VanillaPositionalEncoding, to_angle, sec_div_max
import numpy as np


class BackAzimuthModel:
    def setup_model(
        self,
        tam_feat_mlp,
        tam_feat_lstm,
        largo_cota,
        nro_output=2,
        global_feat=True,
        binary=False,
    ):
        self.global_feat = global_feat
        self.binary = binary
        input_mlp = Input(shape=(tam_feat_mlp,))

        # Modelo LSTM
        lstm_input = Input(shape=(largo_cota, tam_feat_lstm))

        CNN1 = Conv1D(24, 3, activation="relu")(lstm_input)  # 32 8
        CNN1_drop = Dropout(0.2)(CNN1)
        pool1 = MaxPooling1D(pool_size=2)(CNN1_drop)

        CNN2 = Conv1D(16, 3, activation="relu")(pool1)  # 24
        CNN2_drop = Dropout(0.2)(CNN2)
        pool2 = MaxPoolin2g1D(pool_size=2)(CNN2_drop)

        CNN3 = Conv1D(16, 3, activation="relu")(pool2)  # 16
        CNN3_drop = Dropout(0.2)(CNN3)
        pool3 = MaxPooling1D(pool_size=2)(CNN3_drop)

        CNN4 = Conv1D(16, 3, activation="relu")(pool3)  # 16
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
        # hidden4 = Bidirectional(LSTM(10, activation="relu", return_sequences=True))(
        #     hidden3
        # )
        # hidden5 = Bidirectional(LSTM(10, activation="relu", return_sequences=True))(
        #     hidden4
        # )

        # inputs_pos_enc = VanillaPositionalEncoding()(hidden5)
        #
        # hidden1 = MultiHeadAttention(
        #     head_num=2, activation=None, use_bias=False, name="Multi-Head_1"
        # )(
        #     inputs_pos_enc
        # )  # 5
        #
        flat = keras.layers.Flatten()(norm)
        if global_feat:
            concat = Concatenate()([flat, input_mlp])
            mlp_hidden = Dense(20, activation="relu")(concat)  # 15
        else:
            mlp_hidden = Dense(20, activation="relu")(flat)  # 15

        mlp_hidden_drop = Dropout(0.2)(mlp_hidden)

        if binary:
            mlp_out = Dense(1, activation="sigmoid")(mlp_hidden_drop)  # salida
        else:
            mlp_out = Dense(nro_output)(mlp_hidden_drop)

        if global_feat:
            self.model = Model(inputs=[lstm_input, input_mlp], outputs=mlp_out)
        else:
            self.model = Model(inputs=[lstm_input], outputs=mlp_out)

    def setup_training(self, model_path, dummy=False, loss="mean_squared_error", lr=0.00015, patience=50, epochs=1000):
        self.model_path = model_path
        self.epochs = epochs
        opt = Adam(learning_rate=lr)  # 0.00015 mayores4M y todaBD
        self.model.compile(loss=loss, optimizer=opt)
        self.es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=patience, # 30 default
            restore_best_weights=True,
        )
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=1)
        if dummy:
            self.callbacks = [self.checkpoint]
        else:
            self.callbacks = [self.es, self.checkpoint]

    def input_training(self, feat_norm_train_lstm, feat_in_train_mlp, az_real_train):
        self.X_train_lstm, self.X_train_mlp, self.y_train = (
            tf.convert_to_tensor(feat_norm_train_lstm.astype("float32")),
            tf.convert_to_tensor(feat_in_train_mlp),
            tf.convert_to_tensor(az_real_train),
        )

    def input_validation(self, feat_norm_val_lstm, feat_norm_val_mlp, az_real_val):
        self.X_val_lstm, self.X_val_mlp, self.y_val = (
            tf.convert_to_tensor(feat_norm_val_lstm.astype("float32")),
            tf.convert_to_tensor(feat_norm_val_mlp.astype("float32")),
            tf.convert_to_tensor(az_real_val),
        )

    def train(self):
        self.model.summary()  # patience 40
        hist = self.model.fit(
            x=(
                [self.X_train_lstm, self.X_train_mlp]
                if self.global_feat
                else [self.X_train_lstm]
            ),
            y=self.y_train,
            batch_size=1,
            validation_data=(
                (
                    [self.X_val_lstm, self.X_val_mlp]
                    if self.global_feat
                    else [self.X_val_lstm]
                ),
                self.y_val,
            ),
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=2,
        )

        loss_train = hist.history["loss"]
        loss_val = hist.history["val_loss"]
        self.load_model(self.model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "MultiHeadAttention": MultiHeadAttention,
                "VanillaPositionalEncoding": VanillaPositionalEncoding,
            },
        )

    def predict(self, temporal_feat, mlp_feat=None):
        temporal_feat = np.array(temporal_feat)
        temporal_feat = temporal_feat.astype("float")
        temporal_feat = sec_div_max(temporal_feat)
        if mlp_feat is not None:
            mlp_feat = np.array(mlp_feat)
            prediction = self.model.predict(
                [
                    tf.convert_to_tensor(temporal_feat.astype("float32")),
                    tf.convert_to_tensor(mlp_feat),
                ],
                verbose=0,
            )
        else:
            prediction = self.model.predict(
                [
                    tf.convert_to_tensor(temporal_feat.astype("float32")),
                ],
                verbose=0,
            )
        prediction = to_angle(prediction[:, 1], prediction[:, 0])
        return prediction[0]
