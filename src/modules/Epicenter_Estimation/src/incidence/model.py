from keras.models import Model
from tensorflow import keras
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
    Masking,
)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src.utils import MyBatchGenerator_lstm_mlp, pad_and_convert_to_tensor
import numpy as np


class IncidenceModel:

    def setup_model(
        self,
        tam_feat_mlp,
        tam_feat_lstm,
        largo_cota,
        nro_output=1,
        global_feat=True,
        binary=False,
        lstm_units=10,
        use_cnn=True,
        replace_lstm_with_mlp=False  # TODO: INDEV feature, future removal
    ):
        self.global_feat = global_feat

        if global_feat:
            input_mlp = Input(shape=(tam_feat_mlp,))

        if use_cnn:
            lstm_input = Input(shape=(largo_cota, tam_feat_lstm))

            CNN1 = Conv1D(24, 3, activation="relu", padding="same")(lstm_input)  # 32 8
            CNN1_drop = Dropout(0.2)(CNN1)
            pool1 = MaxPooling1D(pool_size=2)(CNN1_drop)

            CNN2 = Conv1D(16, 3, activation="relu", padding="same")(pool1)  # 24
            CNN2_drop = Dropout(0.2)(CNN2)
            pool2 = MaxPooling1D(pool_size=2)(CNN2_drop)

            CNN3 = Conv1D(16, 3, activation="relu", padding="same")(pool2)  # 16
            CNN3_drop = Dropout(0.2)(CNN3)
            pool3 = MaxPooling1D(pool_size=2)(CNN3_drop)

            if replace_lstm_with_mlp:
                hidden1 = Dense(30, activation="relu")(pool3)
                hidden2 = Dense(30, activation="relu")(hidden1)
                hidden3 = Dense(30, activation="relu")(hidden2)
                norm = LayerNormalization()(hidden3)
                dropout = Dropout(0.3)(norm)
                hidden_out = keras.layers.Reshape((30,))(dropout)
            
            else:
                hidden1 = Bidirectional(
                    LSTM(lstm_units, activation="relu", return_sequences=True)
                )(pool3)
                hidden2 = Bidirectional(
                    LSTM(lstm_units, activation="relu", return_sequences=True)
                )(hidden1)
                hidden3 = Bidirectional(
                    LSTM(lstm_units, activation="relu", return_sequences=False)
                )(hidden2)
                norm = LayerNormalization()(hidden3)
                hidden_out = keras.layers.Flatten()(norm)

        else:
            # Modelo LSTM
            lstm_input = Input(shape=(None, tam_feat_lstm))
            masked_lstm_output = Masking(mask_value=0.0)(
                lstm_input
            )  # Add Masking layer to ignore zeros
            hidden_out = Bidirectional(
                LSTM(lstm_units, activation="relu", return_sequences=True)
            )(masked_lstm_output)

        if global_feat:
            print(hidden_out.shape)
            print(input_mlp.shape)
            concat = Concatenate()([hidden_out, input_mlp])
            print(concat.shape)
            mlp_hidden = Dense(30, activation="relu")(concat)  # 15
        else:
            mlp_hidden = Dense(30, activation="relu")(hidden_out)

        if binary:
            mlp_out = Dense(2, activation="softmax")(mlp_hidden)  # salida
        else:
            mlp_out = Dense(nro_output)(mlp_hidden)  # salida

        if global_feat:
            self.model = Model(inputs=[lstm_input, input_mlp], outputs=mlp_out)
        else:
            self.model = Model(inputs=[lstm_input], outputs=mlp_out)

    def setup_training(
        self,
        model_path,
        loss="binary_crossentropy",
        lr=0.0001,
        dummy=False,
        patience=25,
        epochs=1000,
    ):
        self.model_path = model_path
        self.epochs = epochs
        opt = Adam(learning_rate=lr)
        self.model.compile(loss=loss, optimizer=opt)
        self.es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=patience,  # patience 30 4M, microsismicidad, 50 todaBD
            restore_best_weights=True,
        )  # el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        )
        if dummy:
            self.callbacks = [self.checkpoint]
        else:
            self.callbacks = [self.es, self.checkpoint]

    def input_training(
        self, feat_norm_train_lstm, feat_in_train_mlp, az_real_train, batch_size
    ):
        # X_train_lstm, X_train_mlp, y_train = feat_norm_train_lstm, feat_in_train_mlp, az_real_train
        X_train_lstm, X_train_mlp, y_train = (
            pad_and_convert_to_tensor(feat_norm_train_lstm),
            tf.convert_to_tensor(feat_in_train_mlp, dtype=tf.float32),
            az_real_train,
        )
        self.train_feat_target = MyBatchGenerator_lstm_mlp(
            X_train_lstm,
            X_train_mlp,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            global_feat=self.global_feat,
        )

    def input_validation(
        self, feat_norm_val_lstm, feat_in_val_mlp, az_real_val, batch_size
    ):
        # X_val_lstm, X_val_mlp, y_val = feat_norm_val_lstm, feat_in_val_mlp, az_real_val
        X_val_lstm, X_val_mlp, y_val = (
            pad_and_convert_to_tensor(feat_norm_val_lstm),
            tf.convert_to_tensor(feat_in_val_mlp, dtype=tf.float32),
            az_real_val,
        )
        self.val_feat_target = MyBatchGenerator_lstm_mlp(
            X_val_lstm,
            X_val_mlp,
            y_val,
            batch_size=batch_size,
            shuffle=True,
            global_feat=self.global_feat,
        )

    def train(self):
        self.model.summary()  # patience 40
        hist = self.model.fit(
            self.train_feat_target,
            validation_data=self.val_feat_target,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=2,
        )

        loss_train = hist.history["loss"]
        loss_val = hist.history["val_loss"]

        self.loss_final = {"loss_train": loss_train, "loss_val": loss_val}

        self.raw_load_model(self.model_path)

    def raw_load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def load_model(self, model_path, max_temporal_path, min_temporal_path, max_global_path, min_global_path, max_target_path):
        self.raw_load_model(model_path)
        self.max_target = np.load(max_target_path, allow_pickle=True)
        self.max_temporal_params = np.load(max_temporal_path, allow_pickle=True)
        self.min_temporal_params = np.load(min_temporal_path, allow_pickle=True)
        self.max_global_params = np.load(max_global_path, allow_pickle=True)
        self.min_global_params = np.load(min_global_path, allow_pickle=True)

    # def predict(self, temporal_feat, mlp_feat = None):
    #     temporal_feat = np.array([(temporal_feat[i]-self.min_params)/(self.max_params-self.min_params)
    #                      for i in range(len(temporal_feat))],dtype='float32')
    #     if mlp_feat is not None:
    #         prediction = self.model.predict([tf.convert_to_tensor(temporal_feat),tf.convert_to_tensor(mlp_feat)],verbose=0)[0][0]
    #     else:
    #         prediction = self.model.predict([tf.convert_to_tensor(temporal_feat)],verbose=0)[0][0]
    #     return prediction*self.max_target

    def predict(self, temporal_feat, mlp_feat=None):
        temporal_feat = tf.convert_to_tensor([temporal_feat], dtype=tf.float32)
        if mlp_feat is not None:
            mlp_feat = tf.convert_to_tensor([mlp_feat], dtype=tf.float32)

            prediction = self.model.predict(
                [temporal_feat, mlp_feat],
            )  # [:, 0]
        else:
            prediction = self.model.predict([temporal_feat], verbose=0)
        return np.squeeze(prediction).item() * self.max_target
