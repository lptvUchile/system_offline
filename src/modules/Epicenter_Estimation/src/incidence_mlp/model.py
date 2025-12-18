from keras.models import Model
from keras.layers import (
    Dense,
    Input,
)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np


class IncidenceMLPModel:

    def setup_model(
        self,
        tam_feat_mlp,
        nro_output=1,
    ):

        input_mlp = Input(shape=(tam_feat_mlp,))

        hidden1 = Dense(tam_feat_mlp, activation="relu")(input_mlp)
        
        hidden2 = Dense(32, activation="relu")(hidden1)
        
        hidden3 = Dense(16, activation="relu")(hidden2)
        
        output = Dense(nro_output)(hidden3)

        self.model = Model(inputs=input_mlp, outputs=output)

    def setup_training(
        self,
        model_path,
        loss="mean_squared_error",
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

    def input_training(self, feat_norm_train_mlp, labels_train):
        self.X_train_mlp, self.y_train = (
            tf.convert_to_tensor(feat_norm_train_mlp, dtype=tf.float32),
            tf.convert_to_tensor(labels_train, dtype=tf.float32),
        )

    def input_validation(self, feat_norm_val_mlp, labels_val):
        self.X_val_mlp, self.y_val = (
            tf.convert_to_tensor(feat_norm_val_mlp, dtype=tf.float32),
            tf.convert_to_tensor(labels_val, dtype=tf.float32),
        )

    def train(self):
        self.model.summary()  # patience 40
        hist = self.model.fit(
            x=([self.X_train_mlp]),
            y=self.y_train,
            batch_size=1,
            validation_data=(
                ([self.X_val_mlp]),
                self.y_val,
            ),
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

    def predict(self, mlp_feat):
        mlp_feat = np.array(mlp_feat)

        prediction = self.model.predict(
            [
                tf.convert_to_tensor(mlp_feat, dtype=tf.float32),
            ],
            verbose=0,
        )

        return prediction[0]
