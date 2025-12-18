import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate

class DynamicWeightedLoss(Model):
    def __init__(self, model, **kwargs):
        super(DynamicWeightedLoss, self).__init__(**kwargs)
        self.model = model
        # Learnable log-variances for each task
        self.log_sigma_costero = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        self.log_sigma_distancia = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        self.log_sigma_coseno = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        self.log_sigma_seno = tf.Variable(0.0, trainable=True, dtype=tf.float32)

    def call(self, inputs):
        return self.model(inputs)

    def compute_loss(self, y_true, y_pred):
        # Unpack true and predicted values
        y_costero_true, y_distancia_true, y_coseno_true, y_seno_true = y_true
        y_costero_pred, y_distancia_pred, y_coseno_pred, y_seno_pred = y_pred

        # Individual task losses
        loss_costero = tf.keras.losses.BinaryCrossentropy()(y_costero_true, y_costero_pred)
        loss_distancia = tf.keras.losses.MeanSquaredError()(y_distancia_true, y_distancia_pred)
        loss_coseno = tf.keras.losses.MeanSquaredError()(y_coseno_true, y_coseno_pred)
        loss_seno = tf.keras.losses.MeanSquaredError()(y_seno_true, y_seno_pred)

        # Apply dynamic weighting
        weighted_loss_costero = loss_costero / (2 * tf.exp(self.log_sigma_costero)) + self.log_sigma_costero
        weighted_loss_distancia = loss_distancia / (2 * tf.exp(self.log_sigma_distancia)) + self.log_sigma_distancia
        weighted_loss_coseno = loss_coseno / (2 * tf.exp(self.log_sigma_coseno)) + self.log_sigma_coseno
        weighted_loss_seno = loss_seno / (2 * tf.exp(self.log_sigma_seno)) + self.log_sigma_seno

        # Total loss
        total_loss = weighted_loss_costero + weighted_loss_distancia + weighted_loss_coseno + weighted_loss_seno
        return total_loss
