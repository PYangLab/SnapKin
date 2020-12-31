import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import pi, cos

WIDTHS = np.array([2,4,8,16,32,64,128])

def FFNN_Decoder(num_inputs, num_outputs, num_layers=3,
                 hidden_activation=tf.nn.leaky_relu, outputs_activation=tf.nn.sigmoid):
    tf.keras.backend.clear_session()

    width_idx = np.argmax(WIDTHS[WIDTHS <= num_inputs])
    # Input Layer
    inputs = keras.Input(shape=(num_inputs,))
    # Hidden Layer
    layer = (layers.Dense(WIDTHS[width_idx], activation=hidden_activation)(inputs))
    for i in range(num_layers - 1):
        if width_idx > 0:
            width_idx += -1
        layer = (layers.Dense(WIDTHS[width_idx], activation=hidden_activation)(layer))
    # Output Layer
    outputs = layers.Dense(num_outputs, activation=outputs_activation) (layer)
    # Model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )
    return model


