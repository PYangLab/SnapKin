import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from math import floor

def FFNN(num_inputs, num_outputs, num_layers=3, hidden_activation=tf.nn.leaky_relu, outputs_activation=tf.nn.sigmoid):
    '''
        Feed Forward Neural Network
    '''
    tf.keras.backend.clear_session()
    WIDTHS = np.array([ np.power(2,x) for x in range(1,floor(np.log2(num_inputs))+1)])

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

class CosineAnnealingLearningRateSchedule(keras.callbacks.Callback):
    '''
        SnapKin learning rate scheduler
    '''
    # Constructor 
    def __init__(self, initial_lr, nb_epochs, nb_snapshots, save_dir):
        self.nb_epochs = nb_epochs
        self.nb_snapshots = nb_snapshots
        self.initial_lr = initial_lr
        self.lrates = list()
        self.save_dir = save_dir

    def cosine_annealing(self, epoch):
        cos_inner = np.pi * (epoch % (self.nb_epochs // self.nb_snapshots))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.nb_epochs // self.nb_snapshots
        cos_out = np.cos(cos_inner) + 1
        return float(self.initial_lr / 2 * cos_out)

    # Calculate and set learning rate at start of epoch
    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch)
        # Set learning rate 
        backend.set_value(self.model.optimizer.lr, lr)
        # Log learning rate 
        self.lrates.append(lr)

    # Save model at end of cycle 
    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle = floor(self.nb_epochs/self.nb_snapshots)
        if epoch != 0 and (epoch+1) % epochs_per_cycle == 0:
            self.model.save(self.save_dir)