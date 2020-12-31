import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import backend
from math import floor
import numpy as np

class CosineAnnealingLearningRateSchedule(keras.callbacks.Callback):
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
            # save_fp = '{}{}'.format(self.save_dir, int((epoch+1)/epochs_per_cycle))
            self.model.save(self.save_dir)