import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from BatchGenerator import BatchGenerator
from Helpers import *
from Models import FFNN_Decoder
from functools import reduce
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
import pickle
import os

def FFNN_tune(data, fold_set, fold_columns, clss_column, class_columns, num_layers=3, num_outputs=1, 
            verbose=0, batch_sizes=[32,64], num_epochs=100, learning_rates=[ 0.0001, 0.001, 0.01], loss_fn=keras.losses.BinaryCrossentropy(),
            num_batches=1, pseudo=True):
    '''
        Hyperparameter tuning via cross-validation.
    '''
    folds = set(reduce(lambda x,y: x+y,map(lambda x:x[fold_set].to_list() , data)))
    aucs = [[[] for _ in range(len(learning_rates))] for _ in range(len(batch_sizes))]
    
    num_inputs = ([dat.shape[1] - 1 - len(fold_columns) - len(class_columns) for dat in data])[0] # Only one dataset
    for fold in folds: # For every fold
        batch_gen = BatchGenerator(data, 
                                   fold=fold_set, 
                                   fold_cols = fold_columns, 
                                   exclude=fold, 
                                   label=clss_column, 
                                   class_columns=class_columns,
                                   pseudo=pseudo)
        for j, batch_size in enumerate(batch_sizes):
            for k, lr in enumerate(learning_rates): # Compute AUC for each learning rate
                tf.keras.backend.clear_session()
                model = FFNN_Decoder(num_inputs=num_inputs, num_outputs=num_outputs, num_layers=num_layers)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss=loss_fn,
                    metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.TruePositives(name='true_positives'), keras.metrics.AUC(name="auc")]
                )
                
                # Training
                for _ in range(num_epochs):
                    X_train, y_train = batch_gen.generate_batch_full()
                    y_train = np.array(y_train)
                    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_batches, verbose=verbose)

                # Testing
                y_true = []
                X_test, y_test = batch_gen.get_test_selective(isPositive=True)
                if X_test[0].shape[0] == 0: # Skip if no positives
                    continue

                hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)

                pred_pos = model.predict(X_test)
                y_true += y_test
                X_test, y_test = batch_gen.get_test_selective(isPositive=False, subsample=True)
                pred_neg = model.predict(X_test)
                y_true += y_test

                aucs[j][k] += [roc_auc_score(y_true, np.concatenate((pred_pos, pred_neg)))]
                del model
    # Learning rate with highest AUC
    aucs = [list(map(np.mean, row)) for row in aucs]
    batch_idx = np.argmax([np.max(row) for row in aucs])
    lr_idx = np.argmax(aucs[batch_idx])
    
    opt_lr = learning_rates[lr_idx]
    batch_size = batch_sizes[batch_idx]
    print('AUC Values {} for learning rates {} and batch size {}'.format(aucs, learning_rates, batch_size))
    
    return opt_lr, batch_size

def fit_FFNN_checkpoint(data, fold_columns, clss_column, class_columns, batch_sizes=[32,64], num_batches=1, num_epochs=100, verbose=0, num_layers=3, num_outputs = 1,
             learning_rates=[0.0001, 0.001, 0.01, 0.1], loss_fn=keras.losses.BinaryCrossentropy(), tune=True, check_dir=None,
             fold_set=None, fold=None, number=5, pseudo=True):
    num_inputs = ([dat.shape[1] - 1 - len(fold_columns) - len(class_columns) for dat in data])[0] # Only one dataset

    folds = set(reduce(lambda x,y: x+y,map(lambda x:x[fold_set].to_list() , data)))
    n = len(folds)

    print('Fold {}/{}'.format(fold, max(folds)))
    if fold not in folds:
        raise Exception('Fold {} not in fold set. Only folds is {}'.format(fold, (folds)))
    batch_gen = BatchGenerator(data, 
                                fold=fold_set, 
                                fold_cols = fold_columns, 
                                exclude=fold, 
                                label=clss_column, 
                                class_columns=class_columns,
                                pseudo=pseudo)
    
    data_train = batch_gen.get_train()
    if tune:
        lr, batch_size = FFNN_tune(data_train, fold_set, fold_columns, clss_column, class_columns, num_layers=num_layers, num_outputs=num_outputs,
                    learning_rates=learning_rates, loss_fn=loss_fn, batch_sizes=batch_sizes, num_epochs=num_epochs, num_batches=num_batches)
    else:
        lr, batch_size = learning_rates[0], batch_sizes[0]
    print('Fold Set {} Fold {}/{} Learning Rate {}'.format(fold_set, fold, n, lr))

    ## Test over ensemble
    pred_pos = []
    pred_neg = []
    for _ in range(number):
        tf.keras.backend.clear_session()
        model = FFNN_Decoder(num_inputs=num_inputs, num_outputs=num_outputs, num_layers=num_layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss_fn,
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.TruePositives(name='true_positives'), keras.metrics.AUC(name="auc")]
        )

        # Training
        for _ in range(num_epochs):
            X_train, y_train = batch_gen.generate_batch_full()
            y_train = np.array(y_train)
            hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_batches, verbose=verbose)

        # Testing
        X_test, y_test_pos = batch_gen.get_test_selective(isPositive=True) 
        if X_test[0].shape[0] == 0:
            print('No positives found.')
            X_test, y_true = batch_gen.get_test_selective(isPositive=False, subsample=True)
            pred_neg.append(list(model.predict(X_test)[:,0]))
        else:
            pred_pos.append(list((model.predict(X_test))[:,0]))
            X_test, y_test_neg = batch_gen.get_test_selective(isPositive=False, subsample=True)
            pred_neg.append(list((model.predict(X_test))[:,0]))
            y_true = y_test_pos + y_test_neg
    
    # Take average prediction over ensemble
    if len(pred_pos) == 0:
        pred_pos = None
    else:
        pred_pos = list(np.apply_along_axis(np.mean, 0, np.array(pred_pos)))
    pred_neg = list(np.apply_along_axis(np.mean, 0, np.array(pred_neg)))

    save_fp = '{}{}_{}.pickle'.format(check_dir, fold_set, fold)
    out = (pred_pos, pred_neg, y_true, lr, batch_size)

    store = Storage(out)
    with open(save_fp, 'wb') as fp:
        pickle.dump(store, fp)
    print('Saved at {}'.format(save_fp))

    del model


