import os
import sys
import shutil
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Model import FFNN, CosineAnnealingLearningRateSchedule
from BatchGenerator import BatchGenerator
from pathlib import Path
from functools import reduce


ARGS = ['train_fp', 'test_fp', 'save_dir'] # Necessary arguments
args_dict = {                              # Default arguments
    'save_test_name': 'predictions.csv',
    'verbose': 1,
    'num_layers': 3,
    'num_snapshots': 10,
    'num_epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.01
}

def snapkin_python(train_fp, test_fp, save_dir, save_test_name='predictions.csv', verbose=1,
            num_layers=3, num_snapshots=10, num_epochs=150, batch_size=32, learning_rate=0.01, 
            hidden_activation=tf.nn.leaky_relu, outputs_activation=tf.nn.sigmoid,
            optimizer=keras.optimizers.SGD(momentum=0.9), loss_fn=keras.losses.BinaryCrossentropy(),
            pseudo=True):
    '''
        train_fp                     :: file path to training csv file
        test_fp                      :: file path to test csv file (requires site column for labels)
        save_dir                     :: folder path where predictions are saved 
        save_test_name               :: filename for predictions 
        verbose                      :: training output and logging - (0) none, (1) progress bar, (2) full output 
        num_layers                   :: number of layers in feed forward network 
        num_snapshots                :: number of snapshots to train 
        num_epochs                   :: number of epochs for each snapshot 
        batch_size                   :: the size of each batch during training 
        learning_rate                :: the maximum learning rate - high learning rates may mean the network won't converge 
        hidden_activation            :: activation function of hidden layers in the neural network 
        outputs_activation           :: activation function of final layer in the neural network
        optimizer                    :: optimisation algorithm used for training 
        loss_fn                      :: neural network loss function 
        pseudo                       :: whether pseudo-positives are used (true|false)
    '''
    start = time.time()
    # Check train and test exist
    if not os.path.exists(train_fp):
        print('Training data file not found at {}'.format(train_fp))
        return
    if not os.path.exists(test_fp):
        print('Test data file not found at {}'.format(test_fp))
        return 
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Open dfs 
    df_train = pd.read_csv(train_fp)
    df_test = pd.read_csv(test_fp)
    
    ## Data Check 
    col_train = df_train.columns.to_list()
    col_test = df_test.columns.to_list()

    if not set(col_train).difference(['y','site']) == set(col_test).difference(['site']):
        print('Mismatching data features in train ({}) and test data ({})'.format(len(col_train), len(col_test)))
        return
    if not 'y' in col_train:
        print("Missing labels in training data. Please include a column of labels denoted by 'y' in the training data.")
        return 
    if not 'site' in col_test:
        print("Missing site label in test data. Please include a column of site labels called 'site' in the test data.")
        return

    # Preprocessing 
    input_columns = list(set(col_train).difference(['y','site']))
    num_inputs = len(input_columns)
    ## Training Data
    train_input = df_train[input_columns]
    train_label = df_train['y']    
    
    batch_gen = BatchGenerator(train_input, train_label, pseudo=pseudo)
    
    # Train 
    model = FFNN(num_inputs, num_outputs=1, num_layers=num_layers, hidden_activation=hidden_activation, outputs_activation=outputs_activation)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )
    save_dirs = ['{}{}'.format(os.path.join(save_dir,'Saved_Models','Model_'),mod_nam) for mod_nam in range(num_snapshots)]

    for save_model_fp in save_dirs:
        ## Generate training batch
        X_train, y_train = batch_gen.get_batch()
        hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose,
                        callbacks=[CosineAnnealingLearningRateSchedule(learning_rate, num_epochs, 1, save_model_fp)])

    # Test 
    X_test = df_test[input_columns]
    sites_test = df_test['site']
    predictions = []

    for save_model_fp in save_dirs:
        model = tf.keras.models.load_model(save_model_fp)

        prediction = model.predict(X_test).flatten().tolist()
        predictions.append(prediction)

    final_predictions = list(np.apply_along_axis(np.mean, 0, np.array(predictions)))

    # Construct prediction csv dataframe
    output_file = pd.DataFrame(data={'site':sites_test,
                                     'prediction':final_predictions})

    # Save results
    save_fp = os.path.join(save_dir, save_test_name)
    output_file.to_csv(save_fp, index=False)

    print('---- FINISHED: {} HOURS ----'.format((time.time() - start)/3600))
    return output_file

def snapkin(df_train, df_test, verbose=1,
            num_layers=3, num_snapshots=10, num_epochs=150, batch_size=32, learning_rate=0.01, 
            hidden_activation=tf.nn.leaky_relu, outputs_activation=tf.nn.sigmoid,
            optimizer=keras.optimizers.SGD(momentum=0.9), loss_fn=keras.losses.BinaryCrossentropy(),
            pseudo=True):
    '''
        train_fp                     :: file path to training csv file
        test_fp                      :: file path to test csv file (requires site column for labels)
        save_dir                     :: folder path where predictions are saved 
        save_test_name               :: filename for predictions 
        verbose                      :: training output and logging - (0) none, (1) progress bar, (2) full output 
        num_layers                   :: number of layers in feed forward network 
        num_snapshots                :: number of snapshots to train 
        num_epochs                   :: number of epochs for each snapshot 
        batch_size                   :: the size of each batch during training 
        learning_rate                :: the maximum learning rate - high learning rates may mean the network won't converge 
        hidden_activation            :: activation function of hidden layers in the neural network 
        outputs_activation           :: activation function of final layer in the neural network
        optimizer                    :: optimisation algorithm used for training 
        loss_fn                      :: neural network loss function 
        pseudo                       :: whether pseudo-positives are used (true|false)
    '''
    start = time.time()
    
    ## Convert data type 
    verbose, num_layers, num_snapshots=int(verbose),int(num_layers),int(num_snapshots)
    num_epochs, batch_size = int(num_epochs), int(batch_size)

    ## Data Check 
    col_train = df_train.columns.to_list()
    col_test = df_test.columns.to_list()

    if not set(col_train).difference(['y','site']) == set(col_test).difference(['site']):
        print('Mismatching data features in train ({}) and test data ({})'.format(len(col_train), len(col_test)))
        return
    if not 'y' in col_train:
        print("Missing labels in training data. Please include a column of labels denoted by 'y' in the training data.")
        return 
    if not 'site' in col_test:
        print("Missing site label in test data. Please include a column of site labels called 'site' in the test data.")
        return

    # Preprocessing 
    input_columns = list(set(col_train).difference(['y','site']))
    num_inputs = len(input_columns)
    ## Training Data
    train_input = df_train[input_columns]
    train_label = df_train['y']    
    
    batch_gen = BatchGenerator(train_input, train_label, pseudo=pseudo)
    
    # Train 
    model = FFNN(num_inputs, num_outputs=1, num_layers=num_layers, hidden_activation=hidden_activation, outputs_activation=outputs_activation)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )
    save_dirs = ['{}{}'.format(os.path.join('./SnapKin_Saved_Models','Model_'),mod_nam) for mod_nam in range(num_snapshots)]

    for save_model_fp in save_dirs:
        ## Generate training batch
        X_train, y_train = batch_gen.get_batch()
        hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose,
                        callbacks=[CosineAnnealingLearningRateSchedule(learning_rate, num_epochs, 1, save_model_fp)])

    # Test 
    X_test = df_test[input_columns]
    sites_test = df_test['site']
    predictions = []

    for save_model_fp in save_dirs:
        model = tf.keras.models.load_model(save_model_fp)

        prediction = model.predict(X_test).flatten().tolist()
        predictions.append(prediction)

    final_predictions = list(np.apply_along_axis(np.mean, 0, np.array(predictions)))

    # Construct prediction csv dataframe
    output_file = pd.DataFrame(data={'site':sites_test,
                                     'prediction':final_predictions})

    shutil.rmtree('./SnapKin_Saved_Models', ignore_errors=True)

    print('---- FINISHED: {} HOURS ----'.format((time.time() - start)/3600))
    return output_file

def run_snapkin(args_dict):
    # Check arg existence
    keys = list(args_dict.keys())
    for arg in ARGS:
        if arg not in keys:
            print('Argument {} not found in argument file'.format(arg))
            return 
    # Run Snapkin 
    snapkin_python(train_fp=args_dict['train_fp'],
            test_fp=args_dict['test_fp'],
            save_dir=args_dict['save_dir'],
            save_test_name=args_dict['save_test_name'],
            verbose=int(args_dict['verbose']),
            num_layers=int(args_dict['num_layers']),
            num_snapshots=int(args_dict['num_snapshots']),
            num_epochs=int(args_dict['num_epochs']),
            batch_size=int(args_dict['batch_size']),
            learning_rate=float(args_dict['learning_rate']))


