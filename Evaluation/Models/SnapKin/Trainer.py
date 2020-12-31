from tqdm import tqdm
from glob import glob
from pathlib import Path
from Snapshot import fit_FFNN_checkpoint
from Helpers import Storage
from tensorflow import keras
import tensorflow as tf
import multiprocessing as mp
import pandas as pd
import pickle
import os
import time
import warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

start = time.time()

# CPU Only
tf.config.set_visible_devices([], 'GPU')
# Number of CPU cores
num_cores = mp.cpu_count()
#num_cores = 2
# Parameters
model = 'SnapKin'
pseudo = False
nam = ''
num_batches = 10
num_snapshots = 1
num_epochs = 100

if not pseudo:
    nam ='-NoPseudo'


run_classes = ['MTOR','MAPK1']
all_classes = ['MAPK1', 'MTOR']
# directory = 'Multi-{}'.format('-'.join(all_datasets))
directory = 'Folds'
data_dir = '../../Data-50/{}/'.format(directory)
save_dir = './Data-50{}/'.format(nam)
all_datasets = ['C2C12','ESC','MLC','NBC','L1-I','L1-F','L1-R','L6']



# FFNN Parameters
learning_rates = [0.01] # [ 0.0001, 0.001, 0.01, 0.1]
optimizer = keras.optimizers.SGD(momentum=0.9)
loss_fn = keras.losses.BinaryCrossentropy()
num_layers = 3
num_folds = 50
batch_sizes= [32] # [32,64]
verbose= 0
num_outputs = 1
tune = False
fold_sets = ['F{}'.format(i + 1) for i in range(num_folds)]


# Create save directory if it doesn't exist
Path(save_dir).mkdir(parents=True, exist_ok=True)
motifs = [x + '_motif' for x in all_classes]
fold_columns = ['F{}'.format(x) for x in range(1,num_folds+1)]
class_columns = ['y.{}'.format(x) for x in all_classes]
run_class_columns = ['y.{}'.format(x) for x in run_classes]

print('-----  Using {} cores  ------'.format(num_cores))
pool = mp.Pool(num_cores)
jobs = []

for dat in all_datasets:
    for clss, clss_column in zip(run_classes, run_class_columns):
        data_fp = '{}{}_{}_FLD.csv'.format(data_dir, dat, clss)
        data = [pd.read_csv(data_fp).drop_duplicates(subset=['site'])]
        check_dir = '{}{}_{}_{}/'.format(save_dir, model, dat, clss)
        Path(check_dir).mkdir(parents=True, exist_ok=True)

        folds = list(set(data[0]['F1'].to_list()))
        folds = [x for x in folds if x > -1] # Ignore negatives
        for fold_set in fold_sets:
            for fold in folds:
                ## Skip if file exists.
                save_fp = '{}{}_{}.pickle'.format(check_dir, fold_set, fold)
                checkpt_dir = '{}Saves/{}_{}/'.format(check_dir, fold_set, fold)
                Path(checkpt_dir).mkdir(parents=True, exist_ok=True)

                if len(glob(save_fp)) > 0: 
                    print('IGNORED {}'.format(save_fp))
                    continue
                args = (data, fold_columns, clss_column, class_columns, checkpt_dir, batch_sizes, num_batches, num_epochs, verbose,
                        num_layers, num_outputs, learning_rates, loss_fn, optimizer, tune, check_dir, fold_set, fold, num_snapshots, pseudo)
                jobs.append(pool.apply_async(func=fit_FFNN_checkpoint, args=args))

pool.close()
results = []
for job in tqdm(jobs):
    results.append(job.get())

pool.join()

## Merge Folds
for dataset in all_datasets:
    for clss in run_classes:
        save = True
        data_fp = '{}{}_{}_FLD.csv'.format(data_dir,dataset,clss)
        folds = list(set(pd.read_csv(data_fp)['F1'].to_list()))

        pickle_dir = '{}{}_{}_{}/'.format(save_dir, model, dataset, clss)
        if not os.path.exists(pickle_dir):
            print('Save directory {} Missing Dataset {} and Class {}'.format(save_dir, dataset, clss))
        
        # Storage variables
        predict_pos, predict_neg = [], [] # Store prediction scores
        fold_preds, fold_labs = [], []    # Store predict
        learn_rates, batch_sizes = [], [] # Storing learned parameters

        for fold_set in fold_columns:
            preds_pos, preds_neg = [], []                  # Stores collective positive and negative prediction scores
            fold_predictions, fold_labels = [], []         # Stores positive/negative prediction scores per fold
            lrs, batch_s = [], []

            for fold in tqdm(folds):
                check_fp = '{}{}_{}.pickle'.format(pickle_dir, fold_set, fold)
                try:
                    with open(check_fp, 'rb') as fp:
                        store = pickle.load(fp)
                except:
                    print(check_fp)
                    save = False
                    continue
                pred_pos, pred_neg, y_true, lr, bs = store.data
                pred_neg = list(pred_neg)
                if pred_pos:
                    pred_pos = list(pred_pos)
                    preds_pos += pred_pos
                    preds_neg += pred_neg
                    fold_predictions.append((pred_pos, pred_neg))
                    fold_labels.append((pred_pos+pred_neg, y_true))
                else:
                    preds_neg += pred_neg
                    fold_predictions.append((None, pred_neg))
                    fold_labels.append((pred_neg, y_true))
                lrs.append(lr)
                batch_s.append(bs)

            predict_pos.append(preds_pos)
            predict_neg.append(preds_neg)

            fold_preds.append(fold_predictions)
            fold_labs.append(fold_labels)

            learn_rates.append(lrs)
            batch_sizes.append(batch_s)
        ## Saving merged data    
        if save:
            out = (predict_pos, predict_neg, fold_preds, fold_labs, learn_rates, batch_sizes)
            save_fp = '{}{}_{}_{}.pickle'.format(save_dir,model,dataset,clss)

            store = Storage(out)
            with open(save_fp, 'wb') as fp:
                pickle.dump(store, fp)
            print('Saved at {}'.format(save_fp))

print('---- FINISHED: {} HOURS ----'.format((time.time() - start)/3600))
