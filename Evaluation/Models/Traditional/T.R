library(tidyverse)
source('../../Helpers/Measures.R')
source('../../Helpers/Processing.R')
source('../../Helpers/PseudoSampler.R')
# source('./Models.R')

source('./Models/KNN.R')
source('./Models/XG.R')
source('./Models/SVM.R')
source('./Models/LR.R')
source('./Models/NB.R')
source('./Models/RF.R')



require('ranger')
require('xgboost')
require('klaR')
require('kernlab')
require('discrim') 
require('foreach')
require('doParallel')
require('tidymodels')
require('themis')
select = dplyr::select

## Parameters
num_folds = 50
num_models = 1
no_pseudo = TRUE

## File Names
data_dir = 'Folds'
data_main = '../../Data-50/'
save_dir = './Data-50'

## Class and Datasets Information
run_classes = c('MAPK1', 'MTOR')
all_classes = c('MAPK1', 'MTOR')
all_datasets = c('C2C12', 'ESC','MLC', 'NBC', 'L1-I', 'L1-F', 'L1-R', 'L6')

if (no_pseudo) {
    save_dir = paste(save_dir, 'NoPseudo', sep='-')
}
if (num_models > 1) {
    save_dir = paste(save_dir, 'Ensemble', sep='-')
}
models = c('KNN','SVM','LR','RF','XG','NB')

motifs = paste(all_classes, 'motif', sep='_')
run_datasets = all_datasets 

registerDoParallel(cores=detectCores())

## Create save directory if it doesn't exist
if (!dir.exists(save_dir)) {
    dir.create(save_dir)
}


ovrll = Sys.time()
foreach (model=models)  %:%
    foreach(dataset=run_datasets) %:% 
    foreach(class=run_classes) %dopar% {
        data_file = paste(data_main, data_dir,'/',dataset, '_', class, '_FLD.csv', sep='')
        save_file = paste('./', save_dir, '/', model,'_', dataset, '_', class, '.RData', sep = '')
        if (file.exists(save_file)) {
            return (NULL)
        }
        class_column = paste('y', class, sep='.')
        class_columns = paste('y', all_classes, sep='.')
        fold_lab = paste('F',1:num_folds, sep='')
        others = c('site', setdiff(class_columns, class_column))
        if (!file.exists(data_file)) {
            print(data_file)
        }
        else {
            # Pre-processing
            dat = read.csv(data_file)
            input = preprocess_multi(dat, fold_lab, class_column, class_columns, others, 'site')
            # Fit
            start_time = Sys.time()
            
            if (model == 'LR') {
                out = fit_LR(input, num_models = num_models, no_pseudo=no_pseudo)
            }
            else if (model == 'SVM') {
                out = fit_SVM(input, num_models = num_models, no_pseudo=no_pseudo)
            }
            else if (model == 'NB') {
                out = fit_NB(input, num_models = num_models, no_pseudo=no_pseudo)
            }
            else if (model == 'KNN') {
                out = fit_KNN(input, num_models = num_models, no_pseudo=no_pseudo)
            }
            else if (model == 'RF') {
                out = fit_RF(input, num_models = num_models, no_pseudo=no_pseudo)
            }
            else if (model == 'XG') {
                out = fit_XG(input, num_models = num_models, no_pseudo=no_pseudo)
            }
            else {
                stop('Model not found')
            }
            
            print(Sys.time() - start_time)
            # Save results
            save(out,
                 class, dataset, model,
                 file=save_file)
        }
    }
print(Sys.time() - ovrll)