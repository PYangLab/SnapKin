library(pROC)
library(tidyverse)
library(tidymodels)
library(themis)
source('../../Helpers/PseudoSampler.R')

fit_RF = function(input, n_trees=500, smote=FALSE, no_pseudo=FALSE, num_models=1) {
  # Output
  # aucs - List of vectors of auc values per fold
  # scores - dataframe of sites and prediction scores
  # roc_curve - dataframe of ROC curves per fold set and fold
  # preds - list of list of prediction scores per fold
  
  # Parameters
  ## Unpacking Input
  num_repeats = length(input$fold_lab)
  fold_lab = input$fold_lab
  num_inputs = ncol(input$obs)
  dat = input$full
  class_column = input$class_name
  sites = input$additional
  
  scores = NULL
  roc_curve = NULL
  aucs = list()
  preds = list()
  
  for (fold_num in 1:num_repeats) {
    print(fold_num)
    # Retrieve folds
    fold_numbers = dat %>%
      pull(fold_lab[fold_num]) %>%
      unique()
    fold_numbers = fold_numbers[fold_numbers > -1]
    
    # Fold Set Storage
    fold_aucs = NULL
    fold_probabilities = list()
    
    for (i in fold_numbers) {
      full = dat %>%
        filter(dat[, fold_lab[fold_num]] != i)
      
      train_balanced = list()
      ## Over-sample and balance
      if (no_pseudo) {
        full = full %>%
          select(-c(all_of(fold_lab)))
        X = full %>%
          select(-all_of(class_column))
        y = full %>%
          select(all_of(class_column))
        ## Split positives and negatives
        index_positives = y > 0
        X_positives = X[as.vector(index_positives, mode = 'logical'),]
        X_unlabelled = X[!as.vector(index_positives, mode = 'logical'),]
        y_positives = y[index_positives]
        y_unlabelled = y[!index_positives]
        
        for (n.models in 1:num_models) {
          ## Determine negative data via sub-sampling
          num_negatives = nrow(X_positives)
          index_negatives = sample(1:nrow(X_unlabelled), size = num_negatives)
          X_negatives = X_unlabelled[index_negatives, ]
          y_negatives = y_unlabelled[index_negatives]
          
          ## Form training set
          X_train = rbind(X_negatives, X_positives)
          y_train = c(y_negatives, y_positives) %>%
            as.factor()
          
          train_balanced[[n.models]] = cbind(X_train, y=y_train)
        }
      }
      else if (smote) {
        X_full = full %>%
          select(-all_of(class_column), -all_of(fold_lab))
        y_full = full %>%
          select(all_of(class_column)) %>%
          pull() %>%
          as.factor()
        train_full = cbind(X_full, y=y_full)
        for (n.models in 1:num_models) {
          train_balanced[[n.models]] = recipe(y~., train_full) %>%
            step_upsample(y, over_ratio = 0.05)%>% 
            step_downsample(y, under_ratio = 1) %>%
            prep() %>%
            juice() 
        }
      }
      else { # Generate Pseudopositives
        full = full %>%
          select(-c(all_of(fold_lab)))
        X = full %>%
          select(-all_of(class_column))
        y = full %>%
          select(all_of(class_column))
        ## Split positives and negatives
        index_positives = y > 0
        X_positives = X[as.vector(index_positives, mode = 'logical'),]
        X_unlabelled = X[!as.vector(index_positives, mode = 'logical'),]
        y_positives = y[index_positives]
        y_unlabelled = y[!index_positives]
        
        # ## Pseudo Positives
        pseudos = pseudoSampler(full, class_column)
        pseudos_dat = pseudos$pseudos
        pseudos_lab = pseudos$pseudo_labs
        num_pseudos = nrow(pseudos_dat)
        
        for (n.models in 1:num_models) {
          ## Determine negative data via sub-sampling
          num_negatives = nrow(X_positives) + num_pseudos
          index_negatives = sample(1:nrow(X_unlabelled), size = num_negatives)
          X_negatives = X_unlabelled[index_negatives, ]
          y_negatives = y_unlabelled[index_negatives]
          
          ## Form training set
          X_train = rbind(X_negatives, X_positives, pseudos_dat)
          y_train = c(y_negatives, y_positives, pseudos_lab) %>%
            as.factor()
          
          train_balanced[[n.models]] = cbind(X_train, y=y_train)
        }
      }
      
      model = list()
      
      for (n.models in 1:num_models) {
        model[[n.models]] = rand_forest(mode='classification', trees=n_trees) %>%
          set_engine('ranger') %>%
          fit(y ~ ., data=train_balanced[[n.models]])
      }
      
      ## Testing set
      full_ = dat %>%
        filter(dat[, fold_lab[fold_num]] == i) %>%
        select(-c(all_of(fold_lab)))
      test_sites = sites[dat[, fold_lab[fold_num]] == i]
      X_ = full_ %>%
        select(-all_of(class_column))
      y_ = full_ %>%
        select(all_of(class_column)) %>%
        pull() %>%
        as.factor()
      
      test_full = cbind(sites=test_sites, X_, y=y_)
      test_pos = test_full %>%
        filter(y == 1) 
      test_neg = test_full %>%
        filter(y == 0) 
      
      pred_pos = NULL
      # Positive Test Data
      if (nrow(test_pos) > 0) {
        X_positives = test_pos %>% select(-sites)
        s_positives = test_pos %>% select(sites)
        
        for (n.models in 1:num_models) {
          ## Positive Prediction Score
          pred_pos_out = model[[n.models]] %>%
            predict(X_positives, type='prob')
          pred_pos = pred_pos_out[,2] %>%
            pull() %>%
            rbind(pred_pos)
        }
        pred_pos = apply(pred_pos,2,mean)
        ## Construct Positive Site Prediction Score df
        positive_score = data.frame(sites = s_positives,
                                    Score = pred_pos,
                                    Label = 1)
      }
      else {
        s_positives = NULL
        positive_score = NULL
      }
      
      # Negative Test Data
      tmp_neg = test_neg %>%
        sample_n(size=nrow(test_pos))
      s_negatives = tmp_neg %>% 
        pull(sites)
      X_negatives = tmp_neg %>% 
        select(-sites)
      
      # s_negatives = test_neg %>% pull(sites)
      # s_negatives = sample(s_negatives, size=nrow(test_pos))
      # X_negatives = test_neg %>% 
      #   filter(sites %in% s_negatives) %>%
      #   distinct(sites, .keep_all=TRUE) %>%
      #   select(-sites)
      
      pred_neg = NULL
      for (n.models in 1:num_models) {
        ## Negative Prediction Score
        pred_neg_out = model[[n.models]] %>%
          predict(X_negatives, type='prob')
        pred_neg = pred_neg_out[,2] %>%
          pull() %>%
          rbind(pred_neg)
      }
      pred_neg = apply(pred_neg,2,mean)
      ## Construct Negative Site Prediction Score df
      negative_score = data.frame(sites = s_negatives,
                                  Score = pred_neg,
                                  Label = 0)
      
      # Store Results
      ## Site Scores
      scores = rbind(positive_score, negative_score, scores)
      ## Compute AUC value and ROC curve
      if (!is.null(pred_pos)) {
        predictions = c(pred_pos, pred_neg)
        labs = c(rep(1, length(pred_pos)), rep(0, length(pred_neg)))
        roc_obj = roc(labs, predictions, quiet = TRUE)
        fold_aucs = c(fold_aucs, roc_obj$auc[1])
        roc_curve = data.frame(
          fpr = 1 - roc_obj$specificities,
          tpr = roc_obj$sensitivities,
          threshold = roc_obj$thresholds,
          fold_set = fold_num,
          fold = i
        ) %>%
          rbind(roc_curve)
      }
      ## Probabilities
      fold_probabilities[[i + 1]] = list(pred_neg = pred_neg, pred_pos =
                                           pred_pos)
    }
    aucs[[fold_num]] = fold_aucs
    preds[[fold_num]] = fold_probabilities
  }
  out = list(
    aucs = aucs,
    scores = scores,
    roc_curve = roc_curve,
    preds = preds
  )
  return (out)
}

# tune_rf = rand_forest(mode='classification', trees=n_trees,
#                       mtry = tune(), min_n=tune()) %>%
#   set_engine('ranger') 
# cv_folds = vfold_cv(train_balanced, strata = y)
# 
# res = workflow() %>%
#   add_model(tune_rf) %>%
#   add_formula(y~.) %>%
#   tune_grid(resamples=cv_folds)
