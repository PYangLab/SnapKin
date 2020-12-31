library(pROC)
library(PRROC)
library(class)
library(tidyverse)

get_KNN_probs = function(knn_out) {
  ## Get the probability of predicting class 1
  if (length(knn_out)  < 1) {
    stop("KNN output has length less than 1.")
  }
  probs = c()
  knn_probs = attr(knn_out, "prob")
  prob = NULL
  
  for (i in 1:length(knn_out)) {
    # Find the probability of class 1
    prob = if (knn_out[i] == 1)
      knn_probs[i]
    else
      (1 - knn_probs[i])
    probs = c(probs, prob)
  }
  return(probs)
}

cross_val_tune = function(dat, class_column, fold, fold_lab, no_pseudo=FALSE) {
  fold_numbers = dat %>%
    pull(fold) %>%
    unique()
  aucs = NULL
  
  # Choose range of k values based off number of positives
  num_pos = sum(dat[class_column])
  if (num_pos == 0) {
    return (NULL)
  }
  params = seq(1, (2 * num_pos), 2)
  
  for (k in params) {
    tmp_aucs = NULL
    fold_numbers = dat %>%
      pull(fold) %>%
      unique()
    for (i in fold_numbers) {
      full = dat %>%
        filter(dat[, fold] != i) %>%
        select(-c(all_of(fold_lab)))
      X = full %>%
        select(-all_of(class_column))
      y = full %>%
        select(all_of(class_column))
      
      ## Split positives and negatives
      index_positives = y > 0
      X_positives = X[as.vector(index_positives, mode = 'logical'), ]
      X_unlabelled = X[!as.vector(index_positives, mode = 'logical'), ]
      y_positives = y[index_positives]
      y_unlabelled = y[!index_positives]
      
      if (no_pseudo) {
        ## Determine negative data via sub-sampling
        num_negatives = nrow(X_positives)
        index_negatives = sample(1:nrow(X_unlabelled), size = num_negatives)
        X_negatives = X_unlabelled[index_negatives, ]
        y_negatives = y_unlabelled[index_negatives]
        
        ## Form training set
        X_train = rbind(X_negatives, X_positives)
        y_train = c(y_negatives, y_positives)
      }
      else {
        # ## Pseudo Positives
        pseudos = pseudoSampler(full, class_column)
        pseudos_dat = pseudos$pseudos
        pseudos_lab = pseudos$pseudo_labs
        num_pseudos = nrow(pseudos_dat)
        
        ## Determine negative data via sub-sampling
        num_negatives = nrow(X_positives) + num_pseudos
        index_negatives = sample(1:nrow(X_unlabelled), size = num_negatives)
        X_negatives = X_unlabelled[index_negatives, ]
        y_negatives = y_unlabelled[index_negatives]
        
        ## Form training set
        X_train = rbind(X_negatives, X_positives, pseudos_dat) 
        y_train = c(y_negatives, y_positives, pseudos_lab) %>%
          as.factor()
      }
      
      
      if (nrow(X_train) < k) {
        next
      }
      
      ## Testing set
      full = dat %>%
        filter(dat[, fold] == i) %>%
        select(-c(all_of(fold_lab)))
      X_ = full %>%
        select(-all_of(class_column))
      y_ = full %>%
        select(all_of(class_column))
      
      # Positive Test Data
      index_positives = y_ > 0
      if (sum(index_positives) > 0) {
        X_positives = as.matrix(X_[as.vector(index_positives, mode = 'logical'), ])
        ## Prediction
        pred_pos_out = knn(X_train,
                           X_positives,
                           y_train,
                           k = k,
                           prob = TRUE)
        pred_pos = get_KNN_probs(pred_pos_out)
      }
      else {
        next
        
      }
      
      # Negative Test Data
      ## Subsample
      num_neg_preds = sum(index_positives)
      index_negatives = which(y_ == 0)
      index_negatives = sample(index_negatives, size=num_neg_preds)
      
      X_negatives = X_[index_negatives, ]

      ## Prediction
      pred_neg_out = knn(X_train,
                         X_negatives,
                         y_train,
                         k = k,
                         prob = TRUE)
      pred_neg = get_KNN_probs(pred_neg_out)
      
      ## AUC Value
      # predictions = c(pred_pos, pred_neg)
      # labs = c(rep(1, length(pred_pos)), rep(0, length(pred_neg)))
      # roc_obj = roc(labs, predictions, quiet = TRUE)
      # tmp_aucs = c(tmp_aucs, roc_obj$auc[1])
      
      pr_tmp = pr.curve(scores.class0=pred_pos, 
                        scores.class1=pred_neg)
      tmp_aucs = c(tmp_aucs, pr_tmp$auc.integral)
      
    }
    if (!is.null(tmp_aucs)) {
      aucs[[k]] = mean(tmp_aucs)
    }
  }
  if (is.null(aucs)) {
    return (NULL)
  }
  k_arg = which.max(unlist(aucs))
  k = params[k_arg]
  return (k)
}

fit_KNN = function(input, no_pseudo=FALSE, num_models=1) {
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
      X = full %>%
        select(-all_of(class_column))
      y = full %>%
        select(all_of(class_column))
      
      
      ## Nest Cross-Validation for Hyper-Parameter Tuning
      k = cross_val_tune(full, class_column, fold_lab[fold_num], fold_lab, no_pseudo=no_pseudo)
      if (is.null(k)) {
        next
      }
      
      ## Split positives and negatives
      index_positives = y == 1
      X_positives = X[as.vector(index_positives, mode = 'logical'), ]
      X_unlabelled = X[!as.vector(index_positives, mode = 'logical'), ]
      y_positives = y[index_positives]
      y_unlabelled = y[!index_positives]
      
      X.train = list()
      y.train = list()
      
      for (n.model in 1:num_models) {
        if (no_pseudo) {
          ## Determine negative data via sub-sampling
          num_negatives = nrow(X_positives)
          index_negatives = sample(1:nrow(X_unlabelled), size = num_negatives)
          X_negatives = X_unlabelled[index_negatives, ]
          y_negatives = y_unlabelled[index_negatives]
          
          ## Form training set
          X_train = rbind(X_negatives, X_positives) %>%
            select(-c(all_of(fold_lab)))
          y_train = c(y_negatives, y_positives)
        }
        else {
          # ## Pseudo Positives
          pseudos = pseudoSampler(full, class_column)
          pseudos_dat = pseudos$pseudos
          pseudos_lab = pseudos$pseudo_labs
          num_pseudos = nrow(pseudos_dat)
          
          ## Determine negative data via sub-sampling
          num_negatives = nrow(X_positives) + num_pseudos
          index_negatives = sample(1:nrow(X_unlabelled), size = num_negatives)
          X_negatives = X_unlabelled[index_negatives, ]
          y_negatives = y_unlabelled[index_negatives]
          
          ## Form training set
          X_train = rbind(X_negatives, X_positives, pseudos_dat) %>%
            select(-c(all_of(fold_lab)))
          y_train = c(y_negatives, y_positives, pseudos_lab) %>%
            as.factor()
        }
        X.train[[n.model]] = X_train
        y.train[[n.model]] = y_train
      }
      
      ## Testing set
      full = dat %>%
        filter(dat[, fold_lab[fold_num]] == i) %>%
        select(-c(all_of(fold_lab)))
      test_sites = sites[dat[, fold_lab[fold_num]] == i]
      X_ = full %>%
        select(-all_of(class_column))
      y_ = full %>%
        select(all_of(class_column))
      
      # Positive Test Data
      index_positives = y_ == 1
      pred_pos = NULL
      if (sum(index_positives) > 0) {
        X_positives = as.matrix(X_[as.vector(index_positives, mode = 'logical'), ])
        s_positives = test_sites[index_positives]
        
        for (n.model in 1:num_models) {
          ## Positive Prediction Score
          pred_pos_out = knn(X.train[[n.model]],
                             X_positives,
                             y.train[[n.model]],
                             k = k,
                             prob = TRUE)
          pred_pos = rbind(pred_pos,get_KNN_probs(pred_pos_out))
        }
        pred_pos = apply(pred_pos,2,mean)
        ## Construct Positive Site Prediction Score df
        positive_score = data.frame(Site = s_positives,
                                    Score = pred_pos,
                                    Label = 1)
      }
      else {
        s_positives = NULL
        positive_score = NULL
      }
      
      # Negative Test Data
      ## Subsample
      num_neg_preds = sum(index_positives)
      index_negatives = which(y_ == 0)
      index_negatives = sample(index_negatives, size=num_neg_preds)
      
      X_negatives = X_[index_negatives, ]
      s_negatives = test_sites[index_negatives]
      
      
      pred_neg = NULL
      for (n.model in 1:num_models) {
        ## Negative Prediction Score
        pred_neg_out = knn(X.train[[n.model]],
                           X_negatives,
                           y.train[[n.model]],
                           k = k,
                           prob = TRUE)
        pred_neg = rbind(pred_neg,get_KNN_probs(pred_neg_out))
      }
      
      pred_neg = apply(pred_neg,2,mean)
      ## Construct Negative Site Prediction Score df
      negative_score = data.frame(Site = s_negatives,
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
