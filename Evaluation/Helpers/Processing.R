preprocess <- function(dat, fold_lab, class_column, additional_removal=NULL) {
    ## Preprocess data for train model
    ## target_class - Single target class
    
    class_index = match(class_column, colnames(dat))
    fold_index = match(fold_lab, colnames(dat))
    additional_index = match(additional_removal, colnames(dat))
    
    class_lab = dat[,class_index]
    fold_cols = dat[,fold_index]
    obs = dat[,-c(class_index, fold_index, additional_index)]
    full= dat
    output = list(obs=obs,
                  class_labs=class_lab,
                  fold_lab=fold_lab,
                  full=full,
                  class_name=class_column)
    return (output)
}

preprocess_multi <- function(dat, fold_lab, class_column, class_columns, additional_removal=NULL, additional=NULL) {
    ## Preprocess data for train model
    ## target_class - Single target class
    
    class_indexes = match(class_columns, colnames(dat))
    target_class = match(class_column, colnames(dat))
    fold_index = match(fold_lab, colnames(dat))
    additional_index = match(additional_removal, colnames(dat))
    additional_keep_index = match(additional, colnames(dat))
    
    class_lab = dat[,target_class]
    fold_cols = dat[,fold_index]
    obs = dat[,-c(class_indexes, fold_index, additional_index)]
    full= dat[,-c(additional_index)]
    additional = dat[,additional_keep_index]
    full_additional = 
    output = list(obs=obs,
                  class_labs=class_lab,
                  fold_lab=fold_lab,
                  full=full,
                  additional=additional,
                  class_name=class_column)
    return (output)
}

preprocess_multiDataset <- function(datasets, dataset_names, 
                                    fold_lab, num_folds, folds_per_set=5,
                                    class_column, class_columns,
                                    additional_removal=NULL, additional=NULL) {
  obs = list()
  full = list()
  additional_dfs = list()
  
  for (i in 1:length(datasets)) {
    dat = datasets[[dataset_names[i]]]
    class_indexes = match(class_columns, colnames(dat))
    target_class = match(class_column, colnames(dat))
    fold_index = match(fold_lab, colnames(dat))
    additional_index = match(additional_removal, colnames(dat))
    additional_keep_index = match(additional, colnames(dat))
    
    
    obs[[i]] = dat[,-c(class_indexes, fold_index, additional_index)]
    full[[i]]= dat[,-c(additional_index)]
    additional_dfs[[i]] = dat[,additional_keep_index]
    
  }
  
  dataset_names = dataset_names 
  class_name=class_column
  fold_lab=fold_lab
  
  ## Returns dataframes with unused class columns removed
  output=list(
    obs=obs,
    full=full,
    additional=additional_dfs,
    dataset_names=dataset_names,
    class_name=class_column, 
    fold_lab=fold_lab,
    num_folds=num_folds,
    folds_per_set=folds_per_set
  )
  return (output)
}

get_class_index <- function(data, class_columns) {
    return (which(colnames(data) %in% class_columns))
}

get_positives_index_vector = function(vector) {
    return (which(vector > 0))
}

get_positives_index = function(matrix) {
    ## Find indexes of positive observations
    ## Matrix of y labels
    return (which(apply(matrix, 1, max) > 0))
}

get_positive_distribution <- function(y) {
    return (apply(y, 2, sum))
}