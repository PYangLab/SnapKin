#' @export
preprocessSnapKin = function(training_set, training_ids=NULL,
                             test_set=NULL, test_ids=NULL) {
    # training_set      :: dataframe containing phosphoproteomic data,
    #                     a column of known phosphosites denoted by y,
    #                     and site information as rownames.
    # train_ids         :: array of site informations for the training set.
    #                     If not null, replaces training set rownames.
    # test_set          :: dataframe containing phosphoproteomic data,
    #                     and site information as rownames. If null, the training
    #                     set is used.
    # test_ids          :: array of site informations for the test set.
    #                     If not null, replaces training set rownames.
    # Returns a list of two dataframes
    #   training        :: stores the processed training data
    #   test            :: stores the processed test data
    cols = colnames(training_set)
    if (length(intersect(cols, c('y'))) != 1) {
        stop('Training set is missing the following column "y"')
    }

    # Preprocessing
    training_set = training_set %>% data.frame()

    if (is.null(test_set)) {
        test_set = training_set %>%
            dplyr::select(-y)
    }
    else {
        test_cols = colnames(test_set)
        if (length(setdiff(cols, test_cols)) != 1) {
            stop(paste('Training and test set features are mismatching.
                       The training set should only have one additional feature, "y",
                       compared with the test set.'))
        }
    }

    # Check data is numeric
    if (sum(!unlist(lapply(training_set,is.numeric))) != 0) {
        stop('Training set should not contain non-numeric data')
    }
    if (sum(!unlist(lapply(test_set,is.numeric))) != 0) {
        stop('Test set should not contain non-numeric data')
    }

    # Extract components from dataset
    ids = if (is.null(training_ids)) rownames(training_set) else training_ids
    test_ids = if (is.null(test_ids)) rownames(test_set) else test_ids

    phospho.raw = training_set %>%
        dplyr::select(-y)
    phospho.raw.test = test_set
    y = training_set %>% dplyr::pull(y)

    # Extract site and sequence information
    sequences = sapply(strsplit(ids, ';'), function(x)x[3])
    sequences.test = sapply(strsplit(test_ids,';'), function(x)x[3])
    substrate.ids = which(y==1)

    # Compute sequence score
    freq.mat = PhosR::createFrequencyMat(sequences[substrate.ids])
    seq.score.raw = PhosR::frequencyScoring(sequences, freq.mat)
    seq.score.raw.test = PhosR::frequencyScoring(sequences.test, freq.mat)

    # Normalisation
    phospho = (phospho.raw - min(phospho.raw))/(max(phospho.raw) - min(phospho.raw))
    phospho.test = (phospho.raw.test - min(phospho.raw.test))/(max(phospho.raw.test) - min(phospho.raw.test))
    seq.score = (seq.score.raw - min(seq.score.raw))/(max(seq.score.raw) - min(seq.score.raw))
    seq.score.test = (seq.score.raw.test - min(seq.score.raw.test))/(max(seq.score.raw.test) - min(seq.score.raw.test))

    # Output dataframes
    train_df = data.frame(site=ids,
                          phospho,
                          score=seq.score,
                          y)
    test_df = data.frame(site=test_ids,
                         phospho.test,
                         score=seq.score.test)

    output = list(training=train_df,
                  test=test_df)

    return(output)
}








