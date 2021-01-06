#' SnapKin Preprocessing Function
#'
#' This function preprocesses Phosphoproteomic data so that it may be used for
#' training a SnapKin model.
#'
#' @param training_set PhosphoExperiment Object containing training set. Assays
#' must contain a 'y' column representing labels.
#' @param test_set PhosphoExperiment Object containing test set. If NULL, the
#' training_set is used.
#' @param train_ids (Optional) Vector of characters to label the train set.
#' @param test_ids (Optional) Vector of characters to label the test set.
#'
#' @return A list of dataframes. training stores the processed training data.
#' test stores the processed test data.
#' @examples
#' @export
preprocessSnapKin = function(train, train_ids=NULL, test=NULL, test_ids = NULL) {

    training_set = train@assays@data[[1]] %>% data.frame()
    cols = colnames(training_set)
    if (length(intersect(cols, c('y'))) != 1) {
        stop('Training set is missing the following column "y"')
    }

    if (is.null(test)) {
        test_set = training_set %>%
            dplyr::select(-y)
        test_ds = train_ids
    }
    else {
        test_set = test@assays@data[[1]] %>% data.frame()
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

    if (is.null(train_ids)) {
        train_ids = train@Sequence
    }

    if (is.null(test_ids)) {
        if (is.null(test)) {
            test_ids = train_ids
        }
        else {
            test_ids = test@Sequence
        }
    }

    # Extract components from dataset
    phospho.raw = training_set %>%
        dplyr::select(-y)
    phospho.raw.test = test_set
    y = training_set %>% dplyr::pull(y)

    # Extract site and sequence information
    sequences = train@Sequence
    sequences.test = if (is.null(test)) train@Sequence else test@Sequence
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
    train_df = data.frame(site=train_ids,
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








