% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/preprocessSnapKin.R
\name{preprocessSnapKin}
\alias{preprocessSnapKin}
\title{SnapKin Preprocessing Function}
\usage{
preprocessSnapKin(train, train_ids = NULL, test = NULL, test_ids = NULL)
}
\arguments{
\item{train_ids}{(Optional) Vector of characters to label the train set.}

\item{test_ids}{(Optional) Vector of characters to label the test set.}

\item{training_set}{PhosphoExperiment Object containing training set. Assays
must contain a 'y' column representing labels.}

\item{test_set}{PhosphoExperiment Object containing test set. If NULL, the
training_set is used.}
}
\value{
A list of dataframes. training stores the processed training data.
test stores the processed test data.
}
\description{
This function preprocesses Phosphoproteomic data so that it may be used for
training a SnapKin model.
}
