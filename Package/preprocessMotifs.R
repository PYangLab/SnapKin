library(phosR)
library(dplyr)

preprocessMotfs = function(data_file) {
    '''
        data_file :: dataframe containing phosphoproteomic data and a column of known phosphosites denoted by y
    '''
    cols = colnames(data_file)
    if length(intersect(cols, c('site','y'))) != 2 {
        stop('Data file is missing at least one of the following columns: "site", "y"')
    }
    
}