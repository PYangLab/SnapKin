library(PhosR)
library(dplyr)

preprocessSnapKinTraining = function(data_file, output_filepath=NULL) {
    #    data_file       :: dataframe containing phosphoproteomic data and a column of known phosphosites denoted by y
    #    output_filepath :: string of where the output csv file is saved. 
    #    Returns a dataframe
    cols = colnames(data_file)
    if (length(intersect(cols, c('site','y'))) != 2) {
        stop('Data file is missing at least one of the following columns: "site", "y"')
    }
    
    # Extract components from dataset
    ids = data_file %>% pull(site)
    phospho.raw = data_file %>%
        select(-site,-y)
    y = data_file %>% pull(y)
    
    # Extract site and sequence information
    sites <- sapply(strsplit(ids, ";"), function(x)paste(x[1], x[2], "", sep=";"))
    sequences <- sapply(strsplit(ids, ";"), function(x)x[3])
    substrate.ids = which(y==1)
    
    # Compute sequence score
    seq.score.raw = frequencyScoring(sequences, createFrequencyMat(sequences[substrate.ids]))
    
    # Normalisation 
    phospho = (phospho.raw - min(phospho.raw))/(max(phospho.raw) - min(phospho.raw))
    seq.score = (seq.score.raw - min(seq.score.raw))/(max(seq.score.raw) - min(seq.score.raw))
    
    # Output dataframe
    df = data.frame(site=ids,
                    phospho,
                    seq.score,
                    y)
    
    if (!is.null(output_filepath)) {
        write.csv(df,
                  output_filepath,
                  row.names=FALSE)
    }
    
    return(df)
}








