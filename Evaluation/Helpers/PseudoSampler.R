## Pseudo Sampler for one class
pseudoSampler <- function(data, class_name) {
    class_index <- which(colnames(data) == class_name)
    y <- data[,class_index]
    y_index <- which(y == 1)
    y_lab <- colnames(data)[class_index]
    
    if (length(y_index) < 2) {
        # stop('There are less than 2 positive samples. Pseudo sampling cannot be used')
        return ('Error')
    }
    
    df <- data[,-class_index]
    pseudos <- c()
    
    positives <- df[y_index,]
    combins <- combn(1:nrow(positives), 2)
    
    for (j in 1:ncol(combins)) {
        pseudos <- rbind(pseudos,colMeans(positives[combins[,j],])) 
    }
    
    pseudo_labs <- rep(1, nrow(pseudos))
    full <- cbind(pseudos, pseudo_labs)
    colnames(full)[ncol(data)] <- y_lab
    
    output <- list(pseudos=pseudos,
                   pseudo_labs=pseudo_labs,
                   full=full)
    return (output)
}


######
#
# Pseudo positive sampling hardcoded for 2 classes
# c('y.MTOR', 'y.ERK1')
#
#####

pseudoPositives <- function(dat, class_columns) {
    y_cols <- dat[, class_columns]
    Erk1.substrates.ids <- which(y_cols[,1] == 1)
    mTOR.substrates.ids <- which(y_cols[,2] == 1)
    
    ERK1.MTOR.Ids <- intersect(Erk1.substrates.ids, mTOR.substrates.ids)
    ERK1.Exclusive.Ids <- setdiff(Erk1.substrates.ids, ERK1.MTOR.Ids)
    MTOR.Exclusive.Ids <- setdiff(mTOR.substrates.ids, ERK1.MTOR.Ids)
    
    print(paste("ERK1.MTOR", length(ERK1.MTOR.Ids), 
                "ERK1.Exclusive", length(ERK1.Exclusive.Ids),
                "MTOR.Exclusive",length(MTOR.Exclusive.Ids)))
    
    class_index = match(class_columns, colnames(dat))
    df <- dat[,-class_index]
    
    pseudos <- c()
    
    # ERK1 Pseudos
    if (length(ERK1.Exclusive.Ids) > 1) {
        positives <- df[ERK1.Exclusive.Ids,]
        combins <- combn(1:nrow(positives), 2)
        tmp.pseudos <- c()
        for (j in 1:ncol(combins)) {
            tmp.pseudos <- rbind(tmp.pseudos, colMeans(positives[combins[,j],]))
        }
        tmp.pseudos <- cbind(tmp.pseudos, "y.ERK1"=rep(1, ncol(combins)), 
                             "y.MTOR"=rep(0, ncol(combins)))
        pseudos <- rbind(pseudos, tmp.pseudos)
    }
    
    # MTOR Pseudos
    if (length(MTOR.Exclusive.Ids) > 1) {
        positives <- df[MTOR.Exclusive.Ids,]
        combins <- combn(1:nrow(positives), 2)
        tmp.pseudos <- c()
        for (j in 1:ncol(combins)) {
            tmp.pseudos <- rbind(tmp.pseudos, colMeans(positives[combins[,j],]))
        }
        tmp.pseudos <- cbind(tmp.pseudos, "y.ERK1"=rep(0, ncol(combins)), 
                             "y.MTOR"=rep(1, ncol(combins)))
        pseudos <- rbind(pseudos, tmp.pseudos)   
    }
    
    # ERK1 + MTOR Pseudos
    if (length(ERK1.MTOR.Ids) > 1) {
        positives <- df[ERK1.MTOR.Ids,]
        combins <- combn(1:nrow(positives), 2)
        tmp.pseudos <- c()
        for (j in 1:ncol(combins)) {
            tmp.pseudos <- rbind(tmp.pseudos, colMeans(positives[combins[,j],]))
        }
        tmp.pseudos <- cbind(tmp.pseudos, "y.ERK1"=rep(1, ncol(combins)), 
                             "y.MTOR"=rep(1, ncol(combins)))
        pseudos <- rbind(pseudos, tmp.pseudos)    
    }
    
    
    return (pseudos)
}

