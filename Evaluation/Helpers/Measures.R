otherMeasures = function(preds) {
    sensitivity = specificity = f1 = NULL
    for (foldset in 1:length(preds)) {
        for (fold in 1:length(preds[[foldset]])) {
            pos = preds[[foldset]][[fold]]$pred_pos
            neg = preds[[foldset]][[fold]]$pred_neg 
            
            TP = sum(pos >= 0.5)
            FN = sum(pos < 0.5)
            TN = sum(neg < 0.5)
            FP = sum(neg >= 0.5)
            
            sensitivity = c(sensitivity, TP/(TP + FN))
            specificity = c(specificity, TN/(TN + FN))
            f1 = c(f1, 2*TP/(2*TP + FP + FN))
        }
    }
    return (list(sensitivity=mean(sensitivity),
                 sensitivity_sd=sd(sensitivity),
                 specificity=mean(specificity),
                 specificity_sd=sd(specificity),
                 f1=mean(f1),
                 f1_sd=sd(f1)))
}


## ROC
fit_ROC = function(fpr, tpr) {
    # fpr in increasing order
    fpr_intervals = seq(min(fpr),max(fpr),0.01)
    tpr_preds = c()
    for (j in 1:length(fpr_intervals)) {
        if (fpr_intervals[j] %in% fpr) {
            tpr_preds = c(tpr_preds, tpr[which.max(fpr_intervals[j] == fpr)])
        }
        else {
            lower_fpr = fpr[max(which(fpr < fpr_intervals[j]))]
            upper_fpr = fpr[min(which(fpr > fpr_intervals[j]))]
            lower_tpr = tpr[max(which(fpr < fpr_intervals[j]))]
            upper_tpr = tpr[min(which(fpr > fpr_intervals[j]))]
            tpr_pred = lower_tpr + (upper_tpr - lower_tpr)/(upper_fpr - lower_fpr) * (fpr_intervals[j] - lower_fpr)
            tpr_preds = c(tpr_preds, tpr_pred)
        }
    }
    return (list(fpr=fpr_intervals, tpr=tpr_preds))
}

get_ROC_values = function(pred_pos, pred_neg, num_points = 100, returnThresholds = FALSE, printConfusion = TRUE) {
    thresholds = ((num_points+1):0)/num_points
    true_labels = factor(c(rep(1, length(pred_pos)), rep(0, length(pred_neg))), levels=c(1,0))
    preds = c(pred_pos, pred_neg)
    tpr = fpr = c()
    
    for (threshold in thresholds) {
        pred_labels = factor((preds >= threshold) + 0, levels=c(1,0))
        confusion = table(pred_labels, true_labels)
        if (threshold == 0.5 && printConfusion) {
            print(confusion)
        }
        tpr = c(tpr, true_positive(confusion))
        fpr = c(fpr, 1 - true_negative(confusion))
    }
    
    out = if (returnThresholds) cbind(fpr, tpr, thresholds) else cbind(fpr, tpr)
    return (out)
}

library(MESS)
## AUC
get_AUC = function(df) {
    auc_vals = c()
    for (i in unique(df[,4])) {
        tmp = df %>%
            filter(df[,4] == i)
        auc_vals = c(auc_vals,auc(tmp[,1], tmp[,2]))
    }
    return (auc_vals)
}

## Performance Measures
# Assumes true positive is at (1,1)

true_positive <- function(matrix) {
    score <- matrix[1,1]/(matrix[1,1] + matrix[2,1])
    if (is.nan(score) | is.na(score)) {
        return (0)
    }
    return (score)
}

true_negative <- function(matrix) {
    score <- matrix[2,2]/(matrix[2,2] + matrix[1,2])
    if (is.nan(score) | is.na(score)) {
        return (0)
    }
    return (score)
}

## Results Helpers
otherMeasures = function(preds) {
    sensitivity = specificity = f1 = NULL
    for (foldset in 1:length(preds)) {
        for (fold in 1:length(preds[[foldset]])) {
            pos = preds[[foldset]][[fold]]$pred_pos
            neg = preds[[foldset]][[fold]]$pred_neg 
            
            TP = sum(pos >= 0.5)
            FN = sum(pos < 0.5)
            TN = sum(neg < 0.5)
            FP = sum(neg >= 0.5)
            
            sensitivity = c(sensitivity, TP/(TP + FN))
            specificity = c(specificity, TN/(TN + FN))
            f1 = c(f1, 2*TP/(2*TP + FP + FN))
        }
    }
    return (list(sensitivity=mean(sensitivity),
                 sensitivity_sd=sd(sensitivity),
                 specificity=mean(specificity),
                 specificity_sd=sd(specificity),
                 f1=mean(f1),
                 f1_sd=sd(f1)))
}

compute_precrec = function(pos,neg) {
    thresholds = seq(-0.01,1.01,0.01)
    prec = reca = NULL 
    for (t in thresholds) {
        TP = sum(pos >= t)
        FN = sum(pos < t)
        TN = sum(neg < t)
        FP = sum(neg >= t)
        
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        
        if (is.nan(precision)) {
            precision = 1
        }
        
        prec = c(prec,precision)
        reca = c(reca,recall)
    }
    return(data.frame(Threshold=thresholds,Precision=prec,Recall=reca))
}

precisionRecall = function(preds) {
    aucs = NULL
    curv = NULL
    for (foldset in 1:length(preds)) {
        for (fold in 1:length(preds[[foldset]])) {
            pos = preds[[foldset]][[fold]]$pred_pos
            neg = preds[[foldset]][[fold]]$pred_neg 
            
            tmp = pr.curve(scores.class0 = pos, 
                           scores.class1 = neg)
            
            aucs = c(aucs, tmp$auc.integral)
            curv = compute_precrec(pos,neg) %>%
                rbind(curv)
        }
    }
    
    curv= curv %>%
        group_by(Threshold) %>%
        summarise(Recall=mean(Recall),
                  Precision=mean(Precision)) %>%
        arrange(desc(Precision))
    
    return (list(mean=mean(aucs), sd=sd(aucs), pr=aucs, curv=curv))
}

