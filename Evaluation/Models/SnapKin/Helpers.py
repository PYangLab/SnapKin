import numpy as np

class Storage():
    def __init__(self, data):
        self.data = data
        
def compute_ROC(pos, neg):
    thresholds = np.linspace(0,1.01,101)
    results = [[] for _ in range(len(pos))]
    for i in range(len(pos)):
        num_pos, num_neg = len(pos[i]), len(neg[i])
        for threshold in thresholds:
            tpr = sum(pos[i]>=threshold)/num_pos
            fpr = 1 - sum(neg[i]<threshold)/num_neg
            results[i].append([tpr, fpr, threshold])
    return results

def fit_ROC(fpr, tpr):
    '''
        Estimate the FPR for certain values from a given 
        ROC curve.
    '''
    fpr_intervals = np.linspace(min(fpr), max(fpr), 101)
    tpr_preds = []
    for fpr_pt in fpr_intervals:
        if (fpr_pt in fpr):
            tpr_preds.append(tpr[fpr.index[fpr == fpr_pt].min()])
        else:
            lower_fpr, lower_tpr = fpr[fpr.index[fpr < fpr_pt].min()], tpr[fpr.index[fpr < fpr_pt].min()]
            upper_fpr, upper_tpr = fpr[fpr.index[fpr >= fpr_pt].max()], tpr[fpr.index[fpr >= fpr_pt].max()]
            tpr_pred = lower_tpr + (upper_tpr - lower_tpr)/(upper_fpr - lower_fpr) * (fpr_pt - lower_fpr)
            tpr_preds.append(tpr_pred)
    return (fpr_intervals, tpr_preds)

def comp_ROC(pos, neg):
    '''
        Compue the ROC for a set of positive and negative scores.
    '''
    thresholds = np.linspace(0,1.01,100)
    results = []
    num_pos, num_neg = len(pos), len(neg)
    for threshold in thresholds:
        tpr = sum(pos>=threshold)/num_pos
        fpr = 1 - sum(neg<threshold)/num_neg
        results.append([fpr, tpr, threshold])
    return results