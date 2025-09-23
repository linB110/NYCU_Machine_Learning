import numpy as np
import matplotlib.pyplot as plt
from HW1_LDA import LDA_classifier

def binarize_label(data_label) : 
    data_label = np.asarray(data_label)
    min = data_label.min()
    
    for i in range(len(data_label)) : 
        if data_label[i] == min :
            data_label[i] = 0
        else : 
            data_label[i] = 1
    
    return data_label

def prediction_score(model, input_data_point) :
    weight = model.model_weight
    bias = model.model_bias
    
    return input_data_point @ weight + bias

def confusion_at_threshold(scores, y_true_bin, thr):
    y_pred = (scores >= thr).astype(int)
    
    TP = np.sum((y_true_bin == 1) & (y_pred == 1))
    FP = np.sum((y_true_bin == 0) & (y_pred == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred == 0))
    TN = np.sum((y_true_bin == 0) & (y_pred == 0))
    
    return TP, FP, FN, TN

def tpr_fpr_from_confusionMatrix(TP, FP, FN, TN):
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    return TPR, FPR 

def roc_curve_from_scores(scores, y_true_bin):
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    tprs, fprs = [], []
    for thr in thresholds:
        TP, FP, FN, TN = confusion_at_threshold(scores, y_true_bin, thr)
        TPR, FPR = tpr_fpr_from_confusionMatrix(TP, FP, FN, TN)
        tprs.append(TPR); fprs.append(FPR)
    fprs = np.asarray(fprs); tprs = np.asarray(tprs)
    order = np.argsort(fprs, kind="mergesort")
    return fprs[order], tprs[order], thresholds[order]

def auc_trapezoid(fprs, tprs):
    return float(np.trapz(tprs, fprs))
