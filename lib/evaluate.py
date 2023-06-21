import os
from sklearn.metrics import roc_curve, auc
# from scipy.optimize import brentq
# from scipy.interpolate import interp1d

def evaluate(labels, scores):
    labels = labels.cpu()
    scores = scores.cpu()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return roc_auc