# Ian Hay - 2023-02-23

import sys
import inspect
import torch
import numpy as np
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


def getTopPrediction(probs):
    return np.argmax(probs, axis=1)


# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=6XPL1Z_RegBF
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def getClassificationMetrics(predictions, labels, threshold=0.5, upperVal=1, lowerVal=0, verbose=True):
    # assumes predictions are a torch tensor object

    probs = predictions
    probs = probs.cpu().data.numpy()
    # y_pred = np.zeros(probs.shape)
    y_pred = np.where(probs >= threshold, upperVal, lowerVal)
    y_true = labels

    f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = metrics.roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average = 'micro')
    precision = metrics.precision_score(y_true, y_pred, average = 'micro')
    confusion_matrix = metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    metrics = {'f1': f1_score,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'confusion matrix': confusion_matrix}
    
    if verbose: print(f"F1: {f1_score}\tAUC: {roc_auc}\tAccuracy: {accuracy}\n"
                        +f"Recall: {recall}\tPrecision: {precision}\n"
                        +f"Confusion matrix:\n{confusion_matrix}")

    return metrics


def getClusterMetrics(pred, x=None, labels=None, supervised=False, verbose=True):
    
    # TODO holy fuck clean this up please


    # TODO
    # perplexity
    # coherence
    # log-likelihood
    if (len(list(set(pred))) < 2):
        return 0., 0., 0., 0., 0., 0., 0. # hard coded, TODO to fix in the future
    if(x is not None):
        _silhouette = metrics.silhouette_score(x, pred)
        _calinskiHarabasz = metrics.calinski_harabasz_score(x, pred)
        _daviesBouldin = metrics.davies_bouldin_score(x, pred)
        if (verbose):
            print(f"Silhouette: {_silhouette}")
            print(f"Calinski-Harabasz: {_calinskiHarabasz}")
            print(f"Davies-Bouldin: {_daviesBouldin}")

        if (supervised and labels is not None):
            _homogeneity = metrics.homogeneity_score(labels, pred)
            _completeness = metrics.completeness_score(labels, pred)
            _vMeasure = metrics.v_measure_score(labels, pred)
            _rand = metrics.adjusted_rand_score(labels, pred)
            if (verbose):
                print(f"Homogeneity: {_homogeneity}") # supervised
                print(f"Completeness: {_completeness}") # supervised
                print(f"V-measure: {_vMeasure}") # supervised
                print(f"Adjusted Rand-Index: {_rand}") # supervised
            return _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand
        return _silhouette, _calinskiHarabasz, _daviesBouldin

    if (supervised and labels is not None):
        _homogeneity = metrics.homogeneity_score(labels, pred)
        _completeness = metrics.completeness_score(labels, pred)
        _vMeasure = metrics.v_measure_score(labels, pred)
        _rand = metrics.adjusted_rand_score(labels, pred)
        if (verbose):
            print(f"Homogeneity: {_homogeneity}") # supervised
            print(f"Completeness: {_completeness}") # supervised
            print(f"V-measure: {_vMeasure}") # supervised
            print(f"Adjusted Rand-Index: {_rand}") # supervised
        return _homogeneity, _completeness, _vMeasure, _rand

