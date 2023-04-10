# Ian Hay - 2023-03-27
# https://github.com/hayitsian/General-Index-Visualization

import copy
import numpy as np
from sklearn.base import TransformerMixin
from sklearn import metrics

import util # local file
from model.basemodel import BaseModel # local file
from model.clusteringmodel import ClusteringModel
from model.classifiermodel import ClassifierModel

class BaseEvaluator(BaseModel, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        """
        Takes in and trains on the data `x` to return desired features `y`.
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def predict(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : ndarray[float] : list of datapoints n samples by d features
        Returns:
            - ypred : ndarray[int] : topic prediction of n samples bc y classes
        """
        util.raiseNotDefined()


    # TODO
    def save(self):
        """
        Saves this model and any associated experiments to a .txt file.\n
        Returns:
            - filename : str : the filename this model's .txt file was saved to
        """
        util.raiseNotDefined()

    # TODO: for view
    def __dict__(self):
        """
        Represents this model as a string.\n
        Returns:
            - tostring : str : string representation of this model.
        """
        util.raiseNotDefined()


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class ClusterEvaluator(BaseEvaluator):

    def __init__(self, yTrue=None, xEmb=None, vocab=None, model:ClusteringModel=None, _baseVal=0.):
        super().__init__()
        self.outMetricList = ["perplexity", "homogeneity", "completeness", "silhouette"]#, "coherence"] # TODO hard coded
        self.metricDict = {_metric: _baseVal for _metric in self.outMetricList}
        self.model = model
        self.yTrue = yTrue
        self.xEmb = xEmb
        self.vocab = vocab

    def fit(self, x, y=None):
        return self
    
    def predict(self, x, y):
        """
        NOTE: weird syntax for this one. This is done solely to match SKLearn's Estimator to fit into
        the pipeline as the final estimator. 

        Parameters:
        x: data used to test the underlying model - e.g., texts
        y: predicted classes for the model
        yTrue: true class labels, optional
        xEmb: embedding of x, optional
        
        Returns:
        y: predicted classes for the model
        metrics: dict of metrics for the predicted clusters
        """
        # "perplexity", "coherence", "homogeneity", "completeness", "silhouette",

        if (len(list(set(y))) < 2):
            return self.metricDict

        if (self.xEmb is not None):
            _silhouette = metrics.silhouette_score(self.xEmb, y)
            self.metricDict["silhouette"] = _silhouette
        if (self.yTrue is not None):
            _homogeneity = metrics.homogeneity_score(self.yTrue, y)
            _completeness = metrics.completeness_score(self.yTrue, y)
            self.metricDict["homogeneity"] = _homogeneity
            self.metricDict["completeness"] = _completeness
        if (self.model is not None):
            _perplexity = self.model.perplexity(x)
            # _coherence = self.model.coherence(x, self.vocab)
            self.metricDict["perplexity"] = _perplexity
            # self.metricDict["coherence"] = _coherence
        
        return self.metricDict
        

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class ClassifierEvaluator(BaseEvaluator):


    def __init__(self, yTrue, model:ClassifierModel=None, probability=False, threshold=0.5, upperVal=1, lowerVal=0, _baseVal=0.):
        super().__init__()
        self.outMetricList = ["accuracy", "recall", "precision", "f1", "confusion matrix"] # TODO hard coded
        self.metricDict = {_metric: _baseVal for _metric in self.outMetricList}

        self.yTrue = yTrue
        self.model = model
        self.probability = probability
        self.threshold = threshold
        self.upperVal = upperVal
        self.lowerVal = lowerVal

    
    def fit(self, x, y=None):
        return self
    

    def predict(self, x, y):

        yPred = copy.deepcopy(y)

        if self.probability: yPred = np.where(yPred >= self.threshold, self.upperVal, self.lowerVal)

        f1_score = metrics.f1_score(self.yTrue, yPred, average='micro')
        accuracy = metrics.accuracy_score(self.yTrue, yPred)
        recall = metrics.recall_score(self.yTrue, yPred, average = 'micro')
        precision = metrics.precision_score(self.yTrue, yPred, average = 'micro')
        confusionMatrix = metrics.confusion_matrix(self.yTrue, yPred)

        self.metricDict = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1_score,
            "confusion matrix": confusionMatrix
            }

        return self.metricDict