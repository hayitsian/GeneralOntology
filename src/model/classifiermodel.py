# Ian Hay - 2023-02-25

import util
from model.basemodel import BaseModel

from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import numpy as np



### --- base class --- ###

class ClassifierModel(BaseModel):

    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        """
        Takes in and trains on the data `x` to return labels `y`.
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def predict(self, x):
        """
        Classifies documents to their topic label.
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


class LogisticRegression(ClassifierModel):

    def __init__(self, maxIter=100, C=0.125, multiClass="multinomial", penalty="l2"):
        super().__init__()

        params = {
            "penalty": penalty,
            "multi_class": multiClass,
            "max_iter": maxIter,
            "n_jobs": -1,
            "C": C
        }
        self.model = LogisticRegression(**params)


    def fit(self, x, y):
        self.model.fit(x, y)


    def predict(self, x):
        return self.model.predict(x)
    

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class NaiveBayes(ClassifierModel):

    def __init__(self):
        self.model = GaussianNB()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class RandomForestClassifier(ClassifierModel):

    def __init__(self, nEstimators=10, criterion="entropy", maxDepth=5, minSamplesSplit=5, verbose=0):

        params = {
            "n_estimators": nEstimators,
            "criterion": criterion,
            "max_depth": maxDepth,
            "min_samples_split": minSamplesSplit,
            "verbose": verbose,
            "n_jobs": -1,
        }
        self.model = ensemble.RandomForestClassifier(**params)


    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
