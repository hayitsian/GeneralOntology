# Ian Hay - 2023-02-25

import util
import model.abstractmodel as abstractmodel

from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import numpy as np



### --- abstract class --- ###

class classifierModel(abstractmodel.abstractModel):

    def train(self, x, y):
        """
        Takes in and trains on the data `x` to return desired features `y`.
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def test(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : ndarray[float] : list of datapoints n samples by d features
        Returns:
            - ypred : ndarray[int] : topic prediction of n samples bc y classes
        """
        util.raiseNotDefined()


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class logisticRegression(classifierModel):

    def __init__(self, maxIter=100, C=0.125, multiClass="multinomial", penalty="l2"):

        params = {
            "penalty": penalty,
            "multi_class": multiClass,
            "max_iter": maxIter,
            "n_jobs": -1,
            "C": C
        }
        self.model = LogisticRegression(**params)


    def train(self, x, y):
        self.model.fit(x, y)


    def test(self, x):
        return self.model.predict(x)
    


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class NaiveBayes(classifierModel):

    def __init__(self):
        self.model = GaussianNB()

    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class RandomForestClassifier(classifierModel):

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


    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)
