# Ian Hay - 2023-03-14

import util as util
import numpy as np

from sklearn.cluster import MiniBatchKMeans as km
from sklearn.decomposition import LatentDirichletAllocation as lda
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from top2vec import Top2Vec

### --- abstract class --- ###

class clusteringModel():

    def train(self, x, y=None):
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

class KMeans(clusteringModel):

    def __init__(self, nClasses, batchSize=4096, maxIter=5000):
        self.model = km(n_clusters=nClasses, batch_size=batchSize, max_iter=maxIter)

    def train(self, x, y=None, verbose=True):
        self.model.fit(x)
        pred = self.model.labels_
        util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

class LDA(clusteringModel):

    def __init__(self, nClasses, batchSize=512, maxIter=10):
        self.model = lda(n_components=nClasses, batch_size=batchSize, max_iter=maxIter)

    def train(self, x, y=None, verbose=True):
        output = self.model.fit_transform(x)
        pred = util.getTopPrediction(output)
        util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)
        """        if (verbose):
            print(f"Perplexity: {lda.perplexity(x)}")
            print(f"Log-likelihood: {lda.score(x)}")"""

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
