# Ian Hay - 2023-03-14

import util as util
import numpy as np

from sklearn.cluster import MiniBatchKMeans as km
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.decomposition import NMF as nmf
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

    def __init__(self):
        pass

    def train(self, x, nClasses, batchSize=4096, maxIter=5000, y=None, verbose=False):
        self.model = km(n_clusters=nClasses, batch_size=batchSize, max_iter=maxIter)
        self.model.fit(x)
        pred = self.model.labels_
        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

class LDA(clusteringModel):

    def __init__(self):
        pass

    def train(self, x, nClasses, batchSize=512, maxIter=10, y=None, verbose=False):
        self.model = lda(n_components=nClasses, batch_size=batchSize, max_iter=maxIter)
        output = self.model.fit_transform(x)
        pred = util.getTopPrediction(output)
        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)
        """        if (verbose):
            print(f"Perplexity: {lda.perplexity(x)}")
            print(f"Log-likelihood: {lda.score(x)}")"""
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

class NMF(clusteringModel):

    def __init__(self):
        pass

    def train(self, x, nClasses, maxIter=1000, y=None, verbose=False):
        self.model = nmf(n_components=nClasses, max_iter=maxIter, solver="mu", init="nndsvd", beta_loss="kullback-leibler", alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1)
        output = self.model.fit_transform(x)
        pred = util.getTopPrediction(output)
        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]