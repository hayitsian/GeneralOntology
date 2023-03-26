# Ian Hay - 2023-03-14

import util as util
import numpy as np

from sklearn.cluster import MiniBatchKMeans as km
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.decomposition import NMF as nmf
from hdbscan import HDBSCAN # TODO
from bertopic import BERTopic # TODO
from bertopic.vectorizers import ClassTfidfTransformer # TODO
from top2vec import Top2Vec # TODO
from gensim.models import ldamulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim import matutils
from timeit import default_timer

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

    def train(self, x, nClasses, vocab, nJobs=14, batchSize=512, maxIter=10, y=None, verbose=False):
        self.model = lda(n_components=nClasses, batch_size=batchSize, max_iter=maxIter, n_jobs=nJobs)
        self.model.fit(x)

    def test(self, x, xEmb, y=None, verbose=False):
        output = self.model.transform(x)
        pred = util.getTopPrediction(output)
        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=xEmb, labels=y, supervised=y is not None, verbose=verbose)
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]


    def perplexity(self, x):
        return self.model.perplexity(x)
    

    def coherence(self, x, vocab, nTop=10):
        # takes in the raw documents - TODO should tokenize them with something better

        # https://stackoverflow.com/questions/60613532/how-do-i-calculate-the-coherence-score-of-an-sklearn-lda-model
        topics = self.model.components_

        n_top_words = nTop
        id2word = dict([(i, s) for i, s in enumerate(vocab)])
        # corpus = matutils.Sparse2Corpus(x.T)

        texts = [[word for word in doc.split()] for doc in x] # this is the problem child

        _dict = corpora.dictionary.Dictionary(texts)
        featnames = [_dict[i] for i in range(len(_dict))]
        corpus = [_dict.doc2bow(text) for text in texts]

        topWords = []
        for _topic in topics:
            topWords.append([featnames[i] for i in _topic.argsort()[:-n_top_words - 1:-1]])
        cm = CoherenceModel(topics=topWords, texts=texts, dictionary=_dict, topn=nTop, coherence='u_mass')
        return cm.get_coherence()

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

class gensimLDA(clusteringModel):

    def __init__(self):
        pass

    def train(self, x, nClasses, vocab, workers=14, batchSize=512, maxIter=10, y=None, verbose=False):
        # needs a sparse array

        # https://gist.github.com/aronwc/8248457
        start = default_timer()
        id2word = dict([(i, s) for i, s in enumerate(vocab)])
        _dictTime = default_timer() - start
        if verbose: print(f"id2word dict creation: {_dictTime:.3f}")

        self.trainCorpus = matutils.Sparse2Corpus(x.T)
        _corpusTime = default_timer() - start - _dictTime
        if verbose: print(f"train corpus creation creation: {_corpusTime:.3f}")
        self.model = ldamulticore.LdaMulticore(self.trainCorpus, 
                                               # id2word=id2word,
                                               num_topics=nClasses, workers=workers, chunksize=batchSize, passes=maxIter)
        
        _trainTime = default_timer() - start - _dictTime - _corpusTime
        if verbose: print(f"model creation and training time: {_trainTime:.3f}")
    
    def test(self, x, xEmb, y=None, verbose=False):
        # needs a sparse array

        corpus = matutils.Sparse2Corpus(x.T)
        output = self.model.get_document_topics(corpus)
        _out = matutils.corpus2csc(output)
        _out = _out.T.toarray()

        pred = util.getTopPrediction(_out)

        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=xEmb, labels=y, supervised=y is not None, verbose=verbose)
        """        if (verbose):
            print(f"Perplexity: {lda.perplexity(x)}")
            print(f"Log-likelihood: {lda.score(x)}")"""
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]


    def perplexity(self, x, y=None, vebose=False):
        _corpus = matutils.Sparse2Corpus(x.T)
        _perp = np.exp(-1. * self.model.log_perplexity(_corpus))
        return _perp
    
    def coherence(self, x, nTop=10, y=None, verbose=False):
        corpus = matutils.Sparse2Corpus(x.T)
        cm = CoherenceModel(model=self.model, corpus=corpus, topn=nTop, coherence='u_mass')
        return cm.get_coherence()

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