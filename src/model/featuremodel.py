# Ian Hay - 2023-03-19

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gensim
from gensim.corpora import Dictionary
import numpy as np

import util # local file
from model.basemodel import BaseModel # local file
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseFeaturizer(BaseModel):

    def __init__(self):
        super().__init__()

    def fit(self, x):
        return self
    
    def transform(self, x):
        util.raiseNotDefined()

    def getVocab(self):
        util.raiseNotDefined()

    def save(self):
        util.raiseNotDefined()

    def __repr__(self):
        util.raiseNotDefined()


###############################################################################


class bowModel(BaseFeaturizer):
    
    def __init__(self, ngram_range=(1,1), minDF=1, maxDF=1.0):
        super().__init__()
        self.model = CountVectorizer(ngram_range=ngram_range, min_df=minDF, max_df=maxDF)

    def fit(self, x, y=None):
        self.model.fit(x)
        return self

    def transform(self, x, y=None):
        return self.model.transform(x)
    
    def getVocab(self):
        return self.model.get_feature_names_out()

###############################################################################


class gensimBowModel(BaseFeaturizer):

    def __init__(self, ngram_range=(1,1), minDF=1, threshold=10.):
        super().__init__()
        self.ngram_range = ngram_range
        self.model = None
        self.minDF = minDF
        self.threshold = threshold

    def fit(self, x, y=None):
        docs = [doc.split() for doc in x]
        docs = self._makeNgrams(docs)
        self.model = Dictionary(docs)
        return self

    def transform(self, x, y=None):
        docs = [doc.split() for doc in x]
        docs = self._makeNgrams(docs)
        _x = self.model.doc2bow(docs)
        return _x
    

    def _ngramCreator(self, doc):
        # https://stackoverflow.com/questions/43918566/trying-to-mimick-scikit-ngram-with-gensim
        if doc is int: return doc
        ngram = gensim.models.Phrases(doc, min_count=self.minDF, threshold=self.threshold)
        ngramMod = gensim.models.phrases.Phraser(ngram)
        return ngramMod

    def _makeNgrams(self, docs):
        ngramModList = []
        i = 1
        while i < self.ngram_range[1]:
            ngrams = [self._ngramCreator(doc) for doc in docs]
            docs = ngrams
            i += 1
        return docs

    def getVocab(self):
        return np.array(list(self.model.token2id.keys()))


###############################################################################


class tfidfModel(BaseFeaturizer):
    
    def __init__(self, ngram_range=(1,1), minDF=1, maxDF=1.0):
        super().__init__()
        self.model = TfidfVectorizer(ngram_range=ngram_range, min_df=minDF, max_df=maxDF)

    def fit(self, x, y=None):
        self.model.fit(x)
        return self

    def transform(self, x, y=None):
        return self.model.transform(x)
    
    def getVocab(self):
        return self.model.get_feature_names_out()


###############################################################################


class bertModel(BaseFeaturizer):

    # TODO: incorporate more embedding types
    #  - abstract out "bert" and make this embeddingFeaturizer
    #  - add getVocab() method implementation

    def __init__(self, normalizeEmbedding=True):
        super().__init__()
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')  #, device=device)
        self.normalizeEmbeddings = normalizeEmbedding

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, verbose=True):
        return self.model.encode(x, show_progress_bar=verbose, normalize_embeddings=self.normalizeEmbeddings)

    def getVocab(self):
        # TODO
        pass