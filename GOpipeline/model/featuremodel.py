# Ian Hay - 2023-03-19

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    def fit(self, x):
        super().fit(x)

    def transform(self, x):
        return self.model.fit_transform(x)
    
    def getVocab(self):
        return self.model.get_feature_names()

###############################################################################

# TODO
class gensimBowModel(BaseFeaturizer):

    def __init__(self, ngram_range=(1,1), minDF=1, maxDF=1.0):
        super().__init__()

    def fit(self, x):
        super().fit(x)

    def transform(self, x):
        # TODO
        pass
    
    def getVocab(self):
        # TODO
        pass

###############################################################################


class tfidfModel(BaseFeaturizer):
    
    def __init__(self, ngram_range=(1,1), minDF=1, maxDF=1.0):
        super().__init__()
        self.model = TfidfVectorizer(ngram_range=ngram_range, min_df=minDF, max_df=maxDF)

    def fit(self, x):
        super().fit(x)

    def transform(self, x):
        return self.model.fit_transform(x)
    
    def getVocab(self):
        return self.model.get_feature_names()


###############################################################################


class bertModel(BaseFeaturizer):

    # TODO: incorporate more embedding types
    #  - abstract out "bert" and make this embeddingFeaturizer
    #  - add getVocab() method implementation

    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')  #, device=device)

    def fit(self, x):
        super().fit(x)

    def transform(self, x, normalizeEmbedding=True, verbose=True):
        return self.model.encode(x, show_progress_bar=verbose, normalize_embeddings=normalizeEmbedding)

    def getVocab(self):
        # TODO
        pass