# Ian Hay - 2023-03-19

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class featurizerModel():

    def train(self, x):
        pass


###############################################################################


class bowModel():
    
    def __init__(self, ngram_range=(1,1)):
        self.model = CountVectorizer(ngram_range=ngram_range)


    def train(self, x):
        return self.model.fit_transform(x)
    
    def getVocab(self):
        return self.model.get_feature_names()
    

###############################################################################


class tfidfModel():
    
    def __init__(self):
        self.model = TfidfVectorizer()


    def train(self, x):
        return self.model.fit_transform(x).toarray()
    

###############################################################################


class bertModel():

    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')  #, device=device)

    def train(self, x, verbose=True):
        return self.model.encode(x, show_progress_bar=verbose, normalize_embeddings=True)