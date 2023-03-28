# Ian Hay - 2023-03-26
# https://github.com/hayitsian/General-Index-Visualization

import string
import pandas as pd
import spacy
import numpy as np
import multiprocessing as mp
from sklearn.base import BaseEstimator

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner']) # default spaCy preprocessing pipeline

import util # local file


class BaseImporter(BaseEstimator):

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._df = None

    def importData(self, _filepath, _colNames=None, _delimiter=",", verbose=False):
        """
        Imports data using Pandas.read_csv() function.
        """
        if verbose: print(f"Loading data into Pandas from {_filepath}...\n")
        self._df = pd.read_csv(_filepath, 
                            sep=_delimiter,
                            names=_colNames,
                            low_memory=False)
        if verbose: print("done")
        if verbose: print(self._df.head())
        if verbose: print(self._df.shape)

        return self._df

    def splitXY(self, _df, _dataCol, _labelCol=None):
        """
        Splits this preprocessor's data into `x` and (optionally) `y` for feeding
        into BaseEstimator's fit() and transform() functions.
        """
        if (_labelCol is not None): # if labelCol is defined
            return _df[_dataCol], _df[_labelCol] # return x, y
        else: return _df[_dataCol] # return x


    def fit(self, x, y=None):
        return self


    def transform(self, x):
        return [_text.split() for _text in x] # at the least, tokenizers on spaces


######################################################################


class BasePreprocessor(BaseEstimator):
# https://stackoverflow.com/questions/45605946/how-to-do-text-pre-processing-using-spacy

    def __init__(self, nlp=nlp, n_jobs=1, verbose=False, **kwds):
        """
        Text preprocessing transformer includes steps:
            1. Punctuation removal
            2. Stop words removal
            3. Lemmatization

        nlp  - spacy model
        n_jobs - parallel jobs to run
        """
        super().__init__(**kwds)
        self.nlp = nlp
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x, y=None):
        return self

    def transform(self, x, _stopwordRemoval=True, _lemmatize=True, _puncRemoval=True, *_):
        """
        
        """
        X_copy = x.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data


    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text, _stopwordRemoval=True, _lemmatize=True, _puncRemoval=True):
        doc = self.nlp(text)
        if _puncRemoval: doc = self._remove_punct(doc)
        if _stopwordRemoval: doc = self._remove_stop_words(doc)
        if _lemmatize: doc = self._lemmatize(doc)
        return doc

    def _remove_punct(self, doc):
        return (t for t in doc if t.text not in string.punctuation)

    def _remove_stop_words(self, doc):
        return (t for t in doc if not t.is_stop)

    def _lemmatize(self, doc):
        return ' '.join(t.lemma_ for t in doc)