# Ian Hay - 2023-02-23


import collections
import pandas as pd

import numpy as np
import multiprocessing as mp

import string
import spacy 
from sklearn.base import TransformerMixin, BaseEstimator

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])



def getBaseCategory(str):
    if ("." in str):
        base = str.split(".")[0]
        if (base=="astro-ph" or base=="nlin" or base=="cond-mat"): return "physics"
        else: return base
    elif str=="q-alg" or str=="alg-geom" or str=="dg-ga" or str=="funct-an": return "math"
    elif str=="q-bio": return "q-bio"
    elif str=="cmp-lg": return "cs"
    else: return "physics"


def getBaseCategories(listStr):
    baseList = [getBaseCategory(s) for s in listStr]
    baseList = list({s:0 for s in baseList}) # ordered set: https://stackoverflow.com/questions/51145707/using-ordered-dictionary-as-ordered-set
    # return list(set(baseList)) # this is stochastic and does not preserve order of elements
    return baseList


# TODO
#
#  - runtime & memory optimization
class preprocessor():

    def __init__(self):
        pass


    def classifyCategories(self, _df, _yLabel, verbose=False):
        # classify the set of categories
        _categories = _df[_yLabel].values
        categoryKeys = set(_categories) # TODO does not preserve order of elements
        categoryDict = {}
        n = 0 # number of categories in total dataset
        for cat in categoryKeys:
            categoryDict[cat] = n
            n += 1

        _df = _df.replace({_yLabel: categoryDict}) # https://stackoverflow.com/questions/68487397/replacing-values-of-a-column-which-is-of-type-list
        if verbose: print(_df.head())
        if verbose: print(_df.shape)

        return _df


    def importData(self, _filepath, _labelCol, _yLabel, _colNames=None, _delimiter=",", classify=True, verbose=False):
        """
        This function takes in a filepath for arXiv abstracts, imports, and preprocesses it.
        
        Parameters:
            - _filepath : str = the filepath to load data from
            - _labelCol : str = the column of the data label
            - _colNames : list[str] | None = the columns to label the data with
            - _delimiter : str (default: ",") = the character to separate data with

        Returns:
            - _df : pd.DataFrame = the data imported and preprocessed _yLabel label column
        """

        # load text data
        if verbose: print("Loading data into Pandas...\n")
        _df = pd.read_csv(_filepath, 
                            sep=_delimiter,
                            names=_colNames,
                            low_memory=False)
        if verbose: print("done")
        if verbose: print(_df.head())
        if verbose: print(_df.shape)

        _df[_labelCol] = _df[_labelCol].str.split(expand=False)
        _df[_yLabel] = _df[_labelCol].str[0] # gets the first (top) category for each abstract

        if classify: _df = self.classifyCategories(_df, _yLabel, verbose=verbose)
        return _df


    def getStratifiedSubset(self, _df, _yLabel, _dataCol, _numClasses, numSamples, replaceLabels=True, randomState=1, verbose=False):
        """

        Returns:
            - x : ndarray[n,d] = the data of size (num samples, num features)
            - y : ndarray[n,1] = the labels of size (num samples, 1)
        """
        
        categoryCount = collections.Counter(_df[_yLabel])
        topNCategories = categoryCount.most_common(_numClasses)

        if verbose: print(f"Top {_numClasses} category counts: {topNCategories}")

        dfSmall = pd.DataFrame()
        for key, value in topNCategories:
            __df = _df[_df[_yLabel] == key]
            dfSmall = pd.concat((dfSmall, __df))
        
        dfSmaller = dfSmall.sample(n=numSamples, random_state=randomState) # random state

        if replaceLabels: dfSmaller = self.classifyCategories(dfSmaller, _yLabel, verbose=True)

        return dfSmaller[_dataCol], dfSmaller[_yLabel]
    

    def preprocessTexts(self, _texts, _stopwordRemoval=True, _lowercase=True, _posTagging=True, _pos=[""]):

        stopwords = list(nlp.Defaults.stop_words)

        if (_lowercase): _texts = [text.lower() for text in _texts]

        textSpacy = nlp(_texts)
        lemmas = [[token.lemma_ for token in document if not token.is_stop] for document in textSpacy]

        return _texts, 



# https://stackoverflow.com/questions/45605946/how-to-do-text-pre-processing-using-spacy
class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 nlp = nlp,
                 n_jobs=1,
                 verbose=False):
        """
        Text preprocessing transformer includes steps:
            1. Punctuation removal
            2. Stop words removal
            3. Lemmatization

        nlp  - spacy model
        n_jobs - parallel jobs to run
        """
        self.nlp = nlp
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

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

    def _preprocess_text(self, text):
        doc = self.nlp(text)
        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return self._lemmatize(removed_stop_words)

    def _remove_punct(self, doc):
        return (t for t in doc if t.text not in string.punctuation)

    def _remove_stop_words(self, doc):
        return (t for t in doc if not t.is_stop)

    def _lemmatize(self, doc):
        return ' '.join(t.lemma_ for t in doc)