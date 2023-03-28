# Ian Hay - 2023-02-23


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import multiprocessing as mp
import util # local file
import string
import spacy 
import copy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

from model.preprocessing import BaseImporter, BasePreprocessor


class AXimporter(BaseImporter):

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def importData(self, _filepath, _colNames=None, _delimiter=",", verbose=False):
        super().importData(_filepath, _colNames, _delimiter, verbose)

    def fit(self, x):
        super().fit(x)

    def transform(self, x):
        super().transform(x)

    def splitXY(self, _df, _dataCol, _labelCol=None):
        return super().splitXY(_df, _dataCol, _labelCol)

    def parseLabels(self, _labelCol, _topLabelColName, _baseLabelColName, _baseTopLabelColName, verbose=False):
        """
        NOTE: By default, this function mutates this preprocessor's _labelCol in place.
        """
        if verbose: print(f"Parsing labels in {_labelCol}...\n")
        if verbose: print(f"Before:\n{self._df[_labelCol].head()}")
        self._df[_labelCol] = self._df[_labelCol].str.split(expand=False)
        if verbose: print(f"After:\n{self._df[_labelCol].head()}")
        self._df[_topLabelColName] = self._df[_labelCol].str[0] # gets the first (top) label
        if verbose: print(f"Top labels:\n{self._df[_topLabelColName].head()}")
        self._df[_baseLabelColName] = self._df[_labelCol].apply(util.getBaseCategories) # TODO: this is relying on implication of type="AX"
        if verbose: print(f"Base labels:\n{self._df[_baseLabelColName].head()}")
        self._df[_baseTopLabelColName] = self._df[_baseLabelColName].str[0] # gets the first (top) base label
        return self._df

    def encodeLabels(self, df, _labelCol, _encodedLabelColName=None, verbose=False):
        """
        NOTE: by default, this function mutates this preprocessor's _labelCol in place.
        
        """
        if _encodedLabelColName is None: _encodedLabelColName = _labelCol
        if verbose: print(f"Encoding labels in {_labelCol}...\n")
        le = LabelEncoder()
        toEncode = copy.deepcopy(df[_labelCol].values)
        df[_encodedLabelColName] = le.fit_transform(toEncode)
        if verbose: print("done")
        if verbose: print(f"Before:\n{df[_labelCol].head()}")
        if verbose: print(f"After:\n{df[_encodedLabelColName].head()}")
        
        return df
        
    def getSubsetFromNClasses(self, df, _labelCol, _numClasses, _numSamples, _replaceLabels=True, randomState=1, verbose=False):
        """
        NOTE: Does not change the internal DataFrame.
        """

        categoryCount = Counter(df[_labelCol].values)
        topNCategories = categoryCount.most_common(_numClasses)

        if verbose: print(f"Top {_numClasses} category counts: {topNCategories}")

        dfSmall = pd.DataFrame()
        for key, value in topNCategories:
            __df = df[df[_labelCol] == key]
            dfSmall = pd.concat((dfSmall, __df))
        
        dfSmaller = dfSmall.sample(n=_numSamples, random_state=randomState) # random state

        if _replaceLabels: dfSmaller = self.encodeLabels(dfSmaller, _labelCol, verbose=verbose)

        return dfSmaller
    

class AXpreprocessor(BasePreprocessor):

    def __init__(self, nlp=nlp, n_jobs=1, verbose=False):
        super().__init__(nlp=nlp, n_jobs=1, verbose=False)

    def fit(self, x, y=None):
        super().fit(x, y)

    def transform(self, x):
        return super().transform(x)