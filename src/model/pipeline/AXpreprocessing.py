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

import sys
import os
sys.path.append(os.path.abspath("/home/ian/Documents/GitHub/General-Index-Visualization/src/model/pipeline"))


import preprocessing


class AXimporter(preprocessing.BaseImporter):
    """
    NOTE: This defaults to encoding and splitting on the base category (e.g., Physics)
    rather than the lower level category (e.g., physics.optics).

    TODO: Add more flexibility for the above points.
    
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.dataCol = "abstract"
        self.labelCol = "categories"
        self.topLabelCol = "top category"
        self.baseLabelCol = "base categories"
        self.topBaseLabelCol = "top base category"

    def importData(self, _filepath, _colNames=None, _delimiter=",", verbose=False):
        super().importData(_filepath, _colNames, _delimiter, verbose)
        self.parseLabels()
        _df = self.encodeLabels(self._df)
        return _df

    def fit(self, x):
        super().fit(x)

    def transform(self, x):
        super().transform(x)

    def splitXY(self, _df):
        return super().splitXY(_df, self.dataCol, self.topBaseLabelCol)


    def parseLabels(self, verbose=False):
        """
        NOTE: By default, this function mutates this preprocessor's _labelCol in place.
        """
        if verbose: print(f"Parsing labels in {self.labelCol}...\n")
        if verbose: print(f"Before:\n{self._df[self.labelCol].head()}")
        self._df[self.labelCol] = self._df[self.labelCol].str.split(expand=False)
        if verbose: print(f"After:\n{self._df[self.labelCol].head()}")
        self._df[self.topLabelCol] = self._df[self.labelCol].str[0] # gets the first (top) label
        if verbose: print(f"Top labels:\n{self._df[self.topLabelCol].head()}")
        self._df[self.baseLabelCol] = self._df[self.labelCol].apply(util.getBaseCategories) # TODO: this is relying on implication of type="AX"
        if verbose: print(f"Base labels:\n{self._df[self.baseLabelCol].head()}")
        self._df[self.topBaseLabelCol] = self._df[self.baseLabelCol].str[0] # gets the first (top) base label
        return self._df


    def encodeLabels(self, df, _encodedLabelColName=None, verbose=False):
        """
        NOTE: by default, this function mutates this preprocessor's _labelCol in place.
        """
        if _encodedLabelColName is None: _encodedLabelColName = self.topBaseLabelCol
        if verbose: print(f"Encoding labels in {self.topBaseLabelCol}...\n")
        le = LabelEncoder()
        toEncode = copy.deepcopy(df[self.topBaseLabelCol].values)
        df[_encodedLabelColName] = le.fit_transform(toEncode)
        if verbose: print("done")
        if verbose: print(f"Before:\n{df[self.topBaseLabelCol].head()}")
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
    

class AXpreprocessor(preprocessing.BasePreprocessor):

    def __init__(self, nlp=nlp, n_jobs=1, verbose=False):
        super().__init__(nlp=nlp, n_jobs=n_jobs, verbose=False)

    def fit(self, x, y=None):
        super().fit(x, y)

    def transform(self, x):
        _x = super().transform(x)
        _x = [_word.replace("\n", " ") for _word in _x]
        return pd.Series(_x)