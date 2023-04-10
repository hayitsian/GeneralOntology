# Ian Hay - 2023-02-03


import ssl
import pandas as pd

import spacy 

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


import preprocessing

# TODO
#
#  - object-ify to fit BasePreprocessor
#  - runtime & memory optimization

class GIimporter(preprocessing.BaseImporter):

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.indexCol = "hash"
        self.dataCol = "ngram_lc"


    def importData(self, _filepath, _colNames=None, _delimiter=",", verbose=False):
        super().importData(_filepath, _colNames, _delimiter, _indexCol=0, verbose=verbose)
        self.parseNgrams()
        return self._df
    
    def splitXY(self, _df, _dataCol, _labelCol=None):
        return super().splitXY(_df, _dataCol, _labelCol)

    def fit(self, x):
        super().fit(x)

    def transform(self, x):
        super().transform(x)


    # This function takes in raw ngram data and preprocesses it.
    #
    # Pipeline:
    #  - group ngrams by index column (parameter)
    def parseNgrams(self):
        _df[self.dataCol] = _df[self.dataCol].astype(str) # some numbers are causing errors and being treated as floats

        # group the data
        print("Grouping manuscripts...\n")
        _df = _df.groupby(self.indexCol).agg(list)
        _df["num_ngrams"] = _df[self.dataCol].apply(lambda x: len(x))

        print(_df["num_ngrams"].describe())
        print(_df[self.dataCol].shape)


        # return imported data
        return _df[self.dataCol]
    

class GIpreprocessor(preprocessing.BasePreprocessor):

    def __init__(self, nlp=nlp, n_jobs=1, verbose=False):
        super().__init__(nlp=nlp, n_jobs=n_jobs, verbose=False)

    def fit(self, x, y=None):
        super().fit(x, y)

    def transform(self, x):
        _x = super().transform(x)
        # TODO v
        _x = [_word.replace("\n", " ") for _word in _x] # is this enough? does this mess with anything?
        return pd.Series(_x)