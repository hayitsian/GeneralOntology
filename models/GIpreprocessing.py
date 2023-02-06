#!/usr/bin/env python
#
# Ian Hay - 2023-02-03


import ssl
import pandas as pd
import nltk
from nltk.corpus import stopwords


# TODO
#
#  - consider making this into a class and adding as a custom component to a spacy pipeline.
#  - runtime & memory optimization



# This function takes in raw ngram data and preprocesses it.
#
# Pipeline:
#  - load in ngrams by filename (parameter)
#  - group ngrams by index column (parameter)
#  - removes stopwords from NLTK's English stopwords (library)
#  - POS tagging by spacy's "en_core_web_sm" (library)
#  - incorporates it into a spacy pipeline
#  - returns the spacy generator object
def preprocess(_filepath, _colNames, _indexCol, _dataCol, _delimiter, _uselessLabel):

    # load some libraries
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')
    _stopWords = stopwords.words("english")


    # load text data
    print("Loading into Pandas...\n")
    _df = pd.read_csv(_filepath, 
                         header=None, 
                         sep=_delimiter,
                         index_col=0,
                         names=_colNames,
                         low_memory=False)
    print("done")
    _df[_dataCol] = _df[_dataCol].astype(str) # some numbers are causing errors and being treated as floats
    _df = _df.drop(columns = [_uselessLabel]) # data_added column is useless and holds mostly \N characters
    print(_df.head())
    print(_df.shape)


    # preprocess the data
    print("Grouping manuscripts...\n")
    _df = _df.groupby(_indexCol).agg(list)
    print("Removing stopwords...\n")
    _df[_dataCol] = _df[_dataCol].apply(lambda x: ". ".join([word for word in x if word not in (list(_stopWords))]))
    print(_df[_dataCol].head())
    print(_df[_dataCol].shape)


    # return preprocessed data
    return _df[_dataCol]