#!/usr/bin/env python
#
# Ian Hay - 2023-02-03



import sys
import io
import ssl
import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from spacy.language import Language


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
    nlp = spacy.load("en_core_web_sm")
    _stopWords = stopwords.words("english")


    # load text data
    _df = pd.read_csv(_filepath, 
                         header=None, 
                         sep=_delimiter,
                         index_col=0,
                         names=_colNames)
    _df[_dataCol] = _df[_dataCol].astype(str) # some numbers are causing errors and being treated as floats
    _df = _df.drop(columns = [_uselessLabel])
    print(_df.dtypes)
    print(_df.head())
    print(_df.shape)


    # preprocess the data
    # TODO
    #
    #  - DONE split by \t
    #  - DONE group by manuscript (hash)
    #  - DONE de-nest list
    #  - DONE remove stopwords
    #  - POS tagging


    _df = _df.groupby(_indexCol).agg(list)
    _df[_dataCol] = _df[_dataCol].apply(lambda x: ". ".join([word for word in x if word not in (list(_stopWords))]))
    print(_df[_dataCol])
    print(_df.shape)


    textPOS = []
    POS=["PROPN", "NOUN", "ADJ", "ADV", "VERB", "X"]
    textPipe = nlp.pipe(_df[_dataCol], batch_size=10, n_process=4)

#     print(list(textPipe))


    # return preprocessed data

    # TODO
    #
    #


    return _df