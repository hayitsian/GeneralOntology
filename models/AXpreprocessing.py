# Ian Hay - 2023-02-23


import ssl
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
import plotly.express as px


# TODO
#
#  - consider making this into a class and adding as a custom component to a spacy pipeline.
#  - runtime & memory optimization



# This function takes in raw arxiv data and preprocesses it.
#
# Pipeline:
#  - load in abstracts by filename (parameter)

def preprocessAbstracts(_filepath, _colNames, _labelCol, _dataCol, _delimiter=","):

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
                         low_memory=False,
                         quoting=csv.QUOTE_NONE)
    print("done")
    print(_df.head())
    print(_df.shape)

    labels = _df[_labelCol]
    print(set(labels))

    return _df