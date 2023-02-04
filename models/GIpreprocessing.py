#!/usr/bin/env python
#
# Ian Hay - 2023-02-03



# This script takes in raw ngram data and preprocesses it.
#
# Pipeline:
#  - load in ngrams by filename (parameter)
#  - group ngrams by index column (parameter)
#  - removes stopwords from NLTK's English stopwords (library)
#  - POS tagging by spacy's "en_core_web_sm" (library)
#  - incorporates it into a spacy pipeline
#  - returns the spacy generator object



import sys
import ssl
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from spacy.language import Language



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

texts = sys.argv[1]
texts = np.array(texts)
# indexCol = sys.argv[2]


# preprocess the data

# TODO
#
#  - split by \t
#  - group by manuscript (hash)
#  - de-nest list

for line in texts:
    splitTexts = line.split("\t")

print(splitTexts)

textTagged = []

        
textTagged = splitTexts.apply(lambda x: " ".join([word for word in x.split() if word not in (list(_stopWords))])).to_list()
textPOS = []
POS=["PROPN", "NOUN", "ADJ", "ADV", "VERB", "X"]

textPipe = nlp.pipe(textTagged, batch_size=10, n_process=4)




# return preprocessed data

# TODO
#
#
