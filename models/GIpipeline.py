#!/usr/bin/env python
#
# Ian Hay - 2023-02-03

import sys
import subprocess

import GIpreprocessing as GIpreprocessing


# take in filename as a command line argument
_rawFilename = sys.argv[1] # takes in the actual file path
_rawDataFilename = "../data/doc_ngrams_0_1M_ngrams.txt"

# hard coded things
_columnListNGrams = ["hash", "ngram", "ngram_lc", "ngram_tokens", "ngram_count", "term_freq", "doc_count", "date_added"]
_columnListKeywords = ["hash", "keywords", "keywords_lc", "keyword_tokens", "keyword_score", "doc_count", "insert_date"]
_dataLabel = "ngram_lc"
_uselessLabel = "date_added"
_indexCol = "hash"
_delimiter = "\t"

# open the data file
print("Importing & preprocessing data...\n\n")

_texts = GIpreprocessing.preprocess(_rawFilename, _columnListNGrams, _indexCol, _dataLabel, _delimiter, _uselessLabel)

print(f"Number of manuscripts in opened from \'{_rawFilename}\': {_texts.shape[0]}")
print("done\n")



# embedding
#
# TODO
#



# dimensional reduction
#
# TODO
#


# clustering
#
# TODO
#


# tokenize
#
# TODO
#


# weighting
#
# TODO
#

