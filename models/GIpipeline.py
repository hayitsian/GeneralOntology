#!/usr/bin/env python
#
# Ian Hay - 2023-02-03

import sys

import GIpreprocessing as GIpreprocessing


# take in filename as a command line argument
_rawFilename = sys.argv[1] # takes in the actual file path
_rawDataFilename = "../data/doc_ngrams_0_1M_ngrams.txt"

# hard coded things
_columnListNGrams = ["hash", "ngram", "ngram_lc", "ngram_tokens", "ngram_count", "term_freq", "doc_count", "date_added"]
_columnListKeywords = ["hash", "keywords", "keywords_lc", "keyword_tokens", "keyword_score", "doc_count", "insert_date"]
_dataLabel = "ngram_lc"
_delimiter = "\t"

# open the data file
print("Importing data...\n\n")

with open(_rawFilename, "r") as file:
    _lines = file.readlines()
print(f"Number of lines in opened {_rawFilename}: {len(_lines)}")
print("done\n")

# preprocess
print("Preprocessing data...\n\n")

_indexCol = _columnListNGrams[0]
_texts = GIpreprocessing(_lines, _indexCol)
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

