#!/usr/bin/env python
#
# Ian Hay - 2023-02-03

import sys
import spacy
from bertopic import BERTopic
from transformers import BertTokenizerFast
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import plotly.express as px

import GIpreprocessing
import GIbertopic

#
# Make sure to run this in your environment!: python3 -m spacy download en_core_web_trf
#


# take in filename as a command line argument
_rawFilename = sys.argv[1] # takes in the actual file path
_rawDataFilename = "../data/doc_ngrams_0_100M_ngrams.txt"
_modelFilename = "BERTopic_doc_ngrams_0_100M_ngrams_model"

# hard coded things
_columnListNGrams = ["hash", "ngram", "ngram_lc", "ngram_tokens", "ngram_count", "term_freq", "doc_count", "date_added"]
_columnListKeywords = ["hash", "keywords", "keywords_lc", "keyword_tokens", "keyword_score", "doc_count", "insert_date"]
_dataLabel = "ngram_lc"
_uselessLabel = "date_added"
_indexCol = "hash"
_delimiter = "\t"

# open the data file
print("\nImporting data...\n\n")

_texts = GIpreprocessing.preprocess(_rawFilename, _columnListNGrams, _indexCol, _dataLabel, _delimiter, _uselessLabel)

print(f"Number of manuscripts opened from \'{_rawFilename}\': {_texts.shape[0]}")

print("done\n")


# preprocess
print("\nPreprocessing data...\n\n")

# load into spaCy generator object
""" nlp = spacy.load("en_core_web_trf", exclude=["tok2vec", "ner", "attribute_ruler"])
nlp.max_length = 3000000
nlp.truncation

print(nlp.pipeline)
print("\n")

textPipe = nlp.pipe(_texts, batch_size=2, n_process=12)"""

# get the preprocessed text

""" GItexts = []

print("looping\n")

for doc in list(textPipe):
  _text = [
    token.text
    for token in doc
      if not token.is_punct
      and not token.is_stop
      and not token.like_num
      and token.is_alpha
        ]
  GItext.append(" ".join(_text).lower()) # https://stackoverflow.com/questions/65850018/processing-text-with-spacy-nlp-pipe
print(len(GItext))
print(len(_texts))

processedNgramLengths = []
for doc in GItexts:
  __text = doc.split(". ")
  processedNgramLengths.append(len(__text))

fig = px.histogram(processedNgramLengths)
fig.write_html(_filepath + "_processed_ngrams_per_manuscript.html") """



print("done\n")


# training the model
print("\nBuilding model...\n\n")


# embedding

# need to figure out how to get GPU compute to work
""" spacy.require_gpu()

embedding_model = spacy.load("en_core_web_trf", exclude=["ner", "attribute_ruler", "lemmatizer", "tagger", "parser"])
embedding_model.max_length = 3000000
print(embedding_model.pipeline) """

embedding_model = BertTokenizerFast.from_pretrained("bert-base-uncased")
embedding_model.max_length = 3000
embedding_model.truncation = True



# BERTopic pipeline
topic_model = GIbertopic.train(_texts, _embeddingModel = None)

topic_model.save(_modelFilename)
print(f"BERTopic model saved as: {_modelFilename}\n")
print("done")


# fit the General Index (processed)
""" topics, probabilities = topic_model.fit_transform(GItext)

topic_model.save(_modelFilename + "_processed")
print(f"BERTopic model saved as: {_modelFilename}_processed\n")
print("done") """