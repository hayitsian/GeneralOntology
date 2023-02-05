#!/usr/bin/env python
#
# Ian Hay - 2023-02-03

import sys
import spacy
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


import GIpreprocessing as GIpreprocessing


#
# Make sure to run this in your environment!: python3 -m spacy download en_core_web_trf
#


# take in filename as a command line argument
_rawFilename = sys.argv[1] # takes in the actual file path
_rawDataFilename = "../data/doc_ngrams_0_10M_ngrams.txt"
_modelFilename = "BERTopic_doc_ngrams_0_10M_ngrams_model"

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
nlp = spacy.load("en_core_web_sm", exclude=["tok2vec", "ner", "attribute_ruler"])
nlp.max_length = 3000000

print(nlp.pipeline)
print("\n")

textPipe = nlp.pipe(_texts, batch_size=2, n_process=12)

# get the preprocessed text

GIdocs = list(textPipe)
GItext = []

print("looping\n")

for doc in GIdocs:
    GItext.append(doc.text) # is this getting the preprocessed text from spaCy pipeline? or the raw text?

print(GItext[0:5])
print(_texts[0:5])
print("done\n")



# embedding
# spacy.require_gpu()
embedding_model = spacy.load("en_core_web_trf", exclude=["ner", "attribute_ruler", "lemmatizer", "tagger", "parser"])
embedding_model.max_length = 3000000

# dimensional reduction
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')


# clustering
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


# tokenize
vectorizer_model = CountVectorizer(stop_words="english")


# topic weighting
ctfidf_model = ClassTfidfTransformer()


# BERTopic pipeline
topic_model = BERTopic(
  embedding_model=embedding_model,    # Step 1 - Extract embeddings
  umap_model=umap_model,              # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
  diversity=0.5,                      # Step 6 - Diversify topic words
  low_memory=False,
  verbose=True 
)


# fit the General Index
print("\nBuilding model...\n\n")
topics, probabilities = topic_model.fit_transform(_texts)

topic_model.save(_modelFilename)
print(f"BERTopic model saved as: {_modelFilename}\n")
print("done")