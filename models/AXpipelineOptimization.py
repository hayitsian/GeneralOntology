#!/usr/bin/env python
#
# Ian Hay - 2023-02-23
# https://github.com/hayitsian/General-Index-Visualization


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- imports ---- # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import collections
import pandas as pd
import numpy as np
from timeit import default_timer

import AXpreprocessing # local file
import supervisedmodel as model # local file
import util as util # local file
import optimize # local file

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # ---- preprocessing ---- # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def main():
   # take in filename as a command line argument
   _rawFilename = sys.argv[1] # file path

   #####################################################
   _dataCol = "abstract"
   _labelCol = "categories"
   _yLabel = "top category"

   numClasses = 4 # value is used later on
   numDataPoints = 15000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
   #####################################################

   # open the data file
   print("\n\nImporting data...\n\n")

   preprocessor = AXpreprocessing.preprocessor()
   df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=True, classify=False)
   dfSmaller = preprocessor.getStratifiedSubset(df, _yLabel, numClasses, numDataPoints)
 
   print("\n\nPreprocessing data...\n\n")

   # TODO preprocess texts for BOW and TF-IDF, dimensional reduction / feature selection
   _texts = dfSmaller[_dataCol].values


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- featurization ---- # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  

  # currently doing minimal preprocessing (no POS tagging, stemming, etc.)

   print("\n\nFeaturizing data...\n\n")

   # BoW
   # _textWords = [re.findall(r'\w+', _text) for _text in _texts] # this is super slow (regex :()
   print("Bag-of-words\n")
   _vectorizer = CountVectorizer()
   _bow = _vectorizer.fit_transform(_texts)

   #TF-iDF
   print("TF-iDF\n")
   _tfidftransformer = TfidfTransformer()
   _tfidf = _tfidftransformer.fit_transform(_bow)

   # BERT
   print("BERT")
   _bertmodel = SentenceTransformer('distilbert-base-nli-mean-tokens') # dilbert model
   _bertembeddings = _bertmodel.encode(_texts, show_progress_bar=True, normalize_embeddings=True)

   # doc2vec - very unimpressive thus far, and does not support GPUs
   """   
   print("doc2vec\n")
   _taggedDocs = [TaggedDocument(doc, [i]) for i, doc in enumerate(_texts)]
   _docmodel = Doc2Vec(_taggedDocs, vector_size=768, window=3, min_count=1, workers=16) # many hyperparameters to optimize
   _docembeddings = _docmodel.dv.get_normed_vectors()
   """

   # top2vec


   # LDA2vec


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- modeling ----# # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


   print("\n\nModeling data...\n\n")

   X = _bertembeddings # , "doc2vec":_docembeddings , "bag-of-words":_bow.toarray(), "tf-idf":_tfidf.toarray()}
   Y = dfSmaller[_yLabel].values

   #####################################################
   _epochs = 100000
   # numHidden1 = list(range(64, 259, 64))
   # numHidden2 = list(range(64, 259, 64))
   # numHidden3 = list(range(64, 259, 64))
   numHidden1 = numHidden2 = numHidden3 = [64, 128, 256]

   learningRate = 0.2
   #####################################################

   # supervised - FFNN, decision tree (boosted), TODO: SVM, EM, naive bayes?, logistic regression?

   optmizer = optimize.modelOptimizer()
   metricDF = optmizer.optimizeNN(X, Y, numHidden1, numHidden2, numHidden3, _epochs, learningRate, verbose=True)
   print(metricDF)
   metricDF.to_csv("optimization_metrics.csv")

   # unsupervised - k-means, HDBSCAN, LDA, top2vec, BERTopic, NVDM-GSM
   # some neural network topic models: https://github.com/zll17/Neural_Topic_Models#NVDM-GSM

   # TODO   

  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- evaluation ----# # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

   print("\n\nEvaluating model...\n\n")





main()