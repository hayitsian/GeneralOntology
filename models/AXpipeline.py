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
import util # local file
import optimize # loca file

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
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

   numClasses = 30 # value is used later on
   numDataPoints = 200000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
   #####################################################

   # open the data file
   print("\n\nImporting data...\n\n")

   preprocessor = AXpreprocessing.preprocessor()
   df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=True, classify=False)

   dfSmaller = preprocessor.getStratifiedSubset(df, _yLabel, numClasses, numDataPoints)
   print(dfSmaller[_yLabel].head())
   print(dfSmaller[_yLabel].describe())
 
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

   X = {"BERT":_bertembeddings} # , "doc2vec":_docembeddings , "bag-of-words":_bow.toarray(), "tf-idf":_tfidf.toarray()}
   Y = dfSmaller[_yLabel].values

   #####################################################
   _epochs = 100000
   numHidden1 = 256
   numHidden2 = 256
   numHidden3 = 256
   learningRate = 0.05
   maxIter=10000
   nEstimators = 1000
   nEstimatorsAda = 50
   numOutput = numClasses
   #####################################################

   # supervised - FFNN, decision tree (boosted), TODO: SVM, EM, naive bayes?, logistic regression?

   models = {
            # "randomForest": model.RandomForestClassifier(nEstimators=nEstimators, criterion="entropy", maxDepth=5, minSamplesSplit=5),
             "logisticRegressor": model.logisticRegression(maxIter=maxIter),
            # "adaBoost": model.adaBoostDecisionTree(nEstimators=nEstimatorsAda, learningRate=learningRate),
            # "ffNN": model.FFNN(input_size=numInput, output_size=numOutput, hidden_size=numHidden, learningRate=learningRate, epochs=_epochs).to(device)
            }

   # NN loss function
   _criterion = "cel"

   for _dataLabel, x in X.items():

      numInput = x.shape[1]

      print(f"\nTraining FFNN with {_criterion} loss on {_dataLabel} with {numDataPoints} abstracts across {numClasses} topics:\n"
      + f"{_epochs} epochs, {learningRate} learning rate, {numInput} input neurons, {numHidden1} + {numHidden2} + {numHidden3} hidden neurons...\n")

      # this actually might be bad, may want unit length normalization instead
      # x = (x - x.mean(axis=0)) / x.std(axis=0) # zero mean, unit variance normalization

      # rowSums = x.sum(axis=1)
      # xNew = x / rowSums[:, np.newaxis] # unit length normalizations - redundant, this is already done during embedding

      xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size=0.2, random_state=42)

      optimizer = optimize.modelOptimizer()
      ffNN = model.FFNN(input_size=numInput, output_size=numOutput, criterion=_criterion, hidden_size_1=numHidden1, hidden_size_2=numHidden2, hidden_size_3=numHidden3, learningRate=learningRate, epochs=_epochs).to(device)

      start = default_timer()
      f1, roc, acc, recall, precision = optimizer.runNN(ffNN, xTrain, yTrain, xTest, yTest)
      _time = default_timer() - start

      print(f"\nTrained FFNN with {_criterion} loss on {_dataLabel} with {numDataPoints} abstracts across {numClasses} topics:\n"
            + f"{_epochs} epochs, {learningRate} learning rate, {numHidden1} + {numHidden2} + {numHidden3} hidden neurons.\n"
            + f"Training took {_time}"
            + f"Average Metrics: \nF1 = {f1:0.3f}  \nROC AUC = {roc:0.3f}  \nAccuracy = {acc:0.3f}  \nRecall = {recall:.3f}  \nPrecision = {precision:.3f}\n")



   """   discreteX = _bow.toarray()
   print(f"\nTraining Naive Bayes on BOW data...\n")
   NB = model.NaiveBayes()
   NB.train(discreteX, Y)
   predy = NB.test(discreteX)
   metrics = util.multi_label_metrics(predy, Y)
   print(metrics)"""


   # unsupervised - k-means, HDBSCAN, LDA, top2vec, BERTopic, NVDM-GSM
   # some neural network topic models: https://github.com/zll17/Neural_Topic_Models#NVDM-GSM

   # TODO   

  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- evaluation ----# # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

   print("\n\nEvaluating model...\n\n")





main()