#!/usr/bin/env python
#
# Ian Hay - 2023-03-20
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
import matplotlib.pyplot as plt

import pipeline.model.AXpreprocessing as AXpreprocessing # local file
import classifiermodel # local file
import neuralnetworkmodel as neuralnetworkmodel
import util # local file
import pipeline.controller.optimizer as optimizer # local file
import clusteringmodel # local file
import featuremodel # local file


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from top2vec import Top2Vec
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
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

   numClasses = 5 # value is used later on
   numDataPoints = 5000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
   #####################################################

   # open the data file
   print("\n\nImporting data...\n\n")

   preprocessor = AXpreprocessing.preprocessor()
   df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=True, classify=False)

   _texts, Y = preprocessor.getStratifiedSubset(df, _yLabel, _dataCol, numClasses, numDataPoints)

   print("\n\nPreprocessing data...\n\n")

   # preprocess texts
   textPrep = AXpreprocessing.TextPreprocessor(n_jobs=-1)
   _preprocessedTexts = textPrep.transform(_texts).values
   _texts = _texts.values

   # TODO: print token num, vocab length, plot of tokens vs vocab in ascending order, histogram of token freq, tokens per manuscript
   # all ^ comparing before & after preprocessing

   X = {"raw texts": _texts, "preprocessed texts": _preprocessedTexts}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- featurization ---- # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


   featurizers = {}

   # BoW
   _name = "Bag-of-words"
   _vectorizer = featuremodel.bowModel()
   featurizers[_name] = _vectorizer

   #TF-iDF
   _name = "TF-iDF"
   _tfidftransformer = featuremodel.tfidfModel()
   featurizers[_name] = _tfidftransformer

   # BERT
   _name = "BERT"
   _bertmodel = featuremodel.bertModel()
   featurizers[_name] = _bertmodel


   # TODO: custom embeddings, ngrams
   # TODO: add feature extraction with preprocessed text
   # TODO: plot features, somehow...


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # ---- feature selection ---- # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


   # TODO: PLSA, PCA, t-SNE, UMAP, etc. or some lasso regularization
   # TODO: visualize features


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # #  ---- topic modeling ---- # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

   ####################### --- classification --- ##############################


   #####################################################
   _epochs = 100000
   # numHidden1 = list(range(64, 259, 64))
   # numHidden2 = list(range(64, 259, 64))
   # numHidden3 = list(range(64, 259, 64))
   numHidden1 = numHidden2 = numHidden3 = 128

   learningRate = 0.2

   VERBOSE=True
   #####################################################


   # supervised - FFNN, random forest, naive bayes, logistic regression

   logr = classifiermodel.logisticRegression()
   nb = classifiermodel.NaiveBayes()
   rfc = classifiermodel.RandomForestClassifier()

   _models = {
      "Logistic Regression": logr,
      "Naive Bayes": nb,
      "Random Forest": rfc
   }


   for _xLabel, _x in X.items(): 

      for _featLabel, _featurizer in featurizers.items():
         print("\nFeaturizing data...\n")
         _xx = _featurizer.train(_x)

         masterDict = {}
         metricList = ["f1", "roc_auc", "accuracy", "recall", "precision", "confusion matrix"]


         for _modelLabel, _model in _models.items():
               
            metricDict = {}
            for _metric in metricList:
               metricDict[_metric] = []

            kfold = StratifiedKFold(n_splits=5)
            for i, (train_index, test_index) in enumerate(kfold.split(_xx)):
               print(f"Training {_xLabel} with {_featLabel} features and {_modelLabel} model\nOn {numDataPoints} abstracts across {numClasses} topics. Fold {i}...")
               xtrain = _xx[train_index]
               ytrain = Y[train_index]
               xtest = _xx[test_index]
               ytest = Y[test_index]
               start = default_timer()
               _model.train(xtrain, ytrain)
               predy = _model.test(xtest)
               _time = default_timer() - start
               print(f"Training took: {_time:.3f}")

               _metrics = util.getClassificationMetrics(predy, ytest, verbose=VERBOSE)
               if VERBOSE: print(f"F1 Score: {_metrics["f1"]:.3f}")
               for _key in metricDict.keys():
                  metricDict[_key].append(_metrics[_key])

            masterDict[_modelLabel] = metricDict

         for _modelLabel, metrics in masterDict.items():
            df = pd.DataFrame.from_dict(metrics)
            df.to_csv(f"Classification Metrics for {_modelLabel} with {_featLabel} features on {_xLabel} data.csv")
   
   """
   optmizer = optimize.modelOptimizer()
   metricDF = optmizer.optimizeNN(X, Y, numHidden1, numHidden2, numHidden3, _epochs, learningRate, verbose=True)
   print(metricDF)
   metricDF.to_csv("optimization_metrics.csv")
   """


  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- evaluation ----# # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

   print("\n\nEvaluating model...\n\n")



   # TODO : plot classification metrics 



main()