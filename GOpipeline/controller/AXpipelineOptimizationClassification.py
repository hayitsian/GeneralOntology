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
from collections import Counter
import itertools

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath("/home/ian/Documents/GitHub/General-Index-Visualization/GOpipeline"))

import model.AXpreprocessing as AXpreprocessing # local file
import model.classifiermodel as classifiermodel # local file
import model.neuralnetworkmodel as neuralnetworkmodel # local file
import util # local file
import optimizer # local file
import model.clusteringmodel as clusteringmodel # local file
import model.featuremodel as featuremodel # local file


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

   numClasses = 10 # value is used later on
   numDataPoints = 15000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
   #####################################################

   # open the data file
   print("\n\nImporting data...\n\n")

   preprocessor = AXpreprocessing.preprocessor()
   df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=True, classify=False)

   categoryCounter = Counter(list(itertools.chain(*df[_labelCol].values)))
   print(f"Total number of labels: {sum(categoryCounter.values())}\n")
   print(categoryCounter)

   print("\n\nPreprocessing data...\n\n")

   _texts, Y = preprocessor.getStratifiedSubset(df, _yLabel, _dataCol, numClasses, numDataPoints, verbose=True)
   Y = Y.values

   # preprocess texts
   _tokensPrior = list(itertools.chain(*[_txt.split() for _txt in _texts]))
   _vocabSizePrior = len(set(_tokensPrior))
   print(f"\nToken number before preprocessing: {len(_tokensPrior)}")
   print(f"Vocab size before preprocessing: {_vocabSizePrior}\n")

   textPrep = AXpreprocessing.TextPreprocessor(n_jobs=-1, verbose=True)
   _preprocessedTexts = textPrep.transform(_texts).values
   _texts = _texts.values

   _tokensAfter = list(itertools.chain(*[_txt.split() for _txt in _preprocessedTexts]))
   _vocabSizeAfter = len(set(_tokensAfter))
   print(f"Token number after preprocessing: {len(_tokensAfter)}")
   print(f"Vocab size after preprocessing: {_vocabSizeAfter}\n")

   
   # TODO: plot of tokens vs vocab in ascending order, histogram of token freq, tokens per manuscript
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
   _epochs = 8000
   # numHidden1 = list(range(64, 259, 64))
   # numHidden2 = list(range(64, 259, 64))
   # numHidden3 = list(range(64, 259, 64))
   numHidden1 = numHidden2 = numHidden3 = 128

   learningRate = 0.2

   nEstimators=1000
   maxIter=1000000

   VERBOSE=True

   #####################################################


   # supervised - FFNN, random forest, naive bayes, logistic regression

   logr = classifiermodel.logisticRegression(maxIter=maxIter)
   nb = classifiermodel.NaiveBayes()
   rfc = classifiermodel.RandomForestClassifier(nEstimators=nEstimators)

   _models = {
      "Naive Bayes": nb,
      "Logistic Regression": logr,
      "Random Forest": rfc
   }


   for _xLabel, _x in X.items(): 

      for _featLabel, _featurizer in featurizers.items():
         print("\nFeaturizing data...\n")
         _xx = _featurizer.train(_x)

         masterDict = {}
         # metricList = ["f1", "roc_auc", "accuracy", "recall", "precision", "confusion matrix"]
         metricList = ["f1", "accuracy", "recall", "precision", "confusion matrix"]

         for _modelLabel, _model in _models.items():
               
            metricDict = {}
            for _metric in metricList:
               metricDict[_metric] = []

            kfold = StratifiedKFold(n_splits=5)
            for i, (train_index, test_index) in enumerate(kfold.split(_xx, Y)):
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

               _metrics = util.getClassificationMetrics(predy, ytest, probability=False, verbose=False)
               _f1 = _metrics['f1']
               if VERBOSE: print(f"F1 Score: {_f1:.3f}\n")
               for _key in metricDict.keys():
                  metricDict[_key].append(_metrics[_key])

            masterDict[_modelLabel] = metricDict


         # now do the neural network bc its different :)
         numInput = _xx.shape[1]

         metricDict = {}
         for _metric in metricList:
            metricDict[_metric] = []

         kfold = StratifiedKFold(n_splits=5)
         for i, (train_index, test_index) in enumerate(kfold.split(_xx, Y)):
            print(f"Training {_xLabel} with {_featLabel} features and Neural Network model\nOn {numDataPoints} abstracts across {numClasses} topics. Fold {i}...")
            _model = neuralnetworkmodel.FFNN(numInput, numClasses, hidden_size_1=numHidden1, hidden_size_2=numHidden2, hidden_size_3=numHidden3, epochs=_epochs)
            xtrain = _xx[train_index]
            ytrain = Y[train_index]
            xtest = _xx[test_index]
            ytest = Y[test_index]
            start = default_timer()
            _model.train(xtrain, ytrain, verbose=VERBOSE)
            predy = _model.test(xtest)
            _time = default_timer() - start
            print(f"Training took: {_time:.3f}")

            predy = predy.cpu().data.numpy()

            numOutput = len(set(ytest))
            y_true = np.zeros((ytest.size, numOutput)) # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
            y_true[np.arange(ytest.size), ytest] = 1

            _metrics = util.getClassificationMetrics(predy, y_true, probability=True, verbose=False)
            _metrics["confusion matrix"] = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
            _f1 = _metrics['f1']
            if VERBOSE: print(f"F1 Score: {_f1:.3f}")
            for _key in metricDict.keys():
               metricDict[_key].append(_metrics[_key])

         masterDict["neural network"] = metricDict


         for _modelLabel, metrics in masterDict.items():
            df = pd.DataFrame.from_dict(metrics)
            df.to_csv(f"../../Visualizations/experiment 2 - 2023-03-22/Classification Metrics for {_modelLabel} with {_featLabel} features on {_xLabel} data.csv")

      
   
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