#!/usr/bin/env python
#
# Ian Hay - 2023-02-23
# https://github.com/hayitsian/General-Index-Visualization

# external dependencies

import sys
import os
from collections import Counter
import itertools
import numpy as np
import pandas as pd
from timeit import default_timer
import gensim
from gensim import corpora
import copy
import matplotlib.pyplot as plt

# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import StratifiedKFold

# internal dependencies

_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath("/home/ian/Documents/GitHub/General-Index-Visualization/GOpipeline"))

import util # local file
import model.AXpreprocessing as AXpreprocessing # local file
import model.featuremodel as featuremodel # local file
import model.clusteringmodel as clusteringmodel # local file
import model.classifiermodel as classifiermodel # local file
import model.neuralnetworkmodel as neuralnetworkmodel # local file


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # ---- import & preprocess ---- # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#####################################################
_dataCol = "abstract"
_labelCol = "categories"
_yLabel = "top category"
_baseLabelCol = "base categories"
_baseLabel = "base category"

numClasses = 3 # value is used later on
numDataPoints = 1000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
VERBOSE=True
#####################################################

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# open the data file
print("\n\nImporting data...\n\n")


            ##### I/O ######
# take in filename as a command line argument
_rawFilename = sys.argv[1] # file path
            ##### I/O ######


preprocessor = AXpreprocessing.preprocessor()
df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=VERBOSE, classify=False)

categoryCounter = Counter(list(itertools.chain(*df[_labelCol].values)))
print(f"\nTotal number of categories: {len(list(set(categoryCounter.keys())))}\n")
print(categoryCounter)

topCategoryCounter = Counter(df[_yLabel].values)
print(f"\nTotal number of top categories: {len(list(set(topCategoryCounter.keys())))}\n")
print(topCategoryCounter)


differentCats = {k:v for k,v in categoryCounter.items() if k not in topCategoryCounter}
print(f"\nDifference of above categories:\n{differentCats}\n")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- featurization ---- # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


df[_baseLabelCol] = df[_labelCol].apply(AXpreprocessing.getBaseCategories)
df[_baseLabel] = df[_baseLabelCol].str[0]
baseCategoryCounter = Counter(df[_baseLabel].values)
print(f"\nTotal number of base categories: {len(list(set(baseCategoryCounter.keys())))}\n")
print(baseCategoryCounter)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print("\n\nPreprocessing data...\n\n")
_texts, Ylower = preprocessor.getStratifiedSubset(df, _yLabel, _dataCol, numClasses, numDataPoints, verbose=VERBOSE)
_texts, Yhigher = preprocessor.getStratifiedSubset(df, _baseLabel, _dataCol, numClasses, numDataPoints, verbose=VERBOSE)

Ylower = Ylower.values
Yhigher = Yhigher.values

stratBaseCounter = Counter(Yhigher)
print(stratBaseCounter)

# preprocess texts 
_tokensPrior = list(itertools.chain(*[_txt.split() for _txt in _texts]))
_vocabSizePrior = len(set(_tokensPrior))
print(f"\nToken number before preprocessing: {len(_tokensPrior)}")
print(f"Vocab size before preprocessing: {_vocabSizePrior}\n")

textPrep = AXpreprocessing.TextPreprocessor(n_jobs=-1, verbose=VERBOSE)
_preprocessedTexts = textPrep.transform(_texts).values
_texts = _texts.values

_tokensAfter = list(itertools.chain(*[_txt.split() for _txt in _preprocessedTexts]))
_vocabSizeAfter = len(set(_tokensAfter))
print(f"Token number after preprocessing: {len(_tokensAfter)}")
print(f"Vocab size after preprocessing: {_vocabSizeAfter}\n")

#####################################################

X = {"raw texts": _texts, "preprocessed texts": _preprocessedTexts}

#####################################################


featurizers = {}

# BoW
_name = "Bag-of-words"
_vectorizer = featuremodel.bowModel()
featurizers[_name] = _vectorizer

"""#TF-iDF
_name = "TF-iDF"
_tfidftransformer = featuremodel.tfidfModel()
featurizers[_name] = _tfidftransformer

# BERT
_name = "BERT"
_bertmodel = featuremodel.bertModel()
featurizers[_name] = _bertmodel"""

# n-gram model (1-5 gram)
_name = "n-grams"
_ngramVectorizer = featuremodel.bowModel(ngram_range=(1,5)) # want to test different lengths
featurizers[_name] = _ngramVectorizer


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # #  ---- topic modeling ---- # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

####################### --- clustering + metadata extraction --- ##############################


#####################################################
maxIter=10
numtopics = range(2, 5*numClasses, 2)
KFOLD=5
#####################################################


# TODO:
#  - abstract out the different LDA models
#  - abstract out metrics for clustering and classification
#  - extract metadata and evaluate


##############################################################################################

for _xLabel, _x in X.items():

    _bertmodel = featuremodel.bertModel()

    _xEmbed = _bertmodel.train(_x)

    for _featLabel, _featurizer in featurizers.items():

        _xx = _featurizer.train(_x)
        # _xxArr = copy.deepcopy(_xx).toarray()
        _vocab = _featurizer.getVocab()

        masterSKLearnDict = {}
        masterGensimDict = {}

        for i in numtopics:
            
            metricList = ["numClusters", "numTopics", "numAbstracts", "preprocessing?",  "perplexity", "coherence", "time", "train time", "test time", "evaluation time"]
            outDataFeatList = ["perplexity", "coherence", "homogeneity", "completeness", "silhouette", "time", "train time", "test time", "evaluation time"]
            # plotDataFeatList = ["perplexity", "coherence", "homogeneity", "completeness", "silhouette", "time"]
            plotDataFeatList = ["perplexity", "homogeneity", "completeness", "silhouette", "time"]
            sklearn_metricDict = {_metric: [] for _metric in outDataFeatList}
            gensim_metricDict = {_metric: [] for _metric in outDataFeatList}

            for _metric in metricList:
                if _metric == "numClusters":
                    sklearn_metricDict["numClusters"] = i
                    gensim_metricDict["numClusters"] = i
                elif _metric == "numTopics":
                    sklearn_metricDict["numTopics"] = numClasses
                    gensim_metricDict["numTopics"] = numClasses
                elif _metric == "numAbstracts":
                    sklearn_metricDict["numAbstracts"] = numDataPoints
                    gensim_metricDict["numAbstracts"] = numDataPoints
                elif _xLabel == "raw texts":
                    sklearn_metricDict["preprocessing?"] = False
                    gensim_metricDict["preprocessing?"] = False
                elif _xLabel == "preprocessed texts":
                    sklearn_metricDict["preprocessing?"] = True
                    gensim_metricDict["preprocessing?"] = True

            kfold = StratifiedKFold(n_splits=KFOLD)
            for j, (train_index, test_index) in enumerate(kfold.split(_xx, Yhigher)): # base topic

                yTrain = Yhigher[train_index]
                yTest = Yhigher[test_index]

                xTrainCSC = _xx[train_index]
                xTestCSC = _xx[test_index]
                xEmbTrain = _xEmbed[train_index]
                xEmbTest = _xEmbed[test_index]


                print(f"\nTraining SKLearn LDA with {_xLabel} and {_featLabel} features \nOn {numDataPoints} abstracts across {numClasses} topics. {i} Clusters, fold {j}...")

                # TODO : get and store "top topics"

                lda = clusteringmodel.LDA()

                start = default_timer()
                lda.train(xTrainCSC, vocab=_vocab, nClasses=i, maxIter=maxIter, y=yTrain)
                _trainTime = default_timer() - start
                print(f"Training took: {_trainTime:.3f}")
                preds, metrics = lda.test(xTestCSC, y=yTest, xEmb=xEmbTest)
                _testTime = default_timer() - start - _trainTime
                print(f"Testing took: {_testTime:.3f}")
                _silhouette = metrics[0]
                _homogeneity = metrics[3]
                _completeness = metrics[4]
                _perp = lda.perplexity(xTestCSC)
                # _coherence = lda.coherence(_x, _vocab)
                if VERBOSE: print(f"SKLearn LDA Perplexity: {_perp:.3f}")
                _evalTime = default_timer() - start - _trainTime - _testTime
                print(f"Evaluation took: {_evalTime:.3f}")

                # how to get topic keywords?
                # how to get topic probabilities

                _time = default_timer() - start
                sklearn_metricDict["time"].append(_time)
                sklearn_metricDict["train time"].append(_trainTime)
                sklearn_metricDict["test time"].append(_testTime)
                sklearn_metricDict["evaluation time"].append(_evalTime)
                sklearn_metricDict["perplexity"].append(_perp)
                # sklearn_metricDict["coherence"].append(_coherence)
                sklearn_metricDict["homogeneity"].append(_homogeneity)
                sklearn_metricDict["completeness"].append(_completeness)
                sklearn_metricDict["silhouette"].append(_silhouette)


                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

                print(f"\nTraining gensim LDA with {_xLabel} and {_featLabel} features \nOn {numDataPoints} abstracts across {numClasses} topics. {i} Clusters, fold {j}...")

                ldagensim = clusteringmodel.gensimLDA()

                start = default_timer()
                ldagensim.train(xTrainCSC, vocab=_vocab, nClasses=i, maxIter=maxIter, y=Yhigher)
                _trainTime = default_timer() - start
                print(f"Training took: {_trainTime:.3f}")
                preds, metrics = ldagensim.test(xTestCSC, y=yTest, xEmb=xEmbTest)
                _testTime = default_timer() - start - _trainTime
                print(f"Testing took: {_testTime:.3f}")
                _silhouette = metrics[0]
                _homogeneity = metrics[3]
                _completeness = metrics[4]
                _perp = ldagensim.perplexity(xTestCSC)
                # _coherence = ldagensim.coherence(xTestCSC)
                if VERBOSE: print(f"Gensim LDA Perplexity: {_perp:.3f}")
                _evalTime = default_timer() - start - _trainTime - _testTime
                print(f"Evaluation took: {_evalTime:.3f}\n\n")

                _time = default_timer() - start
                gensim_metricDict["time"].append(_time)
                gensim_metricDict["train time"].append(_trainTime)
                gensim_metricDict["test time"].append(_testTime)
                gensim_metricDict["evaluation time"].append(_evalTime)
                gensim_metricDict["perplexity"].append(_perp)
                # gensim_metricDict["coherence"].append(_coherence)
                gensim_metricDict["homogeneity"].append(_homogeneity)
                gensim_metricDict["completeness"].append(_completeness)
                gensim_metricDict["silhouette"].append(_silhouette)
            masterSKLearnDict[i] = sklearn_metricDict
            masterGensimDict[i] = gensim_metricDict

        df_sk = pd.DataFrame.from_dict(masterSKLearnDict, orient='index')
        df_sk.to_csv(f"../visualizations/experiment 3 - 2023-03-25/Classification Metrics for SKLearn LDA with {_featLabel} features on {_xLabel} data - base labels.csv")

        df_gn = pd.DataFrame.from_dict(masterGensimDict, orient='index')
        df_gn.to_csv(f"../visualizations/experiment 3 - 2023-03-25/Classification Metrics for Gensim LDA with {_featLabel} features on {_xLabel} data - base labels.csv")

        for _metric in plotDataFeatList:
            skStrVal = df_sk[_metric].values
            gnStrVal = df_gn[_metric].values

            skMeanVals = [np.mean(vals) for vals in skStrVal]
            gnMeanVals = [np.mean(vals) for vals in gnStrVal]
            """
            print(skStrVal)
            print(gnStrVal)

            print(isinstance(skStrVal[0][0], float))
            print(isinstance(skStrVal[0][0], str))

            for _strVal in skStrVal:
                skVal = [float(i) for i in _strVal[0].replace("[","").replace("]","").split(", ")]
            for _strVal in gnStrVal:
                gnVal = [float(i) for i in _strVal[0].replace("[","").replace("]","").split(", ")]
            """
            plt.plot(numtopics, skMeanVals, label="SKlearn LDA")
            plt.plot(numtopics, gnMeanVals, label="Gensim LDA")
            plottitle = f"Comparison of {_metric} performance\nAcross models for {_featLabel} features\nOn {_xLabel} data."
            plotname = f"../visualizations/experiment 3 - 2023-03-25/Comparison of {_metric} performance across models for {_featLabel} features on {_xLabel} data.png"
            plt.title(plottitle)
            plt.xlabel("Number of topics")
            plt.ylabel(_metric)
            plt.legend()
            plt.savefig(plotname)
            plt.close()
##############################################################################################


  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- evaluation ----# # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print("\n\nEvaluating model...\n\n")



# TODO : plot classification metrics 
