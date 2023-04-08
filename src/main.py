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
import model.preprocessing.AXpreprocessing as AXpreprocessing # local file
import model.featuremodel as featuremodel # local file
import model.clusteringmodel as clusteringmodel # local file
import model.classifiermodel as classifiermodel # local file
import model.neuralnetworkmodel as neuralnetworkmodel # local file
from model.evaluationmodel import ClusterEvaluator # local file
from model.pipeline import BasePipeline # local file


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # ---- import & preprocess ---- # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#####################################################
_dataCol = "abstract"
_labelCol = "categories"
_topLabelCol = "top category"
_baseLabelCol = "base categories"
_topBaseLabelCol = "top base category"

numClasses = 8 # value is used later on
numLowerClasses = 45 # value is used later on
numDataPoints = 2200 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
VERBOSE=True
#####################################################

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# open the data file
if VERBOSE: print("\n\nImporting data...\n\n")


            ##### I/O ######
# take in filename as a command line argument
_rawFilename = sys.argv[1] # file path
            ##### I/O ######

importer = AXpreprocessing.AXimporter()
preprocessor = AXpreprocessing.AXpreprocessor(n_jobs=14, verbose=VERBOSE)


df = importer.importData(_rawFilename,verbose=VERBOSE)
df = importer.parseLabels(_labelCol, _topLabelCol, _baseLabelCol, _topBaseLabelCol, verbose=VERBOSE)

categoryCounter = Counter(list(itertools.chain(*df[_labelCol].values)))
if VERBOSE: print(f"\nTotal number of categories: {len(list(set(categoryCounter.keys())))}\n")
if VERBOSE: print(categoryCounter)

topCategoryCounter = Counter(df[_topLabelCol].values)
if VERBOSE: print(f"\nTotal number of top categories: {len(list(set(topCategoryCounter.keys())))}\n")
if VERBOSE: print(topCategoryCounter)

differentCats = {k:v for k,v in categoryCounter.items() if k not in topCategoryCounter}
if VERBOSE: print(f"\nDifference of above categories:\n{differentCats}\n")

baseCategoryCounter = Counter(df[_topBaseLabelCol].values)
if VERBOSE: print(f"\nTotal number of base categories: {len(list(set(baseCategoryCounter.keys())))}\n")
if VERBOSE: print(baseCategoryCounter)


# TODO: tokens per text, vocab lenth of texts, topic distrubtion and co-occurence (lower & higher)



if VERBOSE: print("\n\nPreprocessing data...\n\n")
dfSubsetLowerLabels = importer.getSubsetFromNClasses(df, _topLabelCol, numLowerClasses, numDataPoints, verbose=VERBOSE)
dfSubsetHigherLabels = importer.getSubsetFromNClasses(df, _topBaseLabelCol, numClasses, numDataPoints, verbose=VERBOSE)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- featurization ---- # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

Xlower, Ylower = importer.splitXY(dfSubsetLowerLabels, _dataCol, _topLabelCol)
Xhigher, Yhigher = importer.splitXY(dfSubsetHigherLabels, _dataCol, _topBaseLabelCol)


####################
_texts = Xhigher
####################


# preprocess texts
_abstractPrior = _texts.values[:3]
if VERBOSE: print(f"\nAbstracts before preprocessing:\n{_abstractPrior}\n")
_tokensPrior = list(itertools.chain(*[_txt.split() for _txt in _texts]))
_vocabSizePrior = len(set(_tokensPrior))
if VERBOSE: print(f"\nToken number before preprocessing: {len(_tokensPrior)}")
if VERBOSE: print(f"Vocab size before preprocessing: {_vocabSizePrior}\n")

_preprocessedTexts = preprocessor.transform(_texts) # multiprocessing

_tokensAfter = list(itertools.chain(*[_txt.split() for _txt in _preprocessedTexts]))
_vocabSizeAfter = len(set(_tokensAfter))
if VERBOSE: print(f"Token number after preprocessing: {len(_tokensAfter)}")
if VERBOSE: print(f"Vocab size after preprocessing: {_vocabSizeAfter}\n")
_abstractAfter = _preprocessedTexts.values[:3]
if VERBOSE: print(f"\nAbstracts after preprocessing:\n{_abstractAfter}\n")

#####################################################

_preprocessedTexts = _preprocessedTexts.values
_texts = _texts.values

X = {"preprocessed texts": _preprocessedTexts , "raw texts": _texts }
Y = Yhigher.values

#####################################################


featurizers = {}

# BoW
_name = "Bag-of-words"
_vectorizer = featuremodel.bowModel(minDF=0.0001, maxDF=0.75) # want to test hyperparameters
featurizers[_name] = _vectorizer

"""#TF-iDF
_name = "TF-iDF"
_tfidftransformer = featuremodel.tfidfModel()
featurizers[_name] = _tfidftransformer

# BERT
_name = "BERT"
_bertmodel = featuremodel.bertModel()
featurizers[_name] = _bertmodel"""

# n-gram model (1-3 gram)
_name = "n-grams"
_ngramVectorizer = featuremodel.bowModel(ngram_range=(1,3), minDF=0.0001, maxDF=0.75) # want to test hyperparameters
# https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer
featurizers[_name] = _ngramVectorizer

# improve n-gram data struct:
# https://stackoverflow.com/questions/45264957/storing-ngram-model-python


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # #  ---- topic modeling ---- # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


####################### --- clustering + metadata extraction --- ##############################


#####################################################
maxIter=10
numtopics = range(2, 3*numClasses, 2)
KFOLD=5
#####################################################


# TODO:
#  - abstract out the different LDA models - mostly done
#  - abstract out metrics for clustering and classification - mostly done
#  - extract metadata and evaluate
#  - assign topics and metadata to datapoints, output as DF/csv
#  - guided (semi-supervised) LDA: https://github.com/vi3k6i5/guidedlda
#  - ensemble LDA implementation


##############################################################################################

_x = _preprocessedTexts
_y = Y

_xLabel = "preprocessed texts"
_featLabel = "n-grams"

_bertmodel = featuremodel.bertModel()

_xEmbed = _bertmodel.transform(_x) # multiprocessing ??

# TODO this is bad, need to find way to get rid of
_ngramVectorizer.fit(_x)
_vocab = _ngramVectorizer.getVocab() # how to get vocab to pass along???


masterSKLearnDict = {}
masterGensimDict = {}

for i in numtopics:

    outDataFeatList = ["perplexity", "homogeneity", "completeness", "silhouette", "time", "topic words"]
    plotDataFeatList = ["perplexity", "homogeneity", "completeness", "silhouette"]#, "coherence"]
    sklearn_metricDict = {_metric: [] for _metric in outDataFeatList}
    gensim_metricDict = {_metric: [] for _metric in outDataFeatList}

    kfold = StratifiedKFold(n_splits=KFOLD)
    for j, (train_index, test_index) in enumerate(kfold.split(_x, _y)): # base topic

        yTrain = _y[train_index]
        yTest = _y[test_index]

        xTrain = _x[train_index]
        xTest = _x[test_index]
        xEmbTrain = _xEmbed[train_index]
        xEmbTest = _xEmbed[test_index]


        if VERBOSE: print(f"\nTraining SKLearn LDA with {_xLabel} and {_featLabel} features \nOn {numDataPoints} abstracts across {numClasses} topics. {i} Clusters, fold {j}...")

        featurizer = _ngramVectorizer
        lda = clusteringmodel.SKLearnLDA(vocab=_vocab, nClasses=i, maxIter=maxIter, nJobs=14)
        evaluator = ClusterEvaluator(yTrue=yTest, model=lda, xEmb=xEmbTest, vocab=_vocab)

        pipeline = BasePipeline(featurizer, lda)
        pipeline.compile()

        start = default_timer()
        pipeline.fit(xTrain, y=yTrain) # multiprocessing
        _trainTime = default_timer() - start
        if VERBOSE: print(f"Training took: {_trainTime:.3f}")

        preds = pipeline.predict(xTest)
        _testTime = default_timer() - start - _trainTime
        if VERBOSE: print(f"Testing took: {_testTime:.3f}")
        
        _topicWords = lda.print_topics(nTopWords=10, verbose=VERBOSE)
        metrics = evaluator.predict(_ngramVectorizer.transform(xTest), preds)
        _time = default_timer() - start

        sklearn_metricDict["time"].append(_time)
        sklearn_metricDict["topic words"].append(_topicWords)

        for _metric in plotDataFeatList:
            sklearn_metricDict[_metric].append(metrics[_metric])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        if VERBOSE: print(f"\nTraining Gensim LDA with {_xLabel} and {_featLabel} features \nOn {numDataPoints} abstracts across {numClasses} topics. {i} Clusters, fold {j}...")

        featurizer = featuremodel.gensimBowModel(ngram_range=(1,3))
        lda = clusteringmodel.GensimLDA(vocab=_vocab, nClasses=i, maxIter=maxIter, nJobs=14)
        evaluator = ClusterEvaluator(yTrue=yTest, model=lda, xEmb=xEmbTest, vocab=_vocab)

        pipeline = BasePipeline(featurizer, lda)
        pipeline.compile()

        start = default_timer()
        pipeline.fit(xTrain, y=yTrain) # multiprocessing
        _trainTime = default_timer() - start
        if VERBOSE: print(f"Training took: {_trainTime:.3f}")

        preds = pipeline.predict(xTest)
        _testTime = default_timer() - start - _trainTime
        if VERBOSE: print(f"Testing took: {_testTime:.3f}")
        
        _topicWords = lda.print_topics(nTopWords=10, verbose=VERBOSE)
        metrics = evaluator.predict(_ngramVectorizer.transform(xTest), preds)
        _time = default_timer() - start

        gensim_metricDict["time"].append(_time)
        gensim_metricDict["topic words"].append(_topicWords)

        for _metric in plotDataFeatList:
            gensim_metricDict[_metric].append(metrics[_metric])


    masterSKLearnDict[i] = sklearn_metricDict
    masterGensimDict[i] = gensim_metricDict

df_sk = pd.DataFrame.from_dict(masterSKLearnDict, orient='index')
df_sk.to_csv(f"../visualizations/experiment 5 - 2023-03-28/Clustering Metrics for SKLearn LDA with {_featLabel} features on {_xLabel} data - base labels.csv")

df_gn = pd.DataFrame.from_dict(masterGensimDict, orient='index')
df_gn.to_csv(f"../visualizations/experiment 5 - 2023-03-28/Clustering Metrics for Gensim LDA with {_featLabel} features on {_xLabel} data - base labels.csv")

for _metric in plotDataFeatList:
    skStrVal = df_sk[_metric].values
    gnStrVal = df_gn[_metric].values

    skMeanVals = [np.mean(vals) for vals in skStrVal] # plot the mean from k-fold cross validation
    gnMeanVals = [np.mean(vals) for vals in gnStrVal] # TODO: add std dev or SEM bars

    # TODO: concatenate more conditions to the same plot
    plt.plot(numtopics, skMeanVals, label="SKlearn LDA")
    plt.plot(numtopics, gnMeanVals, label="Gensim LDA")
    plottitle = f"Comparison of {_metric} performance\nAcross models for {_featLabel} features\nOn {_xLabel} data."
    plotname = f"../visualizations/experiment 5 - 2023-03-28/Comparison of {_metric} performance across models for {_featLabel} features on {_xLabel} data.png"
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


# TODO : plot evaluation (and any other) metrics 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- run the program ---- # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import argparse
from controller import BaseController


def main():

    parser = argparse.ArgumentParser()

    # add command line arguments

    # required
    parser.add_argument("action", help="The action for this program to take.",
                        choices=["query", "train", "update"])
    parser.add_argument("type", help="The data type to use for this model.",
                        choices=["AX", "GI", "PM"])
    parser.add_argument("data", help="The datasource to use for this model")
    parser.add_argument("model", help="The model type to use.",
                        choices=["LDA", "SBM", "NN"])

    # optional
    parser.add_argument("-v", "--verbose", help="Set the verbosity of the output.",
                        action="store_true")
    parser.add_argument("-s", "--save", help="Whether to save the model or not and to what filename.")
    parser.add_argument("-l", "--load", help="Whether to load a pretrained model and the filename to load from.")
    parser.add_argument("-o", "--output", help="The output for this program to make.",
                        choices=["JSON", "CSV", "TXT"], default="JSON")


    # build appropriate controller and pass into it
    # for now, just builds a base controller
    # TODO: extend the controller and build based on input args
    args = parser.parse_args()
    cont = BaseController(args)
    cont.run()


main()