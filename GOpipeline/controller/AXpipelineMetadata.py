#!/usr/bin/env python
#
# Ian Hay - 2023-02-23
# https://github.com/hayitsian/General-Index-Visualization

import sys
import os
from collections import Counter
import itertools
import numpy as np
import pandas as pd
from timeit import default_timer

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath("/home/ian/Documents/GitHub/General-Index-Visualization/GOpipeline"))

import util # local file
import model.AXpreprocessing as AXpreprocessing # local file
import model.featuremodel as featuremodel # local file
import model.clusteringmodel as clusteringmodel # local file
import model.classifiermodel as classifiermodel # local file
import model.neuralnetworkmodel as neuralnetworkmodel # local file


# take in filename as a command line argument
_rawFilename = sys.argv[1] # file path

#####################################################
_dataCol = "abstract"
_labelCol = "categories"
_yLabel = "top category"
_baseLabelCol = "base categories"
_baseLabel = "base category"

numClasses = 4 # value is used later on
numDataPoints = 500 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
#####################################################

# open the data file
print("\n\nImporting data...\n\n")

preprocessor = AXpreprocessing.preprocessor()
df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=True, classify=False)

categoryCounter = Counter(list(itertools.chain(*df[_labelCol].values)))
print(f"\nTotal number of categories: {len(list(set(categoryCounter.keys())))}\n")
print(categoryCounter)

topCategoryCounter = Counter(df[_yLabel].values)
print(f"\nTotal number of top categories: {len(list(set(topCategoryCounter.keys())))}\n")
print(topCategoryCounter)


df[_baseLabelCol] = df[_labelCol].apply(AXpreprocessing.getBaseCategories)
df[_baseLabel] = df[_baseLabelCol].str[0]
baseCategoryCounter = Counter(df[_baseLabel].values)
print(f"\nTotal number of base categories: {len(list(set(baseCategoryCounter.keys())))}\n")
print(baseCategoryCounter)


print("\n\nPreprocessing data...\n\n")
_texts, Ylower = preprocessor.getStratifiedSubset(df, _yLabel, _dataCol, numClasses, numDataPoints, verbose=True)
_texts, Yhigher = preprocessor.getStratifiedSubset(df, _baseLabel, _dataCol, numClasses, numDataPoints, verbose=True)

Ylower = Ylower.values
Yhigher = Yhigher.values

stratBaseCounter = Counter(Yhigher)
print(stratBaseCounter)

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

#####################################################

X = {"raw texts": _texts, "preprocessed texts": _preprocessedTexts}

#####################################################


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- featurization ---- # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


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
_ngramVectorizer = featuremodel.bowModel(ngram_range=(1,5))
featurizers[_name] = _ngramVectorizer


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # #  ---- topic modeling ---- # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

####################### --- clustering + metadata extraction --- ##############################


#####################################################
maxIter=50
numtopics = range(2, 5*numClasses)
VERBOSE=True
#####################################################


km = clusteringmodel.KMeans()
lda = clusteringmodel.LDA()
nmf = clusteringmodel.NMF()
ldagensim = clusteringmodel.gensimLDA()

_models = {
    "KMeans": km,
    "LDA": lda,
    "LDA gensim": ldagensim,
    "NMF": nmf
}

for _xLabel, _x in X.items(): 

    for _featLabel, _featurizer in featurizers.items():
        print("\nFeaturizing data...\n")
        _xx = _featurizer.train(_x)

        masterDict = {}
        # metricList = ["f1", "roc_auc", "accuracy", "recall", "precision", "confusion matrix"]
        metricList = ["silhouette", "calinski", "davies", "homogeneity", "completeness", "vMeasure", "rand"] # likely want to abstract these out

        for _modelLabel, _model in _models.items():

            silhouette = []
            calinski = []
            davies = []
            homogeneity = []
            completeness = []
            vMeasure = []
            rand = []
            time = []
            
            metricDict = {}
            for _metric in metricList:
                metricDict[_metric] = []

            for i in numtopics:
                print(f"Training {_xLabel} with {_featLabel} features and {_modelLabel} model\nOn {numDataPoints} abstracts across {numClasses} topics. {i} Topics...")
                start = default_timer()
                preds, metrics = _model.train(_xx, nClasses=i, maxIter=maxIter, y=Yhigher)
                _time = default_timer() - start
                print(f"Training took: {_time:.3f}")
                silhouette.append(metrics[0])
                calinski.append(metrics[1])
                davies.append(metrics[2])
                homogeneity.append(metrics[3])
                completeness.append(metrics[4])
                vMeasure.append(metrics[5])
                rand.append(metrics[6])
                time.append(_time)
                if VERBOSE: print(f"Completeness: {metrics[4]:.3f}\n")

            metricDict = {}
            metricDict["silhouette"] = silhouette
            metricDict["calinski"] = calinski
            metricDict["davies"] = davies
            metricDict["homogeneity"] = homogeneity
            metricDict["completeness"] = completeness
            metricDict["vMeasure"] = vMeasure
            metricDict["rand"] = rand
            metricDict["time"] = time
            masterDict[_modelLabel] = metricDict
        

        for _modelLabel, metrics in masterDict.items():
            df = pd.DataFrame.from_dict(metrics)
            df.to_csv(f"../visualizations/experiment 3 - 2023-03-25/Clustering Metrics for {_modelLabel} with {_featLabel} features on {_xLabel} data - base labels.csv")

    