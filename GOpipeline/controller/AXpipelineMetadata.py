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

numClasses = 5 # value is used later on
numDataPoints = 10000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
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

#TF-iDF
_name = "TF-iDF"
_tfidftransformer = featuremodel.tfidfModel()
featurizers[_name] = _tfidftransformer

# BERT
_name = "BERT"
_bertmodel = featuremodel.bertModel()
featurizers[_name] = _bertmodel


# TODO: n-gram model (1-5 gram)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # #  ---- topic modeling ---- # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

####################### --- clustering + metadata extraction --- ##############################


#####################################################
_epochs = 5000
numHidden1 = numHidden2 = numHidden3 = 64

learningRate = 0.2

nEstimators=1000
maxIter=50000

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
            for i, (train_index, test_index) in enumerate(kfold.split(_xx, Yhigher)):
                print(f"Training {_xLabel} with {_featLabel} features and {_modelLabel} model\nOn {numDataPoints} abstracts across {numClasses} topics. Fold {i}...")
                xtrain = _xx[train_index]
                ytrain = Yhigher[train_index]
                xtest = _xx[test_index]
                ytest = Yhigher[test_index]
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
        for i, (train_index, test_index) in enumerate(kfold.split(_xx, Yhigher)):
            print(f"Training {_xLabel} with {_featLabel} features and Neural Network model\nOn {numDataPoints} abstracts across {numClasses} topics. Fold {i}...")
            _model = neuralnetworkmodel.FFNN(numInput, numClasses, hidden_size_1=numHidden1, hidden_size_2=numHidden2, hidden_size_3=numHidden3, epochs=_epochs)
            xtrain = _xx[train_index]
            ytrain = Yhigher[train_index]
            xtest = _xx[test_index]
            ytest = Yhigher[test_index]
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
            df.to_csv(f"../visualizations/experiment 2 - 2023-03-22/Classification Metrics for {_modelLabel} with {_featLabel} features on {_xLabel} data - base labels.csv")

    