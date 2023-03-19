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
import matplotlib.pyplot as plt

import AXpreprocessing # local file
import classifiermodel # local file
import neuralnetwork
import util # local file
import optimize # local file
import clusteringmodel # local file
import featuremodel


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

   numClasses = 10 # value is used later on
   numDataPoints = 20000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
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


   ####################### --- clustering --- ##############################

   ### specify number of classes

   km = clusteringmodel.KMeans()
   lda = clusteringmodel.LDA()
   nmf = clusteringmodel.NMF()

   _models = {
      "KMeans": km,
      "LDA": lda,
      "NMF": nmf
   }

   maxIter=50
   numtopics = range(2, 2*numClasses)

   for _xLabel, _x in X.items(): 

      for _featLabel, _featurizer in featurizers.items():
         print("\nFeaturizing data...\n")
         _xx = _featurizer.train(_x)

         metricList = ["silhouette", "calinski", "davies", "homogeneity", "completeness", "vMeasure", "rand"]
         masterDict = {}

         for _modelLabel, _model in _models.items():
            print("\nModeling data...\n")
            silhouette = []
            calinski = []
            davies = []
            homogeneity = []
            completeness = []
            vMeasure = []
            rand = []
 
            for i in numtopics:
               print(f"Training {_xLabel} with {_featLabel} features and {_modelLabel} model using {i} topics...")
               start = default_timer()
               preds, metrics = _model.train(_xx, nClasses=i, maxIter=maxIter, y=Y)
               _time = default_timer() - start
               print(f"Training took: {_time:.3f}")
               silhouette.append(metrics[0])
               calinski.append(metrics[1])
               davies.append(metrics[2])
               homogeneity.append(metrics[3])
               completeness.append(metrics[4])
               vMeasure.append(metrics[5])
               rand.append(metrics[6])

            metricDict = {}
            metricDict["silhouette"] = silhouette
            metricDict["calinski"] = calinski
            metricDict["davies"] = davies
            metricDict["homogeneity"] = homogeneity
            metricDict["completeness"] = completeness
            metricDict["vMeasure"] = vMeasure
            metricDict["rand"] = rand
            masterDict[_modelLabel] = metricDict
            print(completeness)

         for _metric in metricList:
            for _modelLabel, _metrics in masterDict.items():
               plt.plot(numtopics, _metrics[_metric], label=_modelLabel)
            plottitle = f"Comparison of {_metric} performance\nAcross models for {_featLabel} features."
            plotname = f"Comparison of {_metric} performance across models for {_featLabel} features"
            plt.title(plottitle)
            plt.xlabel(numtopics)
            plt.ylabel(_metric)
            plt.legend()
            plt.savefig(plotname)
            plt.close()
         
         for _modelLabel, metrics in masterDict.items():
            metrics["num topics"] = numtopics
            df = pd.DataFrame.from_dict(metrics)
            df.to_csv(f"Metrics for {_modelLabel} with {_featLabel} features.csv")


   ### does not specify number of classes

   """   
   hdbscan = HDBSCAN(min_cluster_size=10,  metric='euclidean', cluster_selection_method='eom', prediction_data=True)
   hdbscan.fit(_tfidf.toarray())
   pred = hdbscan.labels_
   pred += 1
   print(pred)
   util.getClusterMetrics(pred, x=_tfidf.toarray(), labels=Y, supervised=True)
   """

   """   
   hdbscan = HDBSCAN(min_cluster_size=10,  metric='euclidean', cluster_selection_method='eom', prediction_data=True)
   hdbscan.fit(_bertembeddings)
   pred = hdbscan.labels_
   pred += 1
   util.getClusterMetrics(pred, x=_bertembeddings, labels=Y, supervised=True)
   """

   """
   print("\nBERTopic, raw Texts + BERT Embedding...")
   bertopic = BERTopic(embedding_model=_bertmodel, 
                              umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
                              hdbscan_model=HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
                              vectorizer_model=CountVectorizer(stop_words="english"),
                              ctfidf_model=ClassTfidfTransformer(),
                              low_memory=False,
                              verbose=True,
                              calculate_probabilities=True)
   topics, probs = bertopic.fit_transform(_texts)
   pred = util.getTopPrediction(probs)
   util.getClusterMetrics(pred, labels=Y, supervised=True)

   print("\nTop2Vec, raw Texts + universal-sentence-encoder...")
   top2vec = Top2Vec(documents=_texts, embedding_model='universal-sentence-encoder')
   docsids = top2vec.document_ids
   pred = top2vec.get_documents_topics(docsids, num_topics=1)[0]
   util.getClusterMetrics(pred, labels=Y, supervised=True)



   models = {
         "HDBSCAN": HDBSCAN(min_cluster_size=10,  metric='euclidean', cluster_selection_method='eom', prediction_data=True),
         "BERTopic": BERTopic(embedding_model=_bertmodel, 
                              umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
                              hdbscan_model=HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
                              vectorizer_model=CountVectorizer(stop_words="english"),
                              ctfidf_model=ClassTfidfTransformer(),
                              low_memory=False,
                              verbose=True,
                              calculate_probabilities=True),
         "Top2Vec": Top2Vec(documents=_texts, embedding_model='universal-sentence-encoder') # this one has texts already passed in
         # TODO: Neural Topic Modeling
      }
   """


   # TODO : plot top words for topics

   #####################################################
   _epochs = 100000
   # numHidden1 = list(range(64, 259, 64))
   # numHidden2 = list(range(64, 259, 64))
   # numHidden3 = list(range(64, 259, 64))
   numHidden1 = numHidden2 = numHidden3 = [64, 128, 256]

   learningRate = 0.2
   #####################################################

   # supervised - FFNN, decision tree (boosted), TODO: SVM, EM, naive bayes?, logistic regression?

   """   optmizer = optimize.modelOptimizer()
   metricDF = optmizer.optimizeNN(X, Y, numHidden1, numHidden2, numHidden3, _epochs, learningRate, verbose=True)
   print(metricDF)
   metricDF.to_csv("optimization_metrics.csv")"""

   # some neural network topic models: https://github.com/zll17/Neural_Topic_Models#NVDM-GSM


   # TODO : plot classification metrics 

  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- evaluation ----# # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

   print("\n\nEvaluating model...\n\n")





main()