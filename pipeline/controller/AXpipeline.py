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

import pipeline.model.AXpreprocessing as AXpreprocessing # local file
import classifiermodel # local file
import neuralnetworkmodel as neuralnetworkmodel
import util # local file
import pipeline.controller.optimizer as optimizer # local file
import clusteringmodel # local file

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from top2vec import Top2Vec
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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

   numClasses = 20 # value is used later on
   numDataPoints = 40000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution
   #####################################################

   # open the data file
   print("\n\nImporting data...\n\n")

   preprocessor = AXpreprocessing.preprocessor()
   df = preprocessor.importData(_rawFilename, _labelCol, _yLabel, verbose=True, classify=False)

   _texts, Y = preprocessor.getStratifiedSubset(df, _yLabel, _dataCol, numClasses, numDataPoints)
 
   print("\n\nPreprocessing data...\n\n")

   # TODO preprocess texts for BOW and TF-IDF, dimensional reduction / feature selection
   _preprocessedTexts = preprocessor.preprocessTexts(_texts)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- featurization ---- # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


  # currently doing minimal preprocessing (no POS tagging, stemming, etc.)

   print("\n\nFeaturizing data...\n\n")

   # BoW
   # _textWords = [re.findall(r'\w+', _text) for _text in _texts] # this is super slow (regex :( )
   print("Bag-of-words\n")
   _vectorizer = CountVectorizer()
   _bow = _vectorizer.fit_transform(_preprocessedTexts)

   #TF-iDF
   print("TF-iDF\n")
   _tfidftransformer = TfidfTransformer()
   _tfidf = _tfidftransformer.fit_transform(_bow)

   # BERT
   print("BERT")
   _bertmodel = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device) # dilbert model
   _bertembeddings = _bertmodel.encode(_texts, show_progress_bar=True, normalize_embeddings=True)

   # doc2vec - very unimpressive thus far, and does not support GPUs
   """   
   print("doc2vec\n")
   _taggedDocs = [TaggedDocument(doc, [i]) for i, doc in enumerate(_texts)]
   _docmodel = Doc2Vec(_taggedDocs, vector_size=768, window=3, min_count=1, workers=16) # many hyperparameters to optimize
   _docembeddings = _docmodel.dv.get_normed_vectors()
   """

   # TODO: custom embeddings, ngrams
   # TODO: add feature extraction with preprocessed text


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # ---- feature selection ---- # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


   # TODO: PLSA, PCA, t-SNE, UMAP, etc. or some lasso regularization


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # #  ---- topic modeling ---- # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


   print("\n\nModeling data...\n")

   X = {"BERT":_bertembeddings} # , "doc2vec":_docembeddings , "bag-of-words":_bow.toarray(), "tf-idf":_tfidf.toarray()}


   ####################### --- clustering --- ##############################

   print("\nKMeans Clustering, TF-IDF Features...")
   start = default_timer()
   km = clusteringmodel.KMeans(nClasses=numClasses, maxIter=5000)
   km.train(_tfidf.toarray(), Y, verbose=True)
   _time = default_timer() - start
   
   print("\nKMeans Clustering, BERT Embedding Features...")
   km = clusteringmodel.KMeans(nClasses=numClasses, maxIter=5000)
   km.train(_bertembeddings, Y, verbose=True)

   print("\nLDA, BOW Features...")
   lda = clusteringmodel.LDA(nClasses=numClasses, maxIter=10)
   lda.train(_bow.toarray(), Y, verbose=True)


   """   hdbscan = HDBSCAN(min_cluster_size=10,  metric='euclidean', cluster_selection_method='eom', prediction_data=True)
      hdbscan.fit(_tfidf.toarray())
      pred = hdbscan.labels_
      pred += 1
      print(pred)
      util.getClusterMetrics(pred, x=_tfidf.toarray(), labels=Y, supervised=True)
   """

   """   hdbscan = HDBSCAN(min_cluster_size=10,  metric='euclidean', cluster_selection_method='eom', prediction_data=True)
      hdbscan.fit(_bertembeddings)
      pred = hdbscan.labels_
      pred += 1
      util.getClusterMetrics(pred, x=_bertembeddings, labels=Y, supervised=True)
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
         "KMeans": KMeans(n_clusters=numClasses),
         "LDA": LDA(n_components=numClasses),
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



   ####################### --- metadata --- ##############################

   # TODO: will want to talk to team about this

   ####################### --- predictions --- ##############################


   #######################
   _epochs = 20000
   numHidden1 = 256
   numHidden2 = 512
   numHidden3 = 256
   learningRate = 0.2
   maxIter=10000
   nEstimators = 1000
   nEstimatorsAda = 50
   verbose = True
   numOutput = numClasses
   #######################


   # supervised - FFNN, decision tree (boosted), TODO: SVC/M, naive bayes?, logistic regression?
   # Reinforcement learning - PPO ???

   models = {
            # "randomForest": model.RandomForestClassifier(nEstimators=nEstimators, criterion="entropy", maxDepth=5, minSamplesSplit=5),
             "logisticRegressor": classifiermodel.logisticRegression(maxIter=maxIter),
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

      optimizer = optimizer.modelOptimizer()
      ffNN = neuralnetworkmodel.FFNN(input_size=numInput, output_size=numOutput, criterion=_criterion, hidden_size_1=numHidden1, hidden_size_2=numHidden2, hidden_size_3=numHidden3, learningRate=learningRate, epochs=_epochs).to(device)
      
      start = default_timer()
      f1, roc, acc, recall, precision = ffNN.train(xTrain, yTrain, verbose=verbose)
      # optimizer.runNN(ffNN, xTrain, yTrain, xTest, yTest, verbose=True)
      _time = default_timer() - start

      print(f"\nTrained FFNN with {_criterion} loss on {_dataLabel} with {numDataPoints} abstracts across {numClasses} topics:\n"
            + f"{_epochs} epochs, {learningRate} learning rate, {numHidden1} + {numHidden2} + {numHidden3} hidden neurons.\n"
            + f"Training took {_time:.3f} seconds\n"
            + f"Metrics: \nF1 = {f1:0.3f}  \nROC AUC = {roc:0.3f}  \nAccuracy = {acc:0.3f}  \nRecall = {recall:.3f}  \nPrecision = {precision:.3f}\n")



   """   discreteX = _bow.toarray()
   print(f"\nTraining Naive Bayes on BOW data...\n")
   NB = classifiermodel.NaiveBayes()
   NB.train(discreteX, Y)
   predy = NB.test(discreteX)
   metrics = util.multi_label_metrics(predy, Y)
   print(metrics)"""


   # some neural network topic models: https://github.com/zll17/Neural_Topic_Models#NVDM-GSM

   # TODO

   ####################### --- embed topics --- ##############################

   # TODO: autoencoder, transformer, pretrained transformer + dimensional reduction

   ######################## --- networking --- ###############################

   # TODO: may want to talk to Chris about this
  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # ---- evaluation ----# # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

   print("\n\nEvaluating model...\n\n")

   # TODO



main()