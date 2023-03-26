#!/usr/bin/env python
# Ian Hay - 2023-03-18
# https://github.com/hayitsian/General-Index-Visualization

import sys
import numpy as np
import multiprocessing as mp

import string
import spacy 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import  LogisticRegressionCV
from sklearn.pipeline import Pipeline

import util as util
import GOpipeline.model.AXpreprocessing as AXpreprocessing

nlp = spacy.load("en_core_web_sm")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



class abstractPipeline():
    '''
    An abstractPipeline contains a feature pipeline, model pipeline and evaluation pipeline.
    '''

    def compile(FeaturePipeline, ModelPipeline, EvaluationPipeline):
        '''

        compile() generates the pipeline object from the given components. 
    
        '''
        util.raiseNotDefined()

    def fit(x, y):
        '''

        train() trains the feature pipeline on input data x and labels y.
    
        '''
        util.raiseNotDefined()

    def predict(x):
        '''

        predict() passes the input data x through the pipeline and produces a prediction of labels.

        '''
        util.raiseNotDefined()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



class pipeline():

    def __init__(self, normalizer=AXpreprocessing.TextPreprocessor(), featurizer=TfidfVectorizer(), classifier=LogisticRegressionCV()):
        self.pipeline = Pipeline(steps=[
            ('normalize', normalizer),
            ('features', featurizer), 
            ('classifier', classifier)])

    def train(self, x, y):
        self.pipeline.fit(x, y)

    def test(self, x):
        return self.pipeline.predict(x)
    

def main():
   # take in filename as a command line argument
   _rawFilename = sys.argv[1] # file path
   
   pass

main()