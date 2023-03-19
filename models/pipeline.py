# Ian Hay - 2023-03-18

import numpy as np
import multiprocessing as mp

import string
import spacy 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import  LogisticRegressionCV
from sklearn.pipeline import Pipeline


import AXpreprocessing

nlp = spacy.load("en_core_web_sm")

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