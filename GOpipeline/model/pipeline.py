#!/usr/bin/env python
# Ian Hay - 2023-03-18
# https://github.com/hayitsian/General-Index-Visualization


from sklearn.pipeline import Pipeline

import util as util
from model.basemodel import BaseModel
from model.featuremodel import BaseFeaturizer
from model.evaluationmodel import BaseEvaluator



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


class AbstractPipeline():
    '''
    An abstractPipeline contains a feature pipeline, model pipeline and evaluation pipeline.
    '''

    def __init__(self):
        super().__init__()

    def compile(FeaturePipeline, ModelPipeline):
        '''
        compile() generates the pipeline object from the given components. 
        '''
        util.raiseNotDefined()

    def fit(self, x, y):
        '''
        train() trains the feature pipeline on input data x and labels y.
        '''
        util.raiseNotDefined()

    def transform(self, x, y=None):
        '''
        transform() trains the feature pipeline on input data x and labels y.
        '''
        util.raiseNotDefined()

    def predict(self, x):
        '''
        predict() passes the input data x through the pipeline and produces a prediction of labels.
        '''
        util.raiseNotDefined()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



class BasePipeline(AbstractPipeline):

    def __init__(self, featurizer=BaseFeaturizer(), topicModel=BaseModel()):
        super().__init__()
        self.featurizer=featurizer
        self.topicModel=topicModel

    def compile(self):
        self.pipeline = Pipeline(steps=[
            ('featurizer', self.featurizer),
            ('topic model', self.topicModel)])

    def fit(self, x, y):
        self.pipeline.fit(x, y)

    def predict(self, x):
        return self.pipeline.predict(x)

    
