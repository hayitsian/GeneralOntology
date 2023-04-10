# Ian Hay - 2023-04-08
# https://github.com/hayitsian/General-Index-Visualization


from pipeline import BasePipeline
from model.featuremodel import bowModel
from model.clusteringmodel import SKLearnLDA

class skLDAPipeline(BasePipeline):

  def __init__(self, nClasses, nGrams=(1,3), minDF=0.00001, maxDF=0.75):
    super().__init__(featurizer=bowModel(nGrams, minDF=minDF, maxDF=maxDF),
                     topicModel=SKLearnLDA(nClasses))

  def compile(self):
    return super().compile()
  
  def fit(self, x, y):
    return super().fit(x, y)
  
  def predict(self, x):
    return super().predict(x)