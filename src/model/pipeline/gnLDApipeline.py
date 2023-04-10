# Ian Hay - 2023-04-08
# https://github.com/hayitsian/General-Index-Visualization


from pipeline import BasePipeline
from model.featuremodel import gensimBowModel
from model.clusteringmodel import GensimLDA

class gnLDAPipeline(BasePipeline):

  def __init__(self, nClasses, nGrams=(1,3)):
    super().__init__(featurizer=gensimBowModel(nGrams),
                     topicModel=GensimLDA(nClasses, vocab))

  def compile(self):
    return super().compile()
  
  def fit(self, x, y):
    return super().fit(x, y)
  
  def predict(self, x):
    return super().predict(x)