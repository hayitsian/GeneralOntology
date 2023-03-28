# Ian Hay - 2023-03-27
# https://github.com/hayitsian/General-Index-Visualization



from sklearn.base import TransformerMixin

import util # local file
from model.basemodel import BaseModel # local file



class BaseEvaluator(BaseModel, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        """
        Takes in and trains on the data `x` to return desired features `y`.
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def predict(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : ndarray[float] : list of datapoints n samples by d features
        Returns:
            - ypred : ndarray[int] : topic prediction of n samples bc y classes
        """
        util.raiseNotDefined()

    # TODO
    def save(self):
        """
        Saves this model and any associated experiments to a .txt file.\n
        Returns:
            - filename : str : the filename this model's .txt file was saved to
        """
        util.raiseNotDefined()

    # TODO: for view
    def __repr__(self):
        """
        Represents this model as a string.\n
        Returns:
            - tostring : str : string representation of this model.
        """
        util.raiseNotDefined()


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class ClusterEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def fit(self, x):
        return self
    
    def predict(self, x, y, labels=None):
        