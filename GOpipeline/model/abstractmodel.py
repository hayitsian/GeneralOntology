# Ian Hay - 2023-03-20
# https://github.com/hayitsian/General-Index-Visualization


import util


class abstractModel():

    def train(self, x=None, y=None):
        """
        Takes in and trains on the data `x` to predict desired features `y` if not None.\n
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def test(self, x=None):
        """
        Tests the trained model on the input datapoints `x`.\n
        Parameters:
            - x : ndarray[float] : list of datapoints n samples by d features
        Returns:
            - ypred : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def evaluate(self):
        """
        Compute evaluation metrics for this model.\n        
        Returns:
            - metrics : dict : computed metrics for this model.
        """
        util.raiseNotDefined()

    def save(self):
        """
        Saves this model and any associated experiments to a .txt file.\n
        Returns:
            - filename : str : the filename this model's .txt file was saved to
        """
        util.raiseNotDefined()

    def __repr__(self):
        """
        Represents this model as a string.\n
        Returns:
            - tostring : str : string representation of this model.
        """
        util.raiseNotDefined()

    
