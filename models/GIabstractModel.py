# Ian Hay - 2023-02-23

import models.util as util

class model():
    """
    An abstract model class, defining train() and test() methods.
    """

    def train(self, x, y):
        util.raiseNotDefined()

    def test(self, x):
        util.raiseNotDefined()


class featurizeModel(model):

    def train(self, x, y):
        """
        Takes in the lowercase ngrams and topic label, and trains on the data to return desired features.
        Parameters:
            - x : list[str] : list of ngrams as strings
            - y : list[str] | str : topic of ngrams
        Returns:
            - feats : list[] : list of features of x
        """
        util.raiseNotDefined()


    def test(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : list[str] : list of ngrams as strings
        Returns:
            - feats : list[] : list of features of x
        """
        util.raiseNotDefined()


class topicModel(model):

    def train(self, x, y):
        """
        Takes in the features and topic label, and trains on the data to cluster the documents by topic.
        Parameters:
            - x : list[str] : list of ngrams as strings
            - y : list[str] | str : topic of ngrams
        Returns:
            - topics : list[] : list of topics of x
        """
        util.raiseNotDefined()


    def test(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : list[str] : list of ngrams as strings
        Returns:
            - topics : list[] : list of topics of x
        """
        util.raiseNotDefined()