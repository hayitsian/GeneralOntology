# Ian Hay - 2023-02-23


import collections
import pandas as pd


# TODO
#
#  - runtime & memory optimization

class preprocessor():

    def __init__(self):
        pass


    def classifyCategories(self, _df, _yLabel, verbose=False):
        # classify the set of categories
        _categories = _df[_yLabel].values
        categoryKeys = set(_categories)
        categoryDict = {}
        n = 0 # number of categories in total dataset
        for cat in categoryKeys:
            categoryDict[cat] = n
            n += 1

        _df = _df.replace({_yLabel: categoryDict}) # https://stackoverflow.com/questions/68487397/replacing-values-of-a-column-which-is-of-type-list
        if verbose: print(_df.head())

        return _df


    def importData(self, _filepath, _labelCol, _yLabel, _colNames=None, _delimiter=",", classify=True, verbose=False):
        """
        This function takes in a filepath for arXiv abstracts, imports, and preprocesses it.
        
        Parameters:
            - _filepath : str = the filepath to load data from
            - _labelCol : str = the column of the data label
            - _colNames : list[str] | None = the columns to label the data with
            - _delimiter : str (default: ",") = the character to separate data with

        Returns:
            - _df : pd.DataFrame = the data imported and preprocessed _yLabel label column
        """

        # load text data
        if verbose: print("Loading data into Pandas...\n")
        _df = pd.read_csv(_filepath, 
                            sep=_delimiter,
                            names=_colNames,
                            low_memory=False)
        if verbose: print("done")
        if verbose: print(_df.head())
        if verbose: print(_df.shape)

        _df[_labelCol] = _df[_labelCol].str.split(expand=False)

        _df[_yLabel] = _df[_labelCol].str[0] # gets the first (top) category for each abstract


        if classify: _df = self.classifyCategories(_df, _yLabel, verbose=verbose)
        return _df


    def getStratifiedSubset(self, _df, _yLabel, _numClasses, numSamples, replaceLabels=True, randomState=1):
        """
        

        Returns:
            - x : ndarray[n,d] = the data of size (num samples, num features)
            - y : ndarray[n,1] = the labels of size (num samples, 1)
        """
        
        categoryCount = collections.Counter(_df[_yLabel])
        topNCategories = categoryCount.most_common(_numClasses)

        dfSmall = pd.DataFrame()
        for key, value in topNCategories:
            __df = _df[_df[_yLabel] == key]
            dfSmall = pd.concat((dfSmall, __df))
        
        dfSmaller = dfSmall.sample(n=numSamples, random_state=randomState) # random state

        if replaceLabels: dfSmaller = self.classifyCategories(dfSmaller, _yLabel, verbose=True)

        return dfSmaller