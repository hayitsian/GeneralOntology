#!/usr/bin/env python
#
# Ian Hay - 2023-04-07
# https://github.com/hayitsian/General-Index-Visualization

# external dependencies
import sys


# handle verbose commands all in the controller
# handle all other visualizations with the view

class BaseController():

    # controller to take command line input
    def __init__(self, args):
        self.args = args


    # controller to take text file input
    def __init__(self, _filename: str):
         # TODO: do something
        pass


    def importDatafile(self, _filename:str):
        pass

    def buildModel(self, _modeltype:str):
        pass
    
    def loadModel(self, _modelname:str):
        pass


    def trainModel(self):
        pass

    def queryModel(self):
        pass

    def updateModel(self):
        pass

    def saveModel(self):
        pass


    def outputJSON(self):
        pass

    def outputTXT(self):
        pass

    def outputCSV(self):
        pass



    # define the commands this program can take

        # e.g., import data, import datastream, 
        # connect to database, disconnect from database,
        # train a model, load a pretrained model, 
        # query a pretrained model, save a model, 
        # close a model, output to JSON, 
        # output to a CSV, output to a plotter
    
    def run(self):

        # call handleCommandLine to build out appropriate pipeline

        # run the pipeline:
            # connect to the data source
            # send to the model
            # collect output dict
            # pass to the view
            # collect output and pass to the user

        pass