# Ian Hay - 2023-04-07
# https://github.com/hayitsian/General-Index-Visualization

# external dependencies
import sys

from model.pipeline.AXpreprocessing import AXimporter, AXpreprocessor
from model.pipeline.GIpreprocessing import GIimporter, GIpreprocessor
from model.pipeline.AXpipeline import AXPipeline
from model.pipeline.GIpipeline import GIPipeline

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


    # define the commands this program can take

        # e.g., import data, import datastream, 
        # connect to database, disconnect from database,
        # train a model, load a pretrained model, 
        # query a pretrained model, save a model, 
        # close a model, output to JSON, 
        # output to a CSV, output to a plotter
    

    def importDatafile(self, _filename:str, _modeltype:str, _preprocessing:bool):
        if (_modeltype == "AX"):
            importer = AXimporter()
            if _preprocessing: preprocessor = AXpreprocessor()
        elif (_modeltype == "GI"):
            importer = GIimporter()
            if _preprocessing: preprocessor = GIpreprocessor()
        else: raise ValueError(f"Invalid model type: {_modeltype}")

        _df = importer.importData(_filename)
        _x, _y = importer.splitXY(_df)
        if _preprocessing: _x = preprocessor.transform(_x)
        else: _x = importer.transform(_x)
        
        self.X = _x
        self.Y = _y

    def buildModel(self, _modeltype:str):

        if (_modeltype == "AX"):
            self.model = AXPipeline()
        elif (_modeltype == "GI"):
            self.model = GIPipeline()
        else: raise ValueError(f"Invalid model type: {_modeltype}")
        self.model.compile()
    
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



    def run(self):

        # run the pipeline:
            # connect to the data source
            # send to the model
            # collect output dict
            # pass to the view
            # collect output and pass to the user

        # required args
        _action = self.args.action
        _type = self.args.type
        _data = self.args.data
        _model = self.args.model

        # optional args
        _verbosity = self.args.verbose
        _save = self.args.save
        _load = self.args.load
        _preprocessing = self.args.preprocess
        _output = self.args.output

        pass