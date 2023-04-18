# Ian Hay - 2023-04-07
# https://github.com/hayitsian/General-Index-Visualization

# external dependencies
import sys
import pickle

from model.pipeline.AXpreprocessing import AXimporter, AXpreprocessor
from model.pipeline.GIpreprocessing import GIimporter, GIpreprocessor
from model.pipeline.skLDApipeline import skLDAPipeline
from src.model.pipeline.gnLDApipeline import gnLDAPipeline

# handle verbose commands all in the controller
# handle all other visualizations with the view
class BaseController():

    # controller to take command line input
    def __init__(self, args):
        self.args = args

    """
    # controller to take text file input
    def __init__(self, _filename: str):
         # TODO: do something
        pass
    """

    # define the commands this program can take

        # e.g., import data, import datastream, 
        # connect to database, disconnect from database,
        # train a model, load a pretrained model, 
        # query a pretrained model, save a model, 
        # close a model, output to JSON, 
        # output to a CSV, output to a plotter
    def importDatafile(self, _filename:str, _dataType:str, _preprocessing:bool):
        if (_dataType == "AX"):
            importer = AXimporter()
            if _preprocessing: preprocessor = AXpreprocessor()
        elif (_dataType == "GI"):
            importer = GIimporter()
            if _preprocessing: preprocessor = GIpreprocessor()
        elif (_dataType == "PM"):
            # TODO
            pass
        else: raise ValueError(f"Invalid data type: {_dataType}")

        _df = importer.importData(_filename)
        _x, _y = importer.splitXY(_df)
        if _preprocessing: _x = preprocessor.transform(_x)
        else: _x = importer.transform(_x)
        
        self.X = _x
        self.Y = _y

    def streamDatasource(self, _path:str, _datatype:str, _preprocessing:bool):
        # TODO
        pass


    def buildModel(self, _modeltype:str, _numTopics:int):

        if (_modeltype == "skLDA"):
            self.model = skLDAPipeline(_numTopics)
        elif (_modeltype == "gnLDA"):
            self.model = gnLDAPipeline(_numTopics)
        else: raise ValueError(f"Invalid model type: {_modeltype}")
        self.model.compile()

    def loadModel(self, _filename:str):
        _f = open(_filename)
        self.model = pickle.load(_f)
        _f.close()

    def saveModel(self, _filename:str):
        _f = open(_filename)
        pickle.dump(self.model, _f)
        _f.close()


    def trainModel(self):
        # split into test & train
        self.model.fit(self.X, self.Y)
        # query with test

    def queryModel(self):
        self.model.predict(self.X, self.Y)

    def updateModel(self):
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

        _nTopics = self.args.nTopics

        # optional args
        _verbosity = self.args.verbose
        _save = self.args.save 
        _load = self.args.load
        _preprocessing = self.args.preprocess
        _output = self.args.output

        self.importDatafile(_data, _type, _preprocessing)

        if _action == "train": 
            self.buildModel(_model, _nTopics)
            self.trainModel()
        elif _action == "update":
            self.loadModel(_load)
            self.trainModel()
        elif _action == "query":
            self.loadModel(_load)
            self.queryModel()

        if _save is not None:
            self.saveModel(_save)
        
        if _output == "JSON":
            self.outputJSON()
        elif _output == "CSV":
            self.outputCSV()
        elif _output == "TXT":
            self.outputTXT()