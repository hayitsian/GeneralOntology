# Ian Hay - 2023-02-28

import supervisedmodel
import itertools
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import util
from timeit import default_timer

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class modelOptimizer():

    def __init__(self):
        pass

    def runNN(self, NN, xtrain, ytrain, xtest, ytest, verbose=False):
        numOutput = len(set(ytrain))

        yTrainTrue = np.zeros((ytrain.size, numOutput)) # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
        yTrainTrue[np.arange(ytrain.size), ytrain] = 1

        yTestTrue = np.zeros((ytest.size, numOutput))
        yTestTrue[np.arange(ytest.size), ytest] = 1

        NN.train(xtrain, yTrainTrue, verbose=verbose)#.to(device)
        ypred = NN.test(xtest)#.to(device)

        f1, roc, acc, recall, precision = util.multi_label_metrics(ypred, yTestTrue).values()

        return f1, roc, acc, recall, precision

    def optimizeNN(self, x, y, numHiddenFirst, numHiddenSecond, numHiddenThird, epochs, learningRate, k=5, verbose=False):
        """
        
        Parameters:
            - numHiddenFirst : list[int] = options for number of hidden neurons in first layer
            - numHiddenSecond : list[int] = options for number of hidden neurons in second layer
            - numHiddenThird : list[int] = options for number of hidden neurons in third layer        
        Returns:
            - metrics : pd.DataFrame = data of classification metrics for each model
        
        """
        skfold = StratifiedKFold(n_splits=k)
        posibilitiesList = [numHiddenFirst, numHiddenSecond, numHiddenThird]
        layerPosibilities = list(itertools.product(*posibilitiesList))

        numSamples = x.shape[0]
        numInput = x.shape[1]
        numOutput = len(set(y))

        metrics = pd.DataFrame()

        for numFirst, numSecond, numThird in layerPosibilities:

            f1List = []
            rocList = []
            accList = []
            recList = []
            precList = []
            timeList = []

            metricDict = {}

            if (verbose):
                print(f"\nTraining FFNN with CEL loss on {numSamples} abstracts across {numOutput} topics:\n"
                + f"{epochs} epochs, {learningRate} learning rate, {numFirst} + {numSecond} + {numThird} hidden neurons...\n")
            

            for train, test in skfold.split(x, y):
                ffNN = supervisedmodel.FFNN(numInput, numOutput, hidden_size_1=numFirst, hidden_size_2=numSecond, hidden_size_3=numThird, epochs=epochs).to(device)
                xTrain = x[train]
                xTest = x[test]
                yTrain = y[train]
                yTest = y[test]

                start = default_timer()
                f1, roc, acc, recall, precision = self.runNN(ffNN, xTrain, yTrain, xTest, yTest)
                _time = default_timer() - start

                timeList.append(_time)
                f1List.append(f1)
                rocList.append(roc)
                accList.append(acc)
                recList.append(recall)
                precList.append(precision)

                if (verbose):
                    print(f"F1: {f1:.3f}")
                    print(f"AUC: {roc:0.3f}")
                    print(f"Accuracy: {acc:.3f}")
                    print(f"Recall: {recall:.3f}")
                    print(f"Precision: {precision:0.3f}")
                    print(f"Time: {_time:.3f}\n")

            if (verbose):
                print(f"\nTrained FFNN with CEL loss with {numSamples} abstracts across {numOutput} topics:\n"
                + f"{epochs} epochs, {learningRate} learning rate, {numFirst} + {numSecond} + {numThird} hidden neurons.\n"
                + f"Training took {np.sum(timeList):0.3f} seconds\n"
                + f"Average Metrics: \nF1 = {np.mean(f1List):0.3f}  \nROC AUC = {np.mean(rocList):0.3f}  \nAccuracy = {np.mean(accList):0.3f}  \nRecall = {np.mean(recList):.3f}  \nPrecision = {np.mean(precList):.3f}\n")


            metricDict["num hidden 1"] = [numFirst]
            metricDict["num hidden 2"] = [numSecond]
            metricDict["num hidden 3"] = [numThird]
            metricDict["f1"] = np.mean(f1List)
            metricDict["auc"] = np.mean(rocList)
            metricDict["acc"] = np.mean(accList)
            metricDict["recall"] = np.mean(recList)
            metricDict["precision"] = np.mean(precList)
            metricDict["time"] = np.mean(timeList)

            metrics = pd.concat([metrics, pd.DataFrame(metricDict)], ignore_index=True)
        return metrics