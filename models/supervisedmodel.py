# Ian Hay - 2023-02-25

import util as util

from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --- abstract class --- ###

class supervisedModel():

    def train(self, x, y):
        """
        Takes in and trains on the data `x` to return desired features `y`.
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def test(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : ndarray[float] : list of datapoints n samples by d features
        Returns:
            - ypred : ndarray[int] : topic prediction of n samples bc y classes
        """
        util.raiseNotDefined()


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

   # models must have:
   # for data with n samples, d features, and c classes
   #  - train(x: ndarray (nxd), y: ndarray (nxc))
   #  - test(x: ndarray (nxd)) result: (ypred : ndarray (nxc))

class FFNN(nn.Module):

    def __init__(self,input_size, output_size, criterion="cel", learningRate = 0.05, hidden_size_1=50, hidden_size_2=50, hidden_size_3=50, epochs=5000):
        super(FFNN,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.l3 = nn.Linear(hidden_size_2,hidden_size_3)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(hidden_size_3,output_size)
        self.epochs = epochs
        self.learningRate = learningRate
        self.initWeights()
        self.criterion = criterion


    def initWeights(self):
        initrange = 0.3
        self.l1.weight.data.uniform_(-initrange, initrange)
        self.l2.weight.data.uniform_(-initrange, initrange)
        self.l3.weight.data.uniform_(-initrange, initrange)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.l1.bias.data.zero_()
        self.l2.bias.data.zero_()
        self.l3.bias.data.zero_()
        self.out.bias.data.zero_()


    def forward(self,x):
        output = self.l1(x) 
        output = self.sigmoid(output)
        output = self.l2(output)
        output = self.sigmoid(output)
        output = self.l3(output)
        output = self.sigmoid(output)
        output = self.out(output)
        return output


    def train(self, x, y):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss()

        # criterion types: "cel", "nll", "he", "kl"
        if (self.criterion == "cel"):
            criterion = nn.CrossEntropyLoss()
        elif (self.criterion == "nll"): # only applies if the final activation is softmax
            criterion = nn.NLLLoss()
        elif (self.criterion == "he"): # this and KLD suck
            criterion = nn.HingeEmbeddingLoss()
        elif (self.criterion == "kl"):
            criterion = nn.KLDivLoss()
        else :
            raise ValueError(f"Invalid criterion for PyTorch Neural Network: {self.criterion}")

        optimizer = torch.optim.SGD(self.parameters(),lr=self.learningRate, momentum=0.9)
        
        costs = []

        n = 0
        while (n < self.epochs):
            #prediction
            y_pred = self(x)
            
            #calculating loss
            cost = criterion(y_pred,y)
        
            #backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if n % (self.epochs/10) == 0:
                print(cost)
                costs.append(cost)
            n += 1
        print(cost)


    def test(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        ypred = self(x)
        return ypred
    

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class EmbeddingNN(nn.Module):

    # TODO

    def __init__(self,input_size, output_size, criterion="cel", learningRate = 0.05, hidden_size_1=50, hidden_size_2=50, hidden_size_3=50, epochs=5000):
        super(FFNN,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.l3 = nn.Linear(hidden_size_2,hidden_size_3)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(hidden_size_3,output_size)
        self.epochs = epochs
        self.learningRate = learningRate
        self.initWeights()
        self.criterion = criterion


    def initWeights(self):
        initrange = 0.3
        self.l1.weight.data.uniform_(-initrange, initrange)
        self.l2.weight.data.uniform_(-initrange, initrange)
        self.l3.weight.data.uniform_(-initrange, initrange)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.l1.bias.data.zero_()
        self.l2.bias.data.zero_()
        self.l3.bias.data.zero_()
        self.out.bias.data.zero_()


    def forward(self,x):
        output = self.l1(x) 
        output = self.sigmoid(output)
        output = self.l2(output)
        output = self.sigmoid(output)
        output = self.l3(output)
        output = self.sigmoid(output)
        output = self.out(output)
        return output


    def train(self, x, y):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss()

        # criterion types: "cel", "nll", "he", "kl"
        if (self.criterion == "cel"):
            criterion = nn.CrossEntropyLoss()
        elif (self.criterion == "nll"): # only applies if the final activation is softmax
            criterion = nn.NLLLoss()
        elif (self.criterion == "he"): # this and KLD suck
            criterion = nn.HingeEmbeddingLoss()
        elif (self.criterion == "kl"):
            criterion = nn.KLDivLoss()
        else :
            raise ValueError(f"Invalid criterion for PyTorch Neural Network: {self.criterion}")

        optimizer = torch.optim.SGD(self.parameters(),lr=self.learningRate, momentum=0.9)
        
        costs = []

        n = 0
        while (n < self.epochs):
            #prediction
            y_pred = self(x)
            
            #calculating loss
            cost = criterion(y_pred,y)
        
            #backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if n % (self.epochs/10) == 0:
                print(cost)
                costs.append(cost)
            n += 1
        print(cost)


    def test(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        ypred = self(x)
        return ypred
    

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class logisticRegression(supervisedModel):

    def __init__(self, maxIter=10000, multiClass="ovr", penalty="l2"):

        params = {
            "penalty": penalty,
            "multi_class": multiClass,
            "max_iter": maxIter,
            "n_jobs": -1,
        }
        self.model = LogisticRegression(**params)


    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)
    


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class NaiveBayes(supervisedModel):

    def __init__(self, alpha=1.0):

        params = {
            "alpha": alpha,
        }
        self.model = CategoricalNB(**params)


    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)



### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class adaBoostDecisionTree(supervisedModel):

    def __init__(self, estimator=DecisionTreeClassifier(max_depth=5), nEstimators=1000, learningRate=0.5):

        params = {
            "estimator": estimator,
            "n_estimators": nEstimators,
            "learning_rate": learningRate,
        }
        self.model = MultiOutputClassifier(ensemble.AdaBoostClassifier(**params))


    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class RandomForestClassifier(supervisedModel):

    def __init__(self, nEstimators=1000, criterion="entropy", maxDepth=25, minSamplesSplit=5, verbose=0):

        params = {
            "n_estimators": nEstimators,
            "criterion": criterion,
            "max_depth": maxDepth,
            "min_samples_split": minSamplesSplit,
            "verbose": verbose,
        }
        self.model = ensemble.RandomForestClassifier(**params)


    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class EM(supervisedModel):

    def __init__(self):
        pass


    def train(self, x, y):
        pass


    def test(self, x):
        pass