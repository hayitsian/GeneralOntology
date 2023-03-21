# Ian Hay - 2023-03-14

import util as util
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

   # models must have:
   # for data with n samples, d features, and c classes
   #  - train(x: ndarray (nxd), y: ndarray (nxc))
   #  - test(x: ndarray (nxd)) result: (ypred : ndarray (nxc))

class FFNN(nn.Module):

    def __init__(self, input_size, output_size, criterion="cel", learningRate = 0.05, hidden_size_1=50, hidden_size_2=50, hidden_size_3=50, epochs=5000):
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


    def forward(self, x):
        output = self.l1(x) 
        output = self.sigmoid(output)
        output = self.l2(output)
        output = self.sigmoid(output)
        output = self.l3(output)
        output = self.sigmoid(output)
        output = self.out(output)
        return self.sigmoid(output)


    def train(self, x, y, verbose=False):
        numOutput = len(set(y))

        y_true = np.zeros((y.size, numOutput)) # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
        y_true[np.arange(y.size), y] = 1
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y_true = torch.tensor(y_true, dtype=torch.float32).to(device)

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
            cost = criterion(y_pred,y_true)
        
            #backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if n % (self.epochs/100) == 0 and verbose == True:
                print(cost)
                costs.append(cost)
            if n % (self.epochs/10) == 0 and verbose == True:
                f1, roc, acc, recall, precision = util.multi_label_metrics(y_pred, y_true.cpu().data.numpy()).values()
                print(f"Metrics: \nF1 = {f1:0.3f}  \nROC AUC = {roc:0.3f}  \nAccuracy = {acc:0.3f}  \nRecall = {recall:.3f}  \nPrecision = {precision:.3f}\n")
            n += 1
        print(cost)
        return f1, roc, acc, recall, precision


    def test(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        ypred = self(x)
        return ypred


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class EmbeddingNN(nn.Module):

    def __init__(self, encode_dims=[2000, 1024, 512, 20], decode_dims=[20, 1024, 2000], dropout=0.0, nonlin='relu', epochs=5000):
        super(EmbeddingNN, self).__init__()
        # https://github.com/zll17/Neural_Topic_Models/blob/6d8f0ce750393de35d3e0b03eae43ba39968bede/models/wae.py#L1
        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-1)
        })

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[nonlin]
        self.z_dim = encode_dims[-1]


    def encode(self, x):
        hid = x
        for i, (_,layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder)-1:
                hid = self.nonlin(hid)
        return hid

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder)-1:
                hid = self.nonlin(self.dropout(hid))
        return hid

    def forward(self, x):
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        x_reconst = self.decode(theta)
        return x_reconst, theta


    def train(self, x):
        x = torch.tensor(x, dtype=torch.str).to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(self.parameters(),lr=self.learningRate, momentum=0.9)
        
        costs = []

        n = 0
        while (n < self.epochs):
            #prediction
            y_pred, theta = self(x)
            
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
        ypred, theta = self(x)
        return theta
    