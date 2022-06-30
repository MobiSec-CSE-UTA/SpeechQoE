from torch.nn import Module
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from numpy import vstack
from sklearn import preprocessing


class EmodelDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path)
        # store the inputs and outputs
        self.X = df.iloc[:, 2:4].values.astype(np.float32)
        self.y = df.iloc[:, 1:2].values.astype(np.float32)

        # normalize
        min_max_scaler = preprocessing.MinMaxScaler()
        self.X_scaled = min_max_scaler.fit_transform(self.X)
        self.X = self.X_scaled

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.layer1 = Linear(n_inputs, 200)
        self.act1 = LeakyReLU()
        # second hidden layer
        self.layer2 = Linear(200, 100)
        self.act2 = LeakyReLU()
        # third hidden layer and output
        self.layer3 = Linear(100, 1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.layer1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.layer2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.layer3(X)
        return X


def preds(dl, model):
    predictions = list()
    for batch_x, batch_y in dl:
        yhat = model(batch_x)
        yhat = yhat.detach().numpy()  # retrieve numpy array

        predictions.append(yhat)

    predictions = vstack(predictions)
    return predictions
