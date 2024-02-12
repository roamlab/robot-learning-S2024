#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class MyDNN(nn.Module):
    def __init__(self, input_dim):
        super(MyDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
	
    def predict(self, features):
        """ 
        Function receives a numpy array, converts to torch, returns numpy again
        """
        self.eval()	#Sets network in eval mode (vs training mode)
        features = torch.from_numpy(features).float()
        return self.forward(features).detach().numpy()

class MyDataset(Dataset):
    def __init__(self, labels, features):
        super(MyDataset, self).__init__()
        self.labels = labels
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):		#This tells torch how to extract a single datapoint from a dataset, Torch randomized and needs a way to get the nth-datapoint
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}
    
class MyDNNTrain(object):
    def __init__(self, network):	#Networks is of datatype MyDNN
        self.network = network
        self.learning_rate = .01
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.num_epochs = 500
        self.batchsize = 100
        self.shuffle = True

    def train(self, labels, features):
        self.network.train()
        dataset = MyDataset(labels, features)
        loader = DataLoader(dataset, shuffle=self.shuffle, batch_size = self.batchsize)
        for epoch in range(self.num_epochs):
            self.train_epoch(loader)

    def train_epoch(self, loader):
        total_loss = 0.0
        for i, data in enumerate(loader):
            features = data['feature'].float()
            labels = data['label'].float()
            self.optimizer.zero_grad()
            predictions = self.network(features)
            loss = self.criterion(predictions, labels)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        print('loss', total_loss/i)

def function(x1, x2):
    return math.cos(x1) * math.cos(x2)

def reject(x1, x2):
    if (x1<0 and x2 > 0): return True

    # s1 = 0
    # s2 = 0
    # dist = 1
    # if ( math.sqrt(np.dot([s1-x1,s2-x2],[s1-x1,s2-x2])) < dist): return True
    return False

def plot_prediction(network):
    X = np.arange(-math.pi, math.pi, 0.1)
    Y = np.arange(-math.pi, math.pi, 0.1)
    X, Y = np.meshgrid(X, Y)

    features = np.concatenate((X.reshape(-1, 1), Y.reshape(-1,1)), axis=1)
    predictions = network.predict(features).reshape(63, 63)

    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    surf = ax1.plot_surface(X, Y, predictions)

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf = ax2.plot_surface(X, Y, np.vectorize(function)(X,Y))

    plt.show()

def main():
    network = MyDNN(2)
    trainer = MyDNNTrain(network)
    features = []
    while len(features) < 1000:
        x1 = np.random.uniform(-math.pi, math.pi)
        x2 = np.random.uniform(-math.pi, math.pi)
        if reject(x1,x2): continue
        features.append([x1, x2])
    features = np.asarray(features)
    labels = np.vectorize(function)(features[:,0], features[:,1]).reshape(-1,1)
    trainer.train(labels, features)
    plot_prediction(network)

if __name__ == '__main__':
    main()
