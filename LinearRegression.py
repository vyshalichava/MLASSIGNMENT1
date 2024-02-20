from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, lr=0.01, regularization=0.0, max_epochs=1000, patience=10, batch_size=32):
        self.lr = lr
        self.lc = [] 
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.wts = None
        self.bs = None
        self.lslist = []

    def fit(self, X, y, xnew=None, ynew=None):
        nosmpls, nfts = X.shape
        self.wts = np.zeros(nfts)
        self.bs = 0
        bwts = np.copy(self.wts)
        bbs = self.bs
        bvl = float('inf')
        c = 0

        for epoch in range(self.max_epochs):
            indices = np.arange(nosmpls)
            np.random.shuffle(indices)
            ranx = X[indices]
            rany = y[indices]
            for i in range(0, nosmpls, self.batch_size):
                xbch = ranx[i:i + self.batch_size]
                ybch = rany[i:i + self.batch_size]
                ypd = np.dot(xbch, self.wts) + self.bs
                mseloss = mean_squared_error(ybch, ypd)
                regloss = 0.5 * self.regularization * np.sum(self.wts ** 2)
                totloss = mseloss + regloss
                self.lslist.append(totloss)
                self.lc.append(np.mean((ybch - ypd) ** 2) + regloss)
                gdwts = -2 * np.dot(xbch.T, (ybch - ypd)) / self.batch_size + self.regularization * self.wts
                gdbs = -2 * np.sum(ybch - ypd) / self.batch_size
                self.wts -= self.lr * gdwts
                self.bs -= self.lr * gdbs
            if xnew is not None and ynew is not None:
                vp = np.dot(xnew, self.wts) + self.bs
                vl = mean_squared_error(ynew, vp)
                if vl < bvl:
                    bwts = np.copy(self.wts)
                    bbs = self.bs
                    bvl = vl
                    c = 0
                else:
                    c += 1
                if c >= self.patience:
                    break
        self.wts = bwts
        self.bs = bbs

    def predict(self, X):
        return np.dot(X, self.wts) + self.bs

    def score(self, X, y):
        ypd = self.predict(X)
        return mean_squared_error(y, ypd)

    def save(self, filename):
        np.savez(filename, wts=self.wts, bs=self.bs)

    def load(self, filename):
        data = np.load(filename)
        self.wts = data['wts']
        self.bs = data['bs']
