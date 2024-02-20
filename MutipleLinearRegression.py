from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

class MultipleLinearRegression:
    def __init__(self, bchlen=32, regularization=0, max_epochs=100, patience=3):
        self.lc = []
        self.bchlen = bchlen
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.wts = None
        self.bs = None

    def fit(self, X, y, xnew=None, ynew=None, bchlen=None, regularization=None, max_epochs=None, patience=None):
        if bchlen is None:
            bchlen = self.bchlen
        if regularization is None:
            regularization = self.regularization
        if max_epochs is None:
            max_epochs = self.max_epochs
        if patience is None:
            patience = self.patience
        nosmpls, nofts = X.shape
        noout = y.shape[1]
        self.wts = np.zeros((nofts, noout))
        self.bs = np.zeros((1, noout))
        bwts = self.wts
        bbs = self.bs
        bvl = float('inf')
        coin = 0

        for epoch in range(max_epochs):
            ranvals = np.random.permutation(nosmpls)
            ranx = X[ranvals]
            rany = y[ranvals]
            for i in range(0, nosmpls, bchlen):
                xbch = ranx[i:i + bchlen]
                ybch = rany[i:i + bchlen]
                ypd = np.dot(xbch, self.wts) + self.bs
                gwts = -2 * np.dot(xbch.T, (ybch - ypd)) / bchlen
                gbs = -2 * np.sum(ybch - ypd, axis=0, keepdims=True) / bchlen 
                self.wts -= gwts
                self.bs -= gbs
            if xnew is not None and ynew is not None:
                yvpd = np.dot(xnew, self.wts) + self.bs
                vl = np.mean((ynew - yvpd) ** 2)
                if vl < bvl:
                    bvl = vl
                    bwts = self.wts
                    bbs = self.bs
                    coin = 0
                else:
                    coin += 1
                    if coin >= patience:
                        break
                l2_regularization = 0.5 * regularization * np.sum(self.wts ** 2)
                self.lc.append(np.mean((ybch - ypd) ** 2) + l2_regularization)
        self.wts = bwts
        self.bs = bbs
 
    def predict(self, X):
        return np.dot(X, self.wts) + self.bs

    def score(self, X, y):
        ypd = self.predict(X)
        return np.mean((y - ypd) ** 2)

iris = load_iris()
X = iris.data[:, :2]
y = iris.data[:, 2:]
xtr, xx, ytr, yy = train_test_split(X, y, test_size=0.3, random_state=42)
xnew, xt, ynew, yt = train_test_split(xx, yy, test_size=0.33, random_state=42)
s = StandardScaler()
xtrsc = s.fit_transform(xtr)
xvsc = s.transform(xnew)
xtsc = s.transform(xt)
m1 = 'multiple_outputs_model'
model = MultipleLinearRegression(bchlen=32, regularization=0.0, max_epochs=100, patience=3)
model.fit(xtrsc, ytr, xvsc, ynew)
plt.figure()
plt.plot(model.lc)
plt.xlabel('Step')
plt.ylabel('Mean Squared Error')
plt.title(f'{m1} Training Loss')
plt.savefig(f'{m1}_training_loss.png')
mse = model.score(xtsc, yt)
print(f"{m1} - Mean Squared Error on Test Set: {mse:.4f}")
np.savez(f'{m1}_model.npz', wts=model.wts, bs=model.bs)


