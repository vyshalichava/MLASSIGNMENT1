#2.1 Logistic Regression
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, lr=0.01, max_epochs=100, tol=1e-4):
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.wts = None
        self.bs = None
        self.lc = [] 
        self.meth = None

    def smd(self, z): 
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        nospls, nofts = X.shape
        if nospls < nofts and nospls < 1000: 
            self.meth = 'normal_equations'
        else:
            self.meth = 'gradient_descent'
        self.wts = np.zeros(nofts)
        self.bs = 0
        if self.meth == 'gradient_descent':
            for epoch in range(self.max_epochs):
                m1 = np.dot(X, self.wts) + self.bs
                ypd = self.smd(m1)
                gdwts = np.dot(X.T, (ypd - y)) / nospls
                gdbs = np.sum(ypd - y) / nospls
                self.wts -= self.lr * gdwts
                self.bs -= self.lr * gdbs
                ls = -np.mean(y * np.log(ypd) + (1 - y) * np.log(1 - ypd))
                self.lc.append(ls)
                if epoch > 0 and abs(ls - self.lc[epoch - 1]) < self.tol:
                    break
        else:
            X_b = np.c_[np.ones((nospls, 1)), X] 
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bs = theta[0]
            self.wts = theta[1:]

    def predict(self, X):
        m1 = np.dot(X, self.wts) + self.bs
        ranvals = self.smd(m1)
        ppds = (ranvals >= 0.5).astype(int)
        return ppds

    def score(self, X, y):
        ppds = self.predict(X)
        accuracy = np.mean(ppds == y)
        return accuracy

inpp = load_iris()
X = inpp.data  
y = inpp.target 
xtr, xt, yrt, yt = train_test_split(X, y, test_size=0.3, random_state=42)
# For the first classifier
logcls = LogisticRegression(lr=0.1, max_epochs=100, tol=1e-4)
inds = [2, 3] 
xtrpet = xtr[:, inds]
ytrpet = (yrt == 2).astype(int) 
logcls.fit(xtrpet, ytrpet)
xtpet = xt[:, inds]
ytpet = (yt == 2).astype(int)
accpet = logcls.score(xtpet, ytpet)
print(f"Logistic regression Petal - Classification accuracy on testset: {accpet:.2f}")
