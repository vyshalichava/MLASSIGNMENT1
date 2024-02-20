#LDA 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.pi = None 
        self.m = None 
        self.mat1 = None 
        self.cls = None 
        self.s = None 
        self.nocls = None  

    def fit(self, X, y):
        self.cls = np.unique(y)
        self.nocls = len(self.cls)
        self.s = X.shape[1]
        self.pi = np.zeros(self.nocls)
        self.m = np.zeros((self.nocls, self.s))
        self.mat1 = np.zeros((self.s, self.s))
        tot = len(y)
        for i, c in enumerate(self.cls):
            X_c = X[y == c]
            n_c = len(X_c)
            self.pi[i] = n_c / tot
            self.m[i] = np.mean(X_c, axis=0)
            self.mat1 += np.cov(X_c, rowvar=False) * (n_c - 1)
        self.mat1 /= (tot - self.nocls)

    def predict(self, X):
        l1 = []
        for x in X:
            l2 = []
            for i in range(self.nocls):
                mean_diff = x - self.m[i]
                var1 = -0.5 * mean_diff.dot(np.linalg.solve(self.mat1, mean_diff))
                l2.append(var1)
            predd = self.cls[np.argmax(l2)]
            l1.append(predd)
        return np.array(l1)

    def score(self, X, y):
        l1 = self.predict(X)
        acc = np.mean(l1 == y)
        return acc

inpp = load_iris()
X = inpp.data 
y = inpp.target
xtr, xt, ytr, yt = train_test_split(X, y, test_size=0.3, random_state=42)
mm = LinearDiscriminantAnalysis()
mm.fit(xtr, ytr)
acc = mm.score(xt, yt)
print(f"Linear Discriminant Analysis - Classification acc on Test Set: {acc:.2f}")
