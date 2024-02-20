import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class clswrp:
    def __init__(self, classifier):
        self.clf = classifier
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self, X):
        return self.clf.predict(X)
    def score(self, X, y):
        return self.clf.score(X, y)

inp = load_iris()
X = inp.data  
y = inp.target
xtr, xt, ytr, yt = train_test_split(X, y, test_size=0.3, random_state=42)
inpp = {
    'petal': [2, 3],
    'sepal': [0, 1], 
    'all': [0, 1, 2, 3]
}
classifiers = {
    'Logistic Regression': clswrp(LogisticRegression()),
    'Decision Tree': clswrp(DecisionTreeClassifier()),
    'SVM': clswrp(SVC(kernel='linear'))
}
for i, j in inpp.items():
    plt.figure(figsize=(12, 4))
    acclist = {}
    for i, clf_wrapper in classifiers.items():
        clf_wrapper.fit(xtr[:, j], ytr)
        acc = clf_wrapper.score(xt[:, j], yt)
        acclist[i] = acc
        print(f"Accuracy with {i} features using {i}: {acc:.2f}")
    if all(acc == 1.00 for acc in acclist.values()):
        continue
    for idx, (i, clf_wrapper) in enumerate(classifiers.items(), 1):
        plt.subplot(1, 3, idx)
        if acclist[i] < 1.00:
            plot_decision_regions(xt[:, j], yt, clf=clf_wrapper.clf)
            plt.xlabel(inp.feature_names[j[0]])
            plt.ylabel(inp.feature_names[j[1]])
            plt.title(i)
    plt.tight_layout()
    plt.show()
