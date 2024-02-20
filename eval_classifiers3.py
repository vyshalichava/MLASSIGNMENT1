import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logisticRegression import LogisticRegression
from linearDiscriminantAnalysis import LinearDiscriminantAnalysis

iris = load_iris()
X = iris.data 
y = iris.target 
xtr, xt, ytr, yt = train_test_split(X, y, test_size=0.3, random_state=42)
lc = LogisticRegression()
ld = LinearDiscriminantAnalysis()
lc.fit(xtr, ytr)
ld.fit(xtr, ytr)
acclc = lc.score(xt, yt)
accld = ld.score(xt, yt)
print(f"Logistic Regression (Variant 3 - All Features) - Classification accuracy on testset: {acclc:.2f}")
print(f"Linear Discriminant Analysis (Variant 3 - All Features) - Classification accuracy on testset: {accld:.2f}")
