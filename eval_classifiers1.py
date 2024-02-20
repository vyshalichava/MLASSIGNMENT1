import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from linearDiscriminantAnalysis import LinearDiscriminantAnalysis
from logisticRegression import LogisticRegression

inpp = load_iris()
X = inpp.data
y = inpp.target
xtr, xt, ytr, yt = train_test_split(X, y, test_size=0.3, random_state=42)
lc = LogisticRegression(lr=0.1, max_epochs=100, tol=1e-4)
ld = LinearDiscriminantAnalysis()
lcind = [2, 3]
xtr1 = xtr[:, lcind]
ytr1 = (ytr == 2).astype(int)  
lc.fit(xtr1, ytr1)
ld.fit(xtr1, ytr1)
xtest = xt[:, lcind]
ytest = (yt == 2).astype(int) 
acclc = lc.score(xtest, ytest)
accld = ld.score(xtest, ytest)
print(f"Logistic Regression (Petal Length/Width) - Classification accuracy on testset: {acclc:.2f}")
print(f"Linear Discriminant Analysis (Petal Length/Width) - Classification accuracy on testset: {accld:.2f}")
