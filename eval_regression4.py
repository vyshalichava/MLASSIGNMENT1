from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

inp = load_iris()
X = inp.data
y = inp.target
xtr, xx, ytr, yy = train_test_split(X, y, test_size=0.3, random_state=42)
xnew, xtest, ynew, ytest = train_test_split(xx, yy, test_size=0.5, random_state=42)
redmse = StandardScaler()
xtr = redmse.fit_transform(xtr)
xnew = redmse.transform(xnew)
xtest = redmse.transform(xtest)
model = LinearRegression(lr=0.01, regularization=0.0, max_epochs=100, patience=3, batch_size=32)
model.load('model4.npz')
m4fts = [0, 1, 2, 3]
finmse = model.score(xtest[:, m4fts], ytest)
print()
print(f"Model 4 - Mean Squared Error: {finmse:.4f}")
