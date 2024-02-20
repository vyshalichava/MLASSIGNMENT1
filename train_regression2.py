from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression  

inp = load_iris()
X = inp.data
y = inp.target
xtr, xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=42)
xnew, xt, ynew, yt = train_test_split(xtmp, ytmp, test_size=0.5, random_state=42)
redmse = StandardScaler()
xtr = redmse.fit_transform(xtr)
xnew = redmse.transform(xnew)
xt = redmse.transform(xt)
m1 = LinearRegression(lr=0.01, regularization=0.0, max_epochs=100, patience=3, batch_size=32)
m1fts = [2, 3]
m1.fit(xtr[:, m1fts], ytr, xnew[:, m1fts], ynew)
m1.save('model2.npz')
test_mse1 = m1.score(xt[:, m1fts], yt)
print(f"Model 2 - Test Mean Squared Error: {test_mse1:.4f}")
