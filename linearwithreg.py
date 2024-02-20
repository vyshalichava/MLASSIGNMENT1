import numpy as np
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression, X_temp, y_temp

xnew, xt, ynew, yt = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
class RegularizedLinearRegression(LinearRegression):
    def __init__(self, regularization=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularization = regularization
    def fit(self, X, y, xnew=None, ynew=None):
        super().fit(X, y, xnew, ynew)
        reg_loss = 0.5 * self.regularization * np.sum(self.wts ** 2)
        self.lc = [loss + reg_loss for loss in self.lc]
newmod = LinearRegression(lr=0.01, regularization=0.0, max_epochs=100, patience=3, batch_size=32)
newmod.load('model4.npz')
redmod = RegularizedLinearRegression(regularization=0.01, lr=0.01, max_epochs=100, patience=3, batch_size=32)
redmod.wts = np.copy(newmod.wts)
redmod.bs = newmod.bs
mse = redmod.score(xnew[:, [0, 1, 2, 3]], ynew)
print(f"Model 4 - Test Mean Squared error with L2 regularization: {mse:.4f}")