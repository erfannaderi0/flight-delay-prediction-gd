# src/gradient_descent.py

import numpy as np
from tqdm import trange

'''
#batch gradient descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
'''
#mini batch gradient descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, batch_size=64):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in trange(self.n_iters, desc="Training"):
            # Shuffle the data before each epoch
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            # Loop over mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.bias
                dw = (-2 / len(X_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
                db = (-2 / len(X_batch)) * np.sum(y_batch - y_pred)

                # Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Optional: compute epoch loss for plotting
            y_full_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y - y_full_pred) ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
