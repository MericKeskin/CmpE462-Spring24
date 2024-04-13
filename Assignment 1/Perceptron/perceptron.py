import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.activation_func = unit_step_func


    def train(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = unit_step_func(y)

        for _ in range(self.n_iterations):
            for ind, x_i in enumerate(X):
                linear_out = np.dot(x_i, self.weights) + self.bias
                y_predict = self.activation_func(linear_out)

                learn = self.learning_rate * (y_[ind] - y_predict)
                self.weights += learn * x_i
                self.bias += learn


    def predict(self, X):
        linear_out = np.dot(X, self.weights) + self.bias
        y_predict = self.activation_func(linear_out)

        return y_predict
    