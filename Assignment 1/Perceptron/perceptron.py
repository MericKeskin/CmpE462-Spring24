import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, -1)

class Perceptron:

    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.activation_func = unit_step_func


    def train(self, X, y, init: int=None):
        n_samples, n_features = X.shape

        if init:
            self.weights = np.random.rand(n_features) * init
            self.bias = np.random.randint(0, init)
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0
        y_ = self.activation_func(y)

        n_iteration = 0
        while True:
            n_iteration += 1
            done_check = 0
            for ind, x_i in enumerate(X):
                linear_out = np.dot(x_i, self.weights) + self.bias
                y_predict = self.activation_func(linear_out)

                learn = self.learning_rate * (y_[ind] - y_predict)
                if learn == 0:
                    done_check += 1
                self.weights += learn * x_i
                self.bias += learn
            if done_check == n_samples:
                break
        return n_iteration


    def predict(self, X):
        linear_out = np.dot(X, self.weights) + self.bias
        y_predict = self.activation_func(linear_out)

        return y_predict
    