import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, iterations=1000) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.activation_func = unit_step_func

    def train(self, X, y):
        pass

    def predict(self, X):
        pass