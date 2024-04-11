import numpy as np
import pandas as pd


class Data:
    def __init__(self, path: str):
        self.df = pd.read_csv(path, header=None)
        
    def get_X(self):
        return self.df.drop([0, 1], axis=1).values

    def get_y(self):
        self.df[1] = self.df[1].map({'M': 1, 'B': 0})
        return self.df[1].values
    
    def split_data(self, X, y, test_size=0.1):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        num_train = int(X.shape[0] * (1 - test_size))
        
        X_train = X[:num_train]
        X_test = X[num_train:]
        y_train = y[:num_train]
        y_test = y[num_train:]
        
        return X_train, X_test, y_train, y_test