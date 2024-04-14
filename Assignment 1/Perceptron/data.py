import numpy as np

class Data:

    def __init__(self, data_filename: str, label_filename: str):
        self.df = self.load_file(data_filename)
        self.lf = self.load_file(label_filename)


    def load_file(self, filename):
        return np.load(filename)
    

    def get_X(self):
        return self.df[:, [1, 2]]
    

    def get_y(self):
        return self.lf


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