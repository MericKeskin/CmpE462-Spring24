import numpy as np
import pandas as pd
from scipy.io import arff

class Data:
    def __init__(self, path: str):
        self.path = path
        self.data, self.meta = self.load_data()
        
    def load_data(self):
        if self.path.endswith('.arff'):
            data, meta = arff.loadarff(self.path)
        elif self.path.endswith('.data'):
            column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
            data = pd.read_csv(self.path, header=None, names=column_names)
            data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': -1})
            meta = None
        else:
            raise ValueError("Unsupported file type")
        return data, meta
    
    def get_X(self):
        if self.meta:
            arrays = [self.data[field].astype(np.float64) for field in self.meta.names()[:-1]]
            X = np.column_stack(arrays)
        else:
            X = self.data.iloc[:, 2:].values.astype(np.float64)
        return X
    
    def get_y(self):
        if self.meta:
            return np.where(self.data[self.meta.names()[-1]] == b'M', 1, -1)
        else:
            return self.data['Diagnosis'].values

    def normalize_data(self, X):
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)

        X_standardized = np.where(std_dev != 0, (X - mean) / std_dev, 0)
        return X_standardized
    
    def scale_data(self, X):
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        scale = np.where(X_max - X_min != 0, X_max - X_min, 1)
        X_scaled = (X - X_min) / scale
        return X_scaled

    
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