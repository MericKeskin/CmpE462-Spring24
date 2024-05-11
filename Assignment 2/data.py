import os
import struct
import numpy as np
import pandas as pd
from scipy.io import arff

class DecisionTreeData:
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

    def normalize_data(self, X):
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)

        X_standardized = np.where(std_dev != 0, (X - mean) / std_dev, 0)
        return X_standardized

class LogisticRegressionData:
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

def read_svm_data(dataset: str, path: str, labels_to_keep: list):
    if dataset == "training":
        max_samples_per_label = 5000
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        max_samples_per_label = 1000
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(fname_lbl, 'rb') as flbl:
        magic, num_items = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.uint8)

    with open(fname_img, 'rb') as fimg:
        magic, num_items, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(-1, 784)

    mask = np.isin(lbl, labels_to_keep)
    lbl = lbl[mask]
    img = img[mask]

    final_lbl = []
    final_img = []
    
    for label in labels_to_keep:
        mask = lbl == label
        lbls = lbl[mask]
        imgs = img[mask]

        if len(lbls) > max_samples_per_label:
            idx = np.random.choice(len(lbls), max_samples_per_label, replace=False)
            lbls = lbls[idx]
            imgs = imgs[idx]
        final_lbl.append(lbls)
        final_img.append(imgs)
    
    final_lbl = np.concatenate(final_lbl)
    final_img = np.concatenate(final_img)

    return final_lbl, final_img
