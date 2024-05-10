import os
import struct
import numpy as np
import pandas as pd

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
