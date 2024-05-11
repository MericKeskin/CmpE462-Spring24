import numpy as np

class KMeans:

    def __init__(self, K):
        self.K = K

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_inds = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[ind] for ind in random_inds]

        while True:
            pass
