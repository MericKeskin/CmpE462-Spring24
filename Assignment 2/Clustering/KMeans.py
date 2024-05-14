import numpy as np

def eucledian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


def cosine_similarity_distance(x1, x2):
    cosine_similarity = np.dot(x1, x2) / np.linalg.norm(x1) * np.linalg.norm(x2)

    return 1 - cosine_similarity


class KMeans:

    def __init__(self, K, converge_func):
        self.K = K
        self.converge_func = converge_func

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_inds = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[ind] for ind in random_inds]

        while True:
            self.clusters = self._form_clusters(self.centroids)

            old_centroids = self.centroids
            self.centroids = self._form_centroids(self.clusters)

            if self._is_converged(old_centroids, self.centroids):
                break

        
    def _form_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]

        for ind, sample in enumerate(self.X):
            distances = [self.converge_func(sample, point) for point in centroids]
            centroid_ind = np.argmin(distances)
            clusters[centroid_ind].append(ind)

        return clusters

    
    def _form_centroids(self, clusters):
        centroids = np.zeros(self.K, self.n_features)

        for cluster_ind, cluster in enumerate(clusters):
            new_centroid = np.mean(self.X[cluster], axis=0)
            centroids[cluster_ind] = new_centroid
        
        return centroids


    def _is_converged(self, old_centroids, new_centroids):
        distances = [self.converge_func(old_centroids[ind], new_centroids[ind]) for ind in range(self.K)]

        return sum(distances) == 0
