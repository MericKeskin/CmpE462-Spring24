import numpy as np

class NaiveBayes:
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.p_Y = {}
        
    def train(self, X, y):
        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.p_Y[c] = X_c.shape[0] / X.shape[0]

    def p_x_given_y(self, x, y):
        mean = self.means[y]
        var = self.variances[y]
        probability = self.gaussian_normal(x, mean, var)
        return probability
    
    def gaussian_normal(self, x, mean, var):
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.p_Y:
                p_y = self.p_Y[c]
                p_x_given_y = np.prod(self.p_x_given_y(x, c), axis=0)
                p_y_given_x = p_x_given_y * p_y 
                posteriors.append(p_y_given_x)
            
            predictions.append(np.argmax(posteriors))
        
        return np.array(predictions)