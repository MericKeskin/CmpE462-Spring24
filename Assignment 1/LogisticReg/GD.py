import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def loss(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return np.mean(np.log(1 + np.exp(-y * (x @ w))))

def g(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return -np.mean((y[:, np.newaxis] * x) / (1 + np.exp(y * (x @ w))[:, np.newaxis]), axis=0)

def l2(w: np.ndarray, _lambda: float) -> float:
    return _lambda * (w @ w)

def l2_g(w: np.ndarray, _lambda: float) -> np.ndarray:
    return 2 * _lambda * w

def init_w(n: int) -> np.ndarray:
    return np.ones(n)

def predict(X, weights) -> np.ndarray:
    probabilities = sigmoid(X @ weights)
    predictions = np.where(probabilities >= 0.5, 1, -1)
    return predictions
