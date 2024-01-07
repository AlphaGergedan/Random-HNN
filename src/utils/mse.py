import numpy as np

def MSE(y, y_hat):
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    assert y.shape == y_hat.shape

    return np.sum((y - y_hat)**2) / y.shape[0]
