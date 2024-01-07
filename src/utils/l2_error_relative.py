import numpy as np

def l2_error_relative(y, y_hat):
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    assert y.shape == y_hat.shape

    return np.linalg.norm(y_hat-y, ord=2) / np.linalg.norm(y, ord=2)
