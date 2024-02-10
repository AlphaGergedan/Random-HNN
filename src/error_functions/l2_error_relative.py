import numpy as np

def l2_error_relative(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    assert y_true.shape == y_pred.shape

    return np.linalg.norm(y_pred - y_true, ord=2) / np.linalg.norm(y_true, ord=2)
