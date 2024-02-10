import numpy as np

def l2_error(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    assert y_true.shape == y_pred.shape

    return np.sqrt(np.sum((y_true - y_pred)**2))
