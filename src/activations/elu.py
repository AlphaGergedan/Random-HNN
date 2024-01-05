"""
Implements ELU activation function
"""

import numpy as np

def elu(x, alpha):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def d_elu(x, alpha):
    return np.where(x >= 0, 1, alpha * np.exp(x))
