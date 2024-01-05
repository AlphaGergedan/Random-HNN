"""
Implements identity activation function
"""

import numpy as np

def identity(x):
    return x

def d_identity(x):
    return np.ones_like(x)
