"""
Implements the hyperbolic tangent (tanh) activation function
"""

import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - (np.tanh(x)**2)
