"""
Implements identity activation function
range: (-inf, inf)
order of continuity: C^inf
"""

import numpy as np

def identity(x):
    return x

def d_identity(x):
    return np.ones_like(x)
