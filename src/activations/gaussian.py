"""
Implements Gaussian and its derivative
range: (0, 1]
order of continuity: C^inf
"""

import numpy as np

def gaussian(x):
    return np.exp(-x**2)

def d_gaussian(x):
    return -2 * x * gaussian(x)
