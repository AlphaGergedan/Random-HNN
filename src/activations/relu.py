"""
Implements ReLU activation functions:
    - relu
      range: [0, inf)
      order of continuity: C^0
    - leaky_relu
    - parametric_relu

"""

import numpy as np

def relu(x):
        return np.maximum(x, 0)

def d_relu(x):
    return np.where(x <= 0, 0, 1)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def d_leaky_relu(x):
    return np.where(x <= 0, 0.01, 1)

def parametric_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def d_parametric_relu(x, alpha=0.01):
    return np.where(x <= 0, alpha, 1)
