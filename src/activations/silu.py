import numpy as np

"""
Implements Sigmoid linear unit activation function and its derivative

range: [-0.278..., inf)
order of continuity: C^inf
"""

def silu(x):
    return x / (1 + np.exp(-x))

def d_silu(x):
    return (1 + np.exp(-x) + x * np.exp(-x)) / ( (1 + np.exp(-x))**2 )
