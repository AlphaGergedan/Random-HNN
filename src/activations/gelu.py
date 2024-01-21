import numpy as np

"""
Implements Gaussian Error Linear Unit
range: (-0.17..., inf)
order of continuity: C^inf
"""

def erf(x):
    pass
    # error function TODO
    # return scipy.special.erf(x / np.sqrt(2)) ??

def phi(x):
    pass
    # return 0.5 * x * (1 + erf(x/(np.sqrt(2))))

def gelu(x):
    # TODO
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def d_gelu(x):
    return x * (1 - gelu(x))
