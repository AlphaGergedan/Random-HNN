#import numexpr as ne
import numpy as np

"""
Implements the hyperbolic tangent (tanh) activation function
range: (-1, 1)
order of continuity: C^inf
"""


def tanh(x):
    return np.tanh(x)
    # return ne.evaluate('tanh(x)')

def d_tanh(x):
    return 1 - np.tanh(x)**2
    return ne.evaluate('1 - (tanh(x)**2)')
