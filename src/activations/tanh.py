import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
import numexpr as ne


"""
Implements the hyperbolic tangent (tanh) activation function
"""


def tanh(x):
    return ne.evaluate('tanh(x)')

def d_tanh(x):
    return ne.evaluate('1 - (tanh(x)**2)')
