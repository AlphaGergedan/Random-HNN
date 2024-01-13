import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
import numexpr as ne

"""
Implements ELU activation function
"""

def elu(x, alpha):
    return ne.evaluate('where(x >= 0, x, alpha * (exp(x) - 1))')

def d_elu(x, alpha):
    return ne.evaluate('where(x >= 0, 1, alpha * exp(x))')
