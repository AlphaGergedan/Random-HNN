import numexpr as ne

"""
Implements Softplus activation function
range: (0, inf)
order of continuity: C^inf
"""

def softplus(x):
    return ne.evaluate('log(1 + exp(x))')

def d_softplus(x):
    return ne.evaluate('1 / (1 + exp(-x))')
