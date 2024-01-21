import numexpr as ne

"""
Implements the hyperbolic tangent (tanh) activation function
range: (-1, 1)
order of continuity: C^inf
"""


def tanh(x):
    return ne.evaluate('tanh(x)')

def d_tanh(x):
    return ne.evaluate('1 - (tanh(x)**2)')
