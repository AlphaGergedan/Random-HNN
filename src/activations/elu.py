import numexpr as ne

"""
Implements ELU activation function
range: (-alpha, inf)
order of continuity: C^1 if alpha=1, C^0 otherwise
"""

def elu(x, alpha):
    return ne.evaluate('where(x >= 0, x, alpha * (exp(x) - 1))')

def d_elu(x, alpha):
    return ne.evaluate('where(x >= 0, 1, alpha * exp(x))')
