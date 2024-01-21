import numexpr as ne

"""
Implements sigmoid activation function
range: (0,1)
order of continuity: C^inf
"""

def sigmoid(x):
    return ne.evaluate('1 / (1 + exp(-x))')

def d_sigmoid(x):
    sigmoid_x = sigmoid(x)
    return ne.evaluate('sigmoid_x * (1 - sigmoid_x)')
