import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
import numexpr as ne

"""
Implements sigmoid activation function
"""

def sigmoid(x):
    return ne.evaluate('1 / (1 + exp(-x))')

def d_sigmoid(x):
    sigmoid_x = sigmoid(x)
    return ne.evaluate('sigmoid_x * (1 - sigmoid_x)')
