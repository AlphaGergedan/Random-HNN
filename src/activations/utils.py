# define numexpr threads to spawn
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16' if not 'NUMEXPR_MAX_THREADS' in os.environ else os.environ['NUMEXPR_MAX_THREADS']
os.environ['NUMEXPR_NUM_THREADS'] = '16' if not 'NUMEXPR_NUM_THREADS' in os.environ else os.environ['NUMEXPR_NUM_THREADS']

from . import relu, tanh, sigmoid, elu, identity, gaussian, gelu, silu, softplus

def parse_activation(activation):
    """
    Given activation function string, returns the activation function and its derivative

    @param activation str from list [ 'relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid', 'gaussian', 'gelu', 'silu', 'softplus' ]
    """
    match activation:
        case "relu":
            return relu.relu, relu.d_relu
        case "leaky_relu":
            return relu.leaky_relu, relu.d_leaky_relu
        case "elu":
            return elu.elu, elu.d_elu
        case "tanh":
            return tanh.tanh, tanh.d_tanh
        case "sigmoid":
            return sigmoid.sigmoid, sigmoid.d_sigmoid
        case "gaussian":
            return gaussian.gaussian, gaussian.d_gaussian
        case "gelu":
            return gelu.gelu, gelu.d_gelu
        case "silu":
            return silu.silu, silu.d_silu
        case "softplus":
            return softplus.softplus, softplus.d_softplus
        case _:
            # default to identity
            return identity.identity, identity.d_identity
