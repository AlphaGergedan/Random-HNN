import numpy as np

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def d_relu(x):
        return np.where(x <= 0, 0, 1)

    @staticmethod
    def leaky_relu(x):
        return np.maximum(0.01 * x, x)

    @staticmethod
    def d_leaky_relu(x):
        return np.where(x <= 0, 0.01, 1)

    @staticmethod
    def parametric_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    @staticmethod
    def d_parametric_relu(x, alpha=0.01):
        return np.where(x <= 0, alpha, 1)

    @staticmethod
    def elu(x, alpha):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def d_elu(x, alpha):
        return np.where(x >= 0, 1, alpha * np.exp(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1 - (np.tanh(x)**2)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return Activations.sigmoid(x) * (1 - Activations.sigmoid(x))

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def d_identity(x):
        return np.ones_like(x)
