import numpy as np

class HarmonicOscillator():
    """
    1 dof and undamped
    """

    def __init__(self, k=1):
        """
        example taken from: hirsch differential equations, dynamical systems & introduction to chaos page 208
        constant: k > 0
        """
        assert k > 0
        self.k = k

    # Hamiltonian Formulations

    def H(self, input_x):
        """
        Implementation of the Hamiltonian formulation of undamped harmonic oscillator
        H(x,y) = y^2/2 + kx^2/2
        x_dot = y
        y_dot = -kx
        """
        # x is an array of inputs [x_1,x_2,...] where x_i = (x,y) \in R^{2n} where n is number of degrees of freedom
        # output is array of values [y_1, y_2, ...] where y_i is a real number (1-dimensional)
        (x,y) = input_x[:,0], input_x[:,1]
        return y**2 / 2 + self.k * (x**2) / 2
    def dH(self, input_x):
        """
        Implementaion of the gradient of the Hamiltonian
        dH(x,y) = [kx, y]
        """
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dx, dH/dy)
        (x,y) = input_x[:,0], input_x[:,1]
        dx = self.k * x
        dy = y
        return np.array([dx, dy]).T
