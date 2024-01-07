import numpy as np

class Example2dHamiltonian():
    """
    example taken from: hirsch differential equations, dynamical systems & introduction to chaos page 209
    """
    # Hamiltonian Formulations
    def H(self, input_x):
        """
        Implementation of the Hamiltonian formulation of an example 2d Hamiltonian
        H(x,y) = x^4/4 - x^2/2 + y^2/2+ 1/4
        x_dot = y
        y_dot = -x^3 + x
        """
        # x is an array of inputs [x_1,x_2,...] where x_i = (x,y) \in R^{2n} where n is number of degrees of freedom
        # output is array of values [y_1, y_2, ...] where y_i is a real number (1-dimensional)
        (x,y) = input_x[:,0], input_x[:,1]
        return x**4/4 - x**2/2 + y**2/2 + 1/4
    def dH(self, input_x):
        """
        Implementaion of the gradient of the Hamiltonian
        dH(x,y) = [x^3 - x, y]
        """
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dx, dH/dy)
        (x,y) = input_x[:,0], input_x[:,1]
        dx = x**3 - x
        dy = y
        return np.array([dx, dy]).T
