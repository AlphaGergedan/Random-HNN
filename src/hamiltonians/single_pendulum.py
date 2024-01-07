import numpy as np

class SinglePendulum():
    """
    1 dof
    """
    def __init__(self, m=1, l=1, g=1):
        """
        TODO setup specific
        m: Mass of pendulum
        l: Length of pendulum
        g: Gravitational acceleration
        TODO damped nonlinear pendulum: L*THETA_dot_dot + b*THETA_dot + g*sin(THETA) = 0
        TODO forced damped nonlinear pendulum: L*THETA_dot_dot + b*THETA_dot + g*sin(THETA) = F*cos(wt)
        """
        self.m = m
        self.l = l
        self.g = g

    # Hamiltonian Formulations

    def H(self, x):
        """
        Implementation of the Hamiltonian formulation of single nonlienar pendulum
        Using a Legendre transformation of the Lagrangian
        L = T - V = 1/2 * m * l^2 * THETA_dot^2 - m * g * l * (1 - cos(THETA))
        with p_i = dL/dq_dot_i
        we define H = sum_{i}{p_i * q_dot_i} - L = sum_{i}{p_i * dL/dq_dot_i} where in our formulation q corresponds to THETA and p to THETA_dot
        H = 1/(2 * m * l^2) * p^2 + m * g * l * (1 - cos(q))
        we absorb the constants into p and q (set m = g = l = 1) and get the following formulation
        H = 1/2 * p^2 + (1 - cos(q))
        """
        # x is an array of inputs [x_1,x_2,...] where x_i = (q,p) \in R^{2n} where n is number of degrees of freedom
        # x_i = (q,p) where q,p are real numbers (2n-dimensional)
        # q := position \in R^{n}
        # p := momentum \in R^{n}
        # output is array of values [y_1, y_2, ...] where y_i is a real number (1-dimensional)
        (q,p) = x[:,0], x[:,1]
        return p**2 / 2 + (1 - np.cos(q))
    def dH(self, x):
        """
        Implementaion of the gradient of the Hamiltonian
        """
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq, dH/dp)
        (q,p) = x[:,0], x[:,1]
        dq = np.sin(q)
        dp = p
        return np.array([dq, dp]).T