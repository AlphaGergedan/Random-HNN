import numpy as np

class Trigonometric():
    """
    1 dof
    """
    def __init__(self, freq=1):
        self.freq = freq

    # Hamiltonian Formulations

    def H(self, x):
        """
        Implementation of example Hamiltonian using Trigonometric equations
        H = sin^2(q) + cos^2(p)
        """
        x = x.reshape(-1, 2)
        (q,p) = x[:,0], x[:,1]
        f = np.sin(self.freq * q)**2 + np.cos(self.freq * p)**2
        return f.reshape(-1, 1)

    def dH(self, x):
        """
        Gradient
        dH/dq = 2*sin(q)*cos(q)
        dH/dp = -2*cos(p)*sin(p)
        """
        x = x.reshape(-1, 2)
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq, dH/dp)
        (q,p) = x[:,0], x[:,1]
        dq = self.freq * 2 * np.sin(q) * np.cos(q)
        dp = self.freq * -2 * np.cos(p) * np.sin(p)
        df = np.array([dq, dp]).T
        return df.reshape(-1, 2)
