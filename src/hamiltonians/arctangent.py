import numpy as np

class Arctangent():
    """
    1 dof
    """
    def __init__(self, freq=1):
        self.freq = freq

    # Hamiltonian Formulations

    def H(self, x):
        """
        Implementation of example Hamiltonian using Trigonometric equations
        H = arctan(freq * q^2) + arctan(freq * p^2)
        """
        x = x.reshape(-1, 2)
        (q,p) = x[:,0], x[:,1]
        f = np.arctan(self.freq * q**2) + np.arctan(self.freq * p**2)
        return f.reshape(-1, 1)

    def dH(self, x):
        """
        Gradient
        dH/dq = freq * 2 * q * (1 / (1+ (freq*q^2)^2))
        dH/dp = freq * 2 * p * (1 / (1+ (freq*p^2)^2))
        """
        x = x.reshape(-1, 2)
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq, dH/dp)
        (q,p) = x[:,0], x[:,1]
        dq = self.freq * 2 * q * (1 / (1 + (self.freq * q**2)**2))
        dp = self.freq * 2 * p * (1 / (1 + (self.freq * p**2)**2))
        df = np.array([dq, dp]).T
        return df.reshape(-1, 2)
