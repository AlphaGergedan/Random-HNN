import numpy as np

# see galactic model by Zotos, mentioned in paper https://doi.org/10.3847/1538-4365/ac1ff3
# a 6d example

class GalacticModel():
    """
    3 dof
    """

    def __init__(self):
        self.v0 = self.alpha = self.beta = self.l = self.cb = self.g = self.m = self.cn = 1

    # Hamiltonian Formulations

    # host elliptical galaxy with logarithmic potential
    def Vg(self, q1, q2, q3):
        # if we set constants to 1 then we get
        # (1 / 2) * np.log(q1**2 + q2**2 + q3**2 - q1**3 + 1)
        if (q1**2 + self.alpha * q2**2 + self.beta * q3**2 - self.l * q1**3 + self.cb**2 + 1e-100 <= 0).all():
            assert False
        return (self.v0 / 2) * np.log(q1**2 + self.alpha * q2**2 + self.beta * q3**2 - self.l * q1**3 + self.cb**2 + 1e-100)

    def Vb(self, q1, q2, q3):
        # if we set constants to 1 then we get
        # 1 / np.sqrt(q1**2 + q2**2 + q3**2 + 1)
        return (self.g * self.m) / np.sqrt(q1**2 + q2**2 + q3**2 + self.cn**2)

    def H(self, x):
        assert x.shape == (x.shape[0],6)
        (q1,q2,q3,p1,p2,p3) = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5]
        return (p1**2 + p2**2 + p3**2) / 2 + self.Vg(q1,q2,q3) + self.Vb(q1,q2,q3)

    def dH(self, x):
        assert x.shape == (x.shape[0],6)
        (q1,q2,q3,p1,p2,p3) = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5]

        dq1 = p1
        dq2 = p2
        dq3 = p3
        # if constants are 1 then
        # -(q1 / np.power((1 + q1**2 + q2**2 + q3**2), (3 / 2))) - (((2 * q1 - 3 * q1**2)) / (2 * (1 + q1**2 - q1**3 + q2**2 + q3 ** 2)))
        dp1 = -((self.g * self.m * q1) / np.power((self.cn**2 + q1**2 + q2**2 + q3**2), (3 / 2))) - ((self.v0**2 * (2 * q1 - 3 * self.l * q1**2)) / (2 * (self.cb**2 + q1**2 - self.l * q1**3 + self.alpha * q2**2 + self.beta * q3 ** 2)))
        # -(q2 / np.power((1 + q1**2 + q2**2 + q3**2), (3 / 2))) - (q2 / (1 + q1**2 - q1**3 + q2**2 + q3**2))
        dp2 = -((self.g * self.m * q2) / np.power((self.cn**2 + q1**2 + q2**2 + q3**2), (3 / 2))) - ((self.alpha * self.v0**2 * q2) / (self.cb**2 + q1**2 - self.l * q1**3 + self.alpha * q2**2 + self.beta * q3**2))
        # -(q3 / np.power((1 + q1**2 + q2**2 + q3**2), (3 / 2))) - (q3 / (1 + q1**2 - q1**3 + q2**2 + q3**2))
        dp3 = -((self.g * self.m * q3) / np.power((self.cn**2 + q1**2 + q2**2 + q3**2), (3 / 2))) - ((self.beta * self.v0**2 * q3) / (self.cb**2 + q1**2 - self.l * q1**3 + self.alpha * q2**2 + self.beta * q3**2))

        # (x.shape[0], 6)
        return np.array([ dq1, dq2, dq3, dp1, dp2, dp3 ]).T
