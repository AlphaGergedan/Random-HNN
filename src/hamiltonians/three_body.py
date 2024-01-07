import numpy as np

class ThreeBody():
    def __init__(self, m1=1, m2=1, m3=1, g=1):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.g = g

    # Hamiltonian Formulations

    def H(self, x):
        assert x.shape == (x.shape[0],4)
        (q1,q2,q3,p1,p2,p3) = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5]

        return (p1**2) / (2 * self.m1) + (p2**2) / (2 * self.m2) + (p3**2) / (2 * self.m3) - (self.g * self.m1 * self.m2) / (np.abs(q1 - q2)) - (self.g * self.m2 * self.m3) / (np.abs(q2 - q3)) - (self.g * self.m1 * self.m3) / (np.abs(q1 - q3))

    def dH(self, x):
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq1, dH/dq2, dH/dp1, dH/dp2)
        assert x.shape == (x.shape[0],4)
        (q1,q2,q3,p1,p2,p3) = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5]

        # define helper equations used commonly in dH/dq1 and dH/dq2
        # TODO
        dq1 = 0
        dq2 = 0
        dq3 = 0
        dp1 = 0
        dp2 = 0
        dp3 = 0

        # (x.shape[0], 4)
        return np.array([ dq1, dq2, dq3, dp1, dp2, dp3 ]).T
