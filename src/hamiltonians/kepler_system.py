import numpy as np

class KeplerSystem():
    """
    4 dof systems

    Taken from tong-2021

    Consider eight-dimensional system (4 dof), a two-body problem in 2D space,
    where (q1,q2) and (p1,p2) are position and momentum associated with the first body,
    and (q3,q4) and (p3,p4) are position and momentum associated with the second body.

    In their paper they pick training points from [-3,3]x[-2,2]
    """

    def H(self, x):
        assert x.shape == (x.shape[0],8)
        (q1,q2,q3,q4,p1,p2,p3,p4) = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6], x[:,7]

        f = (p1**2 + p2**2 + p3**2 + p4**2) / 2 - (q1**2 + q2**2 + q3**2 + q4**2)**(-0.5)
        return f.reshape(-1, 1)

    def dH(self, x):
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq1, dH/dq2, dH/dp1, dH/dp2)
        assert x.shape == (x.shape[0],8)
        (q1,q2,q3,q4,p1,p2,p3,p4) = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6], x[:,7]

        # define helper equations used commonly in dH/dq1 and dH/dq2
        dq_term = (q1**2 + q2**2 + q3**2 + q4**2)**(-1.5)

        dq1 = q1 * dq_term
        dq2 = q2 * dq_term
        dq3 = q3 * dq_term
        dq4 = q4 * dq_term

        dp1 = p1
        dp2 = p2
        dp3 = p3
        dp4 = p4

        # (x.shape[0], 4)
        df = np.array([ dq1, dq2, dq3, dq4, dp1, dp2, dp3, dp4 ]).T
        return df.reshape(-1, 8)
