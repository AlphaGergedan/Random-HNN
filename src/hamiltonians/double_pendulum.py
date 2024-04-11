import numpy as np

# see https://diego.assencio.com/?index=e5ac36fcb129ce95a61f8e8ce0572dbf for a good source
# see cranmer-2020
# see jakovac-2022
# see chen-2021
# see jin-2021

class DoublePendulum():
    """
    2 dof
    """
    def __init__(self, m1=1, m2=1, l1=1, l2=1, g=1):
        """
        TODO setup specific
        m1: Mass of first pendulum
        m2: Mass of second pendulum
        l1: length of the first pendulum
        l2: length of the second pendulum
        g: Gravitational acceleration
        TODO damped nonlinear pendulum: L*THETA_dot_dot + b*THETA_dot + g*sin(THETA) = 0
        TODO forced damped nonlinear pendulum: L*THETA_dot_dot + b*THETA_dot + g*sin(THETA) = F*cos(wt)
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    # Hamiltonian Formulations

    def H(self, x):
        """
        Implementation of the Hamiltonian formulation of double nonlienar pendulum
        Using a Legendre transformation of the Lagrangian
        L = T - V = ..
        with p_i = dL/dq_dot_i
        we define H = sum_{i}{p_i * q_dot_i} - L = sum_{i}{p_i * dL/dq_dot_i} where in our formulation q corresponds to THETA and p to THETA_dot
        H = ( (m2 * l2^2 * p1^2 + (m1 + m2) * l1^2 * p2^2 - 2 * m2 * l1 * l2 * p1 * p2 * cos(q1 - q2) ) / (2 * m2 * l1^2 *l2^2 * (m1 + m2 * sin^2(q1 - q2)))) - (m1 + m2) * g * l1 * cos(q1) - m2 * g * l2 * cos(q2)
        we absorb the constants into p and q (set m1 = m2 = l1 = l2 = g = 1) and get the following formulation
        H = ( (p1^2 + 2 * p2^2 - 2 * p1 * p2 * cos(q1 - q2) ) / (2 * (1 + sin^2(q1 - q2)))) - 2 * cos(q1) - cos(q2)
          = ( (p1^2 + 2 * p2^2 - 2 * p1 * p2 * cos(q1 - q2) ) / (2 * (1 + sin^2(q1 - q2)))) - 2 * cos(q1) - cos(q2)
        """
        # x is an array of inputs [x_1,x_2,...] where x_i = (q,p) = (q1,q2,p1,p2) \in R^{2n} = R^{4} where n is number of degrees of freedom
        # x_i = (q,p) where q,p are real numbers (2n-dimensional)
        # q := position \in R^{n} = R^{2}
        # p := momentum \in R^{n} = R^{2}
        # output is array of values [y_1, y_2, ...] where y_i is a real number (1-dimensional)
        # assert x.shape == (x.shape[0],4)
        x = x.reshape(-1,4)
        (q1,q2,p1,p2) = x[:,0], x[:,1], x[:,2], x[:,3]
        f = ( (self.m2 * self.l2**2 * p1**2 + (self.m1 + self.m2) * self.l1**2 * p2**2 - 2 * self.m2 * self.l1 * self.l2 * p1 * p2 * np.cos(q1 - q2) ) / (2 * self.m2 * self.l1**2 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2)) ) - (self.m1 + self.m2) * self.g * self.l1 * np.cos(q1) - self.m2 * self.g * self.l2 * np.cos(q2)
        return f.reshape(-1, 1)

    def dH(self, x):
        """
        Implementaion of the gradient of the Hamiltonian
        dH/dx has 4 dimensions

        Note: the equations are written for m1 = m2 = l1 = l2 = g = 1
        """
        # x is same as above
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq1, dH/dq2, dH/dp1, dH/dp2)
        # assert x.shape == (x.shape[0],4)
        x = x.reshape(-1, 4)
        (q1,q2,p1,p2) = x[:,0], x[:,1], x[:,2], x[:,3]

        # define helper equations used commonly in dH/dq1 and dH/dq2
        h1 = (p1 * p2 * np.sin(q1 - q2)) / (self.l1 * self.l2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))
        h2 = (self.m2 * self.l2**2 * p1**2 + (self.m1 + self.m2) * self.l1**2 * p2**2 - 2 * self.m2 * self.l1 * self.l2 * p1 * p2 * np.cos(q1 - q2)) / (2 * self.l1**2 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2)**2)

        # from paper wu-2020 it is defined without the square in the last term
        # h2 = (self.m2 * self.l2**2 * p1**2 + (self.m1 + self.m2) * self.l1**2 * p2**2 - 2 * self.m2 * self.l1 * self.l2 * p1 * p2 * np.cos(q1 - q2)) / (2 * self.l1**2 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))

        dq1 = (self.m1 + self.m2) * self.g * self.l1 * np.sin(q1) + h1 - h2 * np.sin(2 * (q1 - q2))
        dq2 = self.m2 * self.g * self.l2 * np.sin(q2) - h1 + h2 * np.sin(2 * (q1 - q2))
        dp1 = (self.l2 * p1 - self.l1 * p2 * np.cos(q1 - q2)) / (self.l1**2 * self.l2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))
        dp2 = ((self.m1 + self.m2) * self.l1 * p2 - self.m2 * self.l2 * p1 * np.cos(q1 - q2)) / (self.m2 * self.l1 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))

        # (x.shape[0], 4)
        df = np.array([ dq1, dq2, dp1, dp2 ]).T
        return df.reshape(-1, 4)
