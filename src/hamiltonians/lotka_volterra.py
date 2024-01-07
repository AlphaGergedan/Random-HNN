import numpy as np

class LotkaVolterra():
    """
    Hamiltonian Formulations for the well-known Lotka-Volterra model which is a
    mathematical model used to describe the dynamics of predation-prey populations.

    REF: Y. Tong, S. Xiong, X. He, G. Pan, and B. Zhu, “Symplectic neural networks in Taylor series form for Hamiltonian systems,” J. Comput. Phys., vol. 437, p. 110325, 2021.

    """

    def H(self, x):
        """
        Hamiltonian system with 1 degree of freedom, which means input is (q,p)
        H(q,p) = p - e^p + 2q - e^q
        """
        (q,p) = x[:,0], x[:,1]
        return p - np.e**p + 2*q - np.e**q

    def dH(self, x):
        """
        Implementaion of the gradient of the Hamiltonian
        """
        (q,p) = x[:,0], x[:,1]
        dq = 2 - np.e**q
        dp = 1 - np.e**p
        return np.array([dq, dp]).T
