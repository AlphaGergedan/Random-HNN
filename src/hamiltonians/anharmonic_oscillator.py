import numpy as np

class AnharmonicOscillator():
    """
    1 dof and undamped
    example taken from: yildiz-2023
    """

    # Hamiltonian Formulations

    def H(self, input_x):
        """
        H(q,p) = p^2 / 2 + q^2 / 2 + q^4 / 4
        """
        (q,p) = input_x[:,0], input_x[:,1]
        f = p**2 / 2 + q**2 / 2 + q**4 / 4
        return f.reshape(-1, 1)

    def dH(self, input_x):
        """
        Implementaion of the gradient of the Hamiltonian
        """
        (q,p) = input_x[:,0], input_x[:,1]
        dq = q + q**3
        dp = p
        df = np.array([dq, dp]).T
        return df.reshape(-1, 2)
