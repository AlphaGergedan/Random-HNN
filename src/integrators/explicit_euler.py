import numpy as np

class ExplicitEuler():
    """
    ODE Solver, in our case for solving Hamiltonian systems
    q_dot = dq/dt =  dH/dp
    p_dot = dp/dt = -dH/dq

    q_next = q_prev + dt * q_dot(q_prev, p_prev) = q_prev + dt * dH/dp(q_prev, p_prev)
    p_next = p_prev + dt * p_dot(q_prev, p_prev) = p_prev - dt * dH/dq(q_prev, p_prev)
    """

    def __init__(self):
        pass

    def step(self, x_prev, dH, dt):
        """
        Given that we have K previous states and a (D/2)-DOF system
        solve the explicit Euler equation and compute the next state
        Note that the method is explicit regardless of the used Hamiltonian

        @param x_prev           : state x at previous time step, of shape (K,D)
        @param dH               : gradient of the Hamiltonian (dH/dq, dH/dp): (K,D) -> (K,D)
        @param dt               : time step size

        @returns x_next         : state x at next time step n+1 of shape (K,D)
        """
        _, D = x_prev.shape # (K,D)

        q_prev, p_prev = x_prev[:,:D//2], x_prev[:,D//2:]

        # gradient of H using the previous state values
        dHdq_prev, dHdp_prev = dH(x_prev)[:,:D//2], dH(x_prev)[:,D//2:] # each of shape (K,D//2)

        return np.hstack((
            q_prev + dt * dHdp_prev,  # q_next
            p_prev - dt * dHdq_prev   # p_next
        ))
