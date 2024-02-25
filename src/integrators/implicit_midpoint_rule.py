import numpy as np

class ImplicitMidpointRule():
    """
    ODE Solver, in our case for solving Hamiltonian systems
    q_dot = dq/dt =  dH/dp
    p_dot = dp/dt = -dH/dq

    q_next = q_prev + dt * q_dot((q_prev + q_next) / 2, (p_prev + p_next) / 2)
           = q_prev + dt * dH/dp((q_prev + q_next) / 2, (p_prev + p_next) / 2)
    p_next = p_prev + dt * p_dot((q_prev + q_next) / 2, (p_prev + p_next) / 2)
           = p_prev - dt * dH/dq((q_prev + q_next) / 2, (p_prev + p_next) / 2)
    """

    def __init__(self):
        pass

    def step(self, x_prev, dH, dt, prec=1e-13):
        """
        Given that we have a (D/2)-DOF system
        using fixed point iteration we compute the next state up to an error given by prec
        because the method is not explicit

        @param x_prev           : state x at previous time step, of shape (K,D)
        @param dH               : gradient of the Hamiltonian (dH/dq, dH/dp): (K,D) -> (K,D)
        @param dt               : time step size
        @param prec             : precision of the fixed point iteration

        @returns x_next         : state x at next time step n+1 of shape (K,D)
        """
        K, D = x_prev.shape # (K,D)

        # we will fill the solution here, this is our initial guess
        x_next = x_prev.copy()

        for k in range(K):
            # solve the system for the k-th initial state
            q_prev, p_prev = x_prev[k,:D//2], x_prev[k,D//2:] # each of shape (D//2,)

            while True:
                # store the previous guess
                x_candidate = x_next[k].copy() # of shape (D,)

                # values to update
                q_next, p_next = x_next[k,:D//2], x_next[k,D//2:] # each of shape (D//2,)

                # [[(q_prev + q_next) / 2, (p_prev + p_next) / 2]] of shape (1,D)
                x_midpoint = np.concatenate(((q_prev + q_next) / 2, (p_prev + p_next) / 2)).reshape(-1, D)

                dHdq_midpoint, dHdp_midpoint= dH(x_midpoint)[:, :D//2], dH(x_midpoint)[:, D//2:]

                # update
                q_next = q_prev + dt * dHdp_midpoint
                p_next = p_prev - dt * dHdq_midpoint

                if np.max(np.abs(x_next[k] - x_candidate)) < prec:
                    # fixed point iteration has converged
                    break

        return x_next
