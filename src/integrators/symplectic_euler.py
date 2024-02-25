import numpy as np

class SymplecticEuler():
    """
    ODE Solver, in our case for solving Hamiltonian systems
    q_dot = dq/dt =  dH/dp
    p_dot = dp/dt = -dH/dq

    There 2 variations of this method when solving Hamiltonian systems:

    Implicit q method:
        q_next = q_prev + dt * q_dot(q_next, p_prev) = q_prev + dt * dH/dp(q_next, p_prev)
        p_next = p_prev + dt * p_dot(q_next, p_prev) = p_prev - dt * dH/dq(q_next, p_prev)

    Implicit p method:
        p_next = p_prev + dt * p_dot(q_prev, p_next) = p_prev - dt * dH/dq(q_prev, p_next)
        q_next = q_prev + dt * q_dot(q_prev, p_next) = q_prev + dt * dH/dp(q_prev, p_next)
    """

    def __init__(self):
        pass

    def step_implicit_q(self, x_prev, dH, dt, prec=1e-13):
        """
        Given that we have a (D/2)-DOF system
        using fixed point iteration we compute the next state up to an error given by prec
        because the method is not explicit in general
        Use the first variant explained above (implicit q method)

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

                # [[q_next, p_prev]] of shape (1,D)
                x_implicit_q = np.concatenate((q_next, p_prev)).reshape(-1, D)

                dHdq_implicit_q, dHdp_implicit_q = dH(x_implicit_q)[:, :D//2], dH(x_implicit_q)[:, D//2:]

                # update
                q_next = q_prev + dt * dHdp_implicit_q
                p_next = p_prev - dt * dHdq_implicit_q

                if np.max(np.abs(x_next[k] - x_candidate)) < prec:
                    # fixed point iteration has converged
                    break

        return x_next

    def step_implicit_p(self, x_prev, dH, dt, prec=1e-13):
        """
        Given that we have a (D/2)-DOF system
        using fixed point iteration we compute the next state up to an error given by prec
        because the method is not explicit in general
        Use the first variant explained above (implicit p method)

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

                # [[q_prev, p_next]] of shape (1,D)
                x_implicit_p = np.concatenate((q_prev, p_next)).reshape(-1, D)

                dHdq_implicit_p, dHdp_implicit_p = dH(x_implicit_p)[:, :D//2], dH(x_implicit_p)[:, D//2:]

                # update
                p_next = p_prev - dt * dHdq_implicit_p
                q_next = q_prev + dt * dHdp_implicit_p

                if np.max(np.abs(x_next[k] - x_candidate)) < prec:
                    # fixed point iteration has converged
                    break

        return x_next
