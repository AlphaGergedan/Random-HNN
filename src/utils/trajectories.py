"""
src/utils/trajectories.py

Functions in this file are used in `/src/main.py` to create training set consisting of time-series data of q and p using the exact flow.
"""

from scipy.integrate import solve_ivp
import numpy as np
from math import isclose
from integrators.index import SemiImplicitEuler

def flow_map_symp_euler(x0, dH, dt_flow_true=1e-4, dt_obs=1e-1):
    """
    Given initial state x0, simulate the exact flow using a strictly smaller time step dt_flow_true and
    return the final state after dt_obs seconds

    @param x0: initial state
    @param dH: gradient of the Hamiltonian function
    @param dt_flow_true: time step of the exact flow
    @param dt_obs: time step difference between x0 and the next state to be returned

    @returns: state x0 after dt_obs seconds
    """
    assert dt_obs > dt_flow_true

    semi_implicit_euler = SemiImplicitEuler()

    t_span = [0, dt_obs]
    t_eval = np.arange(0., round(dt_obs + dt_flow_true, int(np.log10(int(1/dt_flow_true)))), dt_flow_true)
    n_steps = len(t_eval)

    dt = (t_span[1] - t_span[0]) / n_steps

    # here numerical comparison might fail with floats, but being sure here is important
    assert isclose(dt, dt_flow_true, rel_tol=1e-2)

    # start at x0
    x0_next = x0
    for _ in range(n_steps-1):
        x0_next = semi_implicit_euler.step(x0_next.reshape(1,-1), dH, dt)

    return x0_next.reshape(-1)

def flow_map_rk45(x0, dH, dt_flow_true=1e-4, dt_obs=1e-1):
    """
    Given initial state x0, simulate the exact flow using a strictly smaller time step dt_flow_true and
    return the final state after dt_obs seconds
    """
    assert dt_obs > dt_flow_true
    # RK45
    t_span = [0, dt_obs]
    t_eval = np.arange(0., round(dt_obs + dt_flow_true, int(np.log10(int(1/dt_flow_true)))), dt_flow_true)

    J = np.array([[0, 1],
                  [-1, 0]])
    dt = lambda _, x: J@dH(x.reshape(x0.shape)).reshape(-1)

    x1 = solve_ivp(dt, t_span, x0, dense_output=False, t_eval=t_eval, rtol=1e-13).y.T[-1]

    return x1
