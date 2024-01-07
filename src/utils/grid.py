import numpy as np

def generate_grid_2d(N_q, N_p, q_lim, p_lim):
    """
    Generates meshgrid for the 2D system with position q and momentum p

    N_q: number of grid points in q
    N_p: number of grid points in p
    q_lim: [q_min, q_max]
    p_lim: [p_min, p_max]
    noise: add gaussian noise to the grid with mean 0 and std = noise
    """
    assert type(q_lim) == list
    assert type(p_lim) == list
    assert len(q_lim) == 2
    assert len(p_lim) == 2
    assert q_lim[0] < q_lim[1], "q_lim should be in ascending order"
    assert p_lim[0] < p_lim[1], "p_lim should be in ascending order"
    (q_range, p_range) = np.linspace(q_lim[0], q_lim[1], N_q), np.linspace(p_lim[0], p_lim[1], N_p)
    q_grid, p_grid = np.meshgrid(q_range, p_range)
    return q_range, p_range, q_grid, p_grid
