import numpy as np

def generate_grid_2d(N_q, N_p, q_lim, p_lim):
    """
    Generates meshgrid for the 2D system with position q and momentum p

    @param N_q: number of grid points in q
    @param N_p: number of grid points in p
    @param q_lim: [q_min, q_max]
    @param p_lim: [p_min, p_max]
    @param noise: add gaussian noise to the grid with mean 0 and std = noise

    @return q_range, p_range, q_grid, p_grid
    """
    assert type(q_lim) == list and type(p_lim) == list
    assert len(q_lim) == 2 and len(p_lim) == 2
    assert q_lim[0] <= q_lim[1], "q_lim should be in ascending order"
    assert p_lim[0] <= p_lim[1], "p_lim should be in ascending order"

    q_range, p_range = np.linspace(q_lim[0], q_lim[1], N_q), np.linspace(p_lim[0], p_lim[1], N_p)
    q_grid, p_grid = np.meshgrid(q_range, p_range)

    return q_range, p_range, q_grid, p_grid

def generate_grid_4d(N_q1, N_q2, N_p1, N_p2, q1_lim, q2_lim, p1_lim, p2_lim):
    """
    Generates meshgrid for the 4D system with positions q and momenta p

    @param N_q1: number of grid points in q1
    @param N_q2: number of grid points in q2
    @param N_p1: number of grid points in p1
    @param N_p2: number of grid points in p2
    @param q1_lim: [q1_min, q1_max]
    @param q2_lim: [q2_min, q2_max]
    @param p1_lim: [p1_min, p1_max]
    @param p2_lim: [p2_min, p2_max]
    @param noise: add gaussian noise to the grid with mean 0 and std = noise

    @return q1_range, q2_range, p1_range, p2_range, q1_grid, q2_grid, p1_grid, p2_grid
    """
    assert type(q1_lim) == list and type(q2_lim) == list and type(p1_lim) == list and type(p2_lim) == list
    assert len(q1_lim) == 2 and len(q2_lim) == 2 and len(p1_lim) == 2 and len(p2_lim) == 2
    assert q1_lim[0] <= q1_lim[1], "q1_lim should be in ascending order"
    assert q2_lim[0] <= q2_lim[1], "q2_lim should be in ascending order"
    assert p1_lim[0] <= p1_lim[1], "p1_lim should be in ascending order"
    assert p2_lim[0] <= p2_lim[1], "p2_lim should be in ascending order"

    q1_range, q2_range = np.linspace(q1_lim[0], q1_lim[1], N_q1), np.linspace(q2_lim[0], q2_lim[1], N_q2)
    p1_range, p2_range = np.linspace(p1_lim[0], p1_lim[1], N_p1), np.linspace(p2_lim[0], p2_lim[1], N_p2)
    q1_grid, q2_grid, p1_grid, p2_grid = np.meshgrid(q1_range, q2_range, p1_range, p2_range)

    return q1_range, q2_range, p1_range, p2_range, q1_grid, q2_grid, p1_grid, p2_grid

def generate_grid_6d(N_q1, N_q2, N_q3, N_p1, N_p2, N_p3, q1_lim, q2_lim, q3_lim, p1_lim, p2_lim, p3_lim):
    """
    Generates meshgrid for the 6D system with positions q and momenta p

    @param N_q1: number of grid points in q1
    @param N_q2: number of grid points in q2
    @param N_q3: number of grid points in q3
    @param N_p1: number of grid points in p1
    @param N_p2: number of grid points in p2
    @param N_p3: number of grid points in p3
    @param q1_lim: [q1_min, q1_max]
    @param q2_lim: [q2_min, q2_max]
    @param q3_lim: [q3_min, q3_max]
    @param p1_lim: [p1_min, p1_max]
    @param p2_lim: [p2_min, p2_max]
    @param p3_lim: [p3_min, p3_max]
    @param noise: add gaussian noise to the grid with mean 0 and std = noise

    @return q1_range, q2_range, q3_range, p1_range, p2_range, p3_range, q1_grid, q2_grid, q3_grid, p1_grid, p2_grid, p3_grid
    """
    assert type(q1_lim) == list and type(q2_lim) == list and type(q3_lim) == list and type(p1_lim) == list and type(p2_lim) == list and type(p3_lim) == list
    assert len(q1_lim) == 2 and len(q2_lim) == 2 and len(q3_lim) == 2 and len(p1_lim) == 2 and len(p2_lim) == 2 and len(p3_lim) == 2
    assert q1_lim[0] <= q1_lim[1], "q1_lim should be in ascending order"
    assert q2_lim[0] <= q2_lim[1], "q2_lim should be in ascending order"
    assert q3_lim[0] <= q3_lim[1], "q3_lim should be in ascending order"
    assert p1_lim[0] <= p1_lim[1], "p1_lim should be in ascending order"
    assert p2_lim[0] <= p2_lim[1], "p2_lim should be in ascending order"
    assert p3_lim[0] <= p3_lim[1], "p3_lim should be in ascending order"

    q1_range, q2_range, q3_range = np.linspace(q1_lim[0], q1_lim[1], N_q1), np.linspace(q2_lim[0], q2_lim[1], N_q2), np.linspace(q3_lim[0], q3_lim[1], N_q3)
    p1_range, p2_range, p3_range = np.linspace(p1_lim[0], p1_lim[1], N_p1), np.linspace(p2_lim[0], p2_lim[1], N_p2), np.linspace(p3_lim[0], p3_lim[1], N_p3)
    q1_grid, q2_grid, q3_grid, p1_grid, p2_grid, p3_grid = np.meshgrid(q1_range, q2_range, q3_range, p1_range, p2_range, p3_range)

    return q1_range, q2_range, q3_range, p1_range, p2_range, p3_range, q1_grid, q2_grid, q3_grid, p1_grid, p2_grid, p3_grid

def generate_grid_8d(N_q1, N_q2, N_q3, N_q4, N_p1, N_p2, N_p3, N_p4, q1_lim, q2_lim, q3_lim, q4_lim, p1_lim, p2_lim, p3_lim, p4_lim):
    """
    Generates meshgrid for the 8D system with positions q and momenta p

    @param N_q1: number of grid points in q1
    @param N_q2: number of grid points in q2
    @param N_q3: number of grid points in q3
    @param N_q4: number of grid points in q4
    @param N_p1: number of grid points in p1
    @param N_p2: number of grid points in p2
    @param N_p3: number of grid points in p3
    @param N_p4: number of grid points in p4
    @param q1_lim: [q1_min, q1_max]
    @param q2_lim: [q2_min, q2_max]
    @param q3_lim: [q3_min, q3_max]
    @param q4_lim: [q4_min, q4_max]
    @param p1_lim: [p1_min, p1_max]
    @param p2_lim: [p2_min, p2_max]
    @param p3_lim: [p3_min, p3_max]
    @param p4_lim: [p4_min, p4_max]
    @param noise: add gaussian noise to the grid with mean 0 and std = noise

    @return q1_range, q2_range, q3_range, q4_range, p1_range, p2_range, p3_range, p4_range, q1_grid, q2_grid, q3_grid, q4_grid, p1_grid, p2_grid, p3_grid, p4_grid
    """
    assert type(q1_lim) == list and type(q2_lim) == list and type(q3_lim) == list and type(q4_lim) == list and type(p1_lim) == list and type(p2_lim) == list and type(p3_lim) == list and type(p4_lim) == list
    assert len(q1_lim) == 2 and len(q2_lim) == 2 and len(q3_lim) == 2 and len(q4_lim) == 2 and len(p1_lim) == 2 and len(p2_lim) == 2 and len(p3_lim) == 2 and len(p4_lim) == 2
    assert q1_lim[0] <= q1_lim[1], "q1_lim should be in ascending order"
    assert q2_lim[0] <= q2_lim[1], "q2_lim should be in ascending order"
    assert q3_lim[0] <= q3_lim[1], "q3_lim should be in ascending order"
    assert q4_lim[0] <= q4_lim[1], "q4_lim should be in ascending order"
    assert p1_lim[0] <= p1_lim[1], "p1_lim should be in ascending order"
    assert p2_lim[0] <= p2_lim[1], "p2_lim should be in ascending order"
    assert p3_lim[0] <= p3_lim[1], "p3_lim should be in ascending order"
    assert p4_lim[0] <= p4_lim[1], "p4_lim should be in ascending order"

    q1_range, q2_range, q3_range, q4_range = np.linspace(q1_lim[0], q1_lim[1], N_q1), np.linspace(q2_lim[0], q2_lim[1], N_q2), np.linspace(q3_lim[0], q3_lim[1], N_q3), np.linspace(q4_lim[0], q4_lim[1], N_q4)
    p1_range, p2_range, p3_range, p4_range = np.linspace(p1_lim[0], p1_lim[1], N_p1), np.linspace(p2_lim[0], p2_lim[1], N_p2), np.linspace(p3_lim[0], p3_lim[1], N_p3), np.linspace(p4_lim[0], p4_lim[1], N_p4)

    q1_grid, q2_grid, q3_grid, q4_grid, p1_grid, p2_grid, p3_grid, p4_grid = np.meshgrid(q1_range, q2_range, q3_range, q4_range, p1_range, p2_range, p3_range, p4_range)

    return q1_range, q2_range, q3_range, q4_range, p1_range, p2_range, p3_range, p4_range, q1_grid, q2_grid, q3_grid, q4_grid, p1_grid, p2_grid, p3_grid, p4_grid

# TODO generalize the functions above for generate_grid_2Nd(N_q, N_p, q_lim, p_lim) where each parameter is a list

# def generate_grid_2Nd(N_q, N_p, q_lim, p_lim):
    # """
    # Generates meshgrid for the 2*N-Dimenional system with position q and momentum p
#
    # @param N_q: list of number of grid points in q, e.g. [N_q1, N_q2] for N=2
    # @param N_p: list of number of grid points in p, e.g. [N_p1, N_p2] for N=2
    # @param q_lim: list of list of [q_min,q_max], e.g. [ [q1_min, q1_max], [q2_min, q2_max] ] for N=2
    # @param p_lim: list of list of [p_min,p_max], e.g. [ [p1_min, p1_max], [p2_min, p2_max] ] for N=2
    # @param noise: add gaussian noise to the grid with mean 0 and std = noise
#
    # @return q_range, p_range, q_grid, p_grid
    # """
    # assert len(N_q) == len(N_p) == len(q_lim) == len(p_lim)
#
    # q_range = p_range = q_grid = p_grid = []
#
    # for i in range(len(N_q)):
        # assert type(q_lim[i]) == list and type(p_lim[i]) == list
        # assert len(q_lim[i]) == 2 and len(p_lim[i]) == 2
#
        # q_range.append(np.linspace(q_lim[i][0], q_lim[i][1], N_q[i]))
        # p_range.append(np.linspace(p_lim[i][0], p_lim[i][1], N_p[i]))
#
    # q_grid, p_grid = np.meshgrid(q_range, p_range)
#
    # return q_range, p_range, q_grid, p_grid
