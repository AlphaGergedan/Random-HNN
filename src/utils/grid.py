"""
src/utils/grid.py

Used in `src/main.py` to create train and test sets
"""

import numpy as np

def generate_uniform_train_test_set(train_set_size, train_q_lims, train_p_lims, train_rng, test_set_size, test_q_lims, test_p_lims, test_rng, dof):
    """
    Uniformly places the train and test points within the given domain. Test set is sampled distinctly.

    @param train_set_size   : number of total points to sample for train set
    @param train_q_lims     : domain for q (position) dimensions, as list([min,max])
    @param train_p_lims     : domain for p (momentum) dimensions, as list([min,max])
    @param train_rng        : random number generator used for generating the train set
    @param ... defined analogously for test
    @param dof              : degree of freedom of the system == decides dimensions of the sampled points

    @return x_train, x_test stacked up as (n_points, 2*dof) dimensional arrays
    """
    q_train_grid = []
    p_train_grid = []

    for d in range(dof):
        # sample points randomly
        q_train_grid.append(train_rng.uniform(low=train_q_lims[d][0], high=train_q_lims[d][1], size=(train_set_size)))
        p_train_grid.append(train_rng.uniform(low=train_p_lims[d][0], high=train_p_lims[d][1], size=(train_set_size)))

    q_train_grid = np.array(q_train_grid)
    p_train_grid = np.array(p_train_grid)

    # sample distinct test samples
    q_test_grid = []
    p_test_grid = []

    for d in range(dof):
        q_test_grid.append([])
        while len(q_test_grid[d]) < test_set_size:
            candidate_samples = test_rng.uniform(low=test_q_lims[d][0], high=test_q_lims[d][1], size=(test_set_size - len(q_test_grid[d])))
            # setdiff1d returns all elements in arr1 that are not in arr2
            new_samples = np.setdiff1d(candidate_samples, q_train_grid[d], assume_unique=True)
            q_test_grid[d].extend(new_samples)

        p_test_grid.append([])
        while len(p_test_grid[d]) < test_set_size:
            candidate_samples = test_rng.uniform(low=test_p_lims[d][0], high=test_p_lims[d][1], size=(test_set_size - len(p_test_grid[d])))
            # setdiff1d returns all elements in arr1 that are not in arr2
            new_samples = np.setdiff1d(candidate_samples, p_train_grid[d], assume_unique=True)
            p_test_grid[d].extend(new_samples)

    q_test_grid = np.array(q_test_grid)
    p_test_grid = np.array(p_test_grid)

    # column stacked (q_i, p_i): (N, 2*dof)
    x_train = np.column_stack([ q.flatten() for q in q_train_grid ] + [ p.flatten() for p in p_train_grid ])
    x_test = np.column_stack([ q.flatten() for q in q_test_grid ] + [ p.flatten() for p in p_test_grid ])
    return x_train, x_test


def generate_train_test_grid(train_qs, train_ps, train_q_lims, train_p_lims, test_qs, test_ps, test_q_lims, test_p_lims, test_rng, dof, linspace=False, train_rng=None):
    """
    Generates meshgrid for Hamiltonian systems with positions q and momenta p
    with uniform sampling from the given ranges. The test set does not include the training
    set after the sampling.

    This function is intended to create train,test split, using randomly spaced but linearly placed
    points. If you want to sample the points completely uniformly, then use generate_random_train_test_set

    @param train_qs: number of grid points in q in each dimension as a list in the train set
    @param train_ps: number of grid points in p in each dimension as a list in the train set
    @param train_q_lims: list of [q_min, q_max] in each dimension in the train set
    @param train_p_lims: list of [p_min, p_max] in each dimension in the train set
    @param test_qs: number of grid points in q in each dimension as a list in the train set
    @param test_ps: number of grid points in p in each dimension as a list in the test set
    @param test_q_lims: list of [q_min, q_max] in each dimension in the test set
    @param test_p_lims: list of [p_min, p_max] in each dimension in the test set
    @param dof: degree of freedom of the system
    @param linspace: if True, then the train set is sampled linearly spaced
    @param random_seed: used in case of uniform sampling

    @return train_q_ranges, train_p_ranges, train_q_grids, train_p_grids, test_q_ranges, test_p_ranges, test_q_grids, test_p_grids
    """
    params = [train_qs, train_ps, train_q_lims, train_p_lims, test_qs, test_ps, test_q_lims, test_p_lims]
    for param in params:
        assert len(param) == dof

    # generate train set first
    train_q_ranges, train_p_ranges, train_q_grid, train_p_grid = generate_grid(train_qs, train_ps, train_q_lims, train_p_lims, dof, linspace=linspace, rng=train_rng)

    # now generate distinct test set
    test_q_ranges = []
    test_p_ranges = []

    # initialize empty set
    for d in range(dof):
        test_q_ranges.append([])
        test_p_ranges.append([])

    # sample distinct test samples
    for d in range(dof):
        while len(test_q_ranges[d]) < test_qs[d]:
            candidate_samples = test_rng.uniform(low=test_q_lims[d][0], high=test_q_lims[d][1], size=(test_qs[d] - len(test_q_ranges[d])))
            # setdiff1d returns all elements in arr1 that are not in arr2
            new_samples = np.setdiff1d(candidate_samples, train_q_ranges[d])
            test_q_ranges[d].extend(new_samples)
        test_q_ranges[d] = np.asarray(test_q_ranges[d][:test_qs[d]])

        while len(test_p_ranges[d]) < test_ps[d]:
            candidate_samples = test_rng.uniform(low=test_p_lims[d][0], high=test_p_lims[d][1], size=(test_ps[d] - len(test_p_ranges[d])))
            # setdiff1d returns all elements in arr1 that are not in arr2
            new_samples = np.setdiff1d(candidate_samples, train_p_ranges[d])
            test_p_ranges[d].extend(new_samples)
        test_p_ranges[d] = np.asarray(test_p_ranges[d][:test_ps[d]])

    test_grids = np.meshgrid(*(test_q_ranges + test_p_ranges))

    # returns q_grids, p_grids
    return train_q_ranges, train_p_ranges, train_q_grid, train_p_grid, test_q_ranges, test_p_ranges, test_grids[:dof], test_grids[dof:]


def generate_grid(N_qs, N_ps, q_lims, p_lims, dof, linspace=False, rng=None):
    """
    Generates meshgrid for Hamiltonian systems with positions q and momenta p

    This function is intended for plotting since it creates the grid linearly spaced.

    @param N_qs: number of grid points in q in each dimension as a list
    @param N_ps: number of grid points in p in each dimension as a list
    @param q_lims: list of [q_min, q_max] in each dimension
    @param p_lims: list of [p_min, p_max] in each dimension
    @param dof: degree of freedom of the system
    @param linspace: if True, the grid is linearly spaced, otherwise uniformly distributed
    @param random_seed: used in case of uniform sampling if given

    @return q_ranges, p_ranges, q_grids, p_grids
    """
    params = [N_qs, N_ps, q_lims, p_lims]
    for param in params:
        assert len(param) == dof

    q_ranges = []
    p_ranges = []

    if linspace:
        for d in range(dof):
            q_ranges.append(np.linspace(q_lims[d][0], q_lims[d][1], N_qs[d]))
            p_ranges.append(np.linspace(p_lims[d][0], p_lims[d][1], N_ps[d]))
    else:
        assert rng != None
        for d in range(dof):
            q_ranges.append(rng.uniform(low=q_lims[d][0], high=q_lims[d][1], size=(N_qs[d])))
            p_ranges.append(rng.uniform(low=p_lims[d][0], high=p_lims[d][1], size=(N_ps[d])))

    grids = np.meshgrid(*(q_ranges + p_ranges))

    # returns q_grids, p_grids
    return q_ranges, p_ranges, grids[:dof], grids[dof:]

