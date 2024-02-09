import numpy as np


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

def generate_train_test_grid(train_qs, train_ps, train_q_lims, train_p_lims, test_qs, test_ps, test_q_lims, test_p_lims, test_rng, dof, linspace=False, train_rng=None):
    """
    Generates meshgrid for Hamiltonian systems with positions q and momenta p
    with uniform sampling from the given ranges. The test set does not include the training
    set after the sampling.

    This function is intended to create train,test split.

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
