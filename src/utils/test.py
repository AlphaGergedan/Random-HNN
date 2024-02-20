# test the random number generator

# Import necessary modules

# common modules
import os
import sys
import numpy as np

# local modules
directory_to_prepend = os.path.abspath("..")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from utils.grid import generate_train_test_grid

train_rng = np.random.default_rng(23451)
test_rng = np.random.default_rng(54321)
[q_train_range], [p_train_range], [q_train_grid], [p_train_grid], [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_train_test_grid([120], [50], [[-2*3.14,2*3.14]], [[-1,1]], [240], [100], [[-2*3.14,2*3.14]], [[-1,1]], test_rng=test_rng, dof=1, linspace=True, train_rng=train_rng)
# train_rng = np.random.default_rng(23451)
[q_train_range2], [p_train_range2], [q_train_grid2], [p_train_grid2], [q_test_range2], [p_test_range2], [q_test_grid2], [p_test_grid2] = generate_train_test_grid([120], [50], [[-2*3.14,2*3.14]], [[-1,1]], [240], [100], [[-2*3.14,2*3.14]], [[-1,1]], test_rng=test_rng, dof=1, linspace=True, train_rng=train_rng)

assert not np.allclose(q_train_range, q_test_range[:120])
assert not np.allclose(p_train_range, p_test_range[:50])
assert not np.allclose(q_train_grid.flatten(), q_test_grid.flatten()[:6000])
assert not np.allclose(p_train_grid.flatten(), p_test_grid.flatten()[:6000])

assert np.array_equal(q_train_range, q_train_range2)
assert np.array_equal(p_train_range, p_train_range2)
assert np.array_equal(q_train_grid.flatten(), q_train_grid2.flatten())
assert np.array_equal(p_train_grid.flatten(), p_train_grid2.flatten())
