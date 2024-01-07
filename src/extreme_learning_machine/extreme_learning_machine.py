import numpy as np

# common activation functions and their derivatives
from activations import Activations

class ELM():
    """
    Extreme Learning Machine with a single hidden layer for regression

    Terminology
        - x_train = (X,dX,x0,f0) where
        - X == training data points of dimension (K,D)
        - dX == derivatives of the target function evaluated at points X of dimension (K,D)
        - x0 == a single data point where we know it's function value, of dimension (D)
        - f0 == f(x0), true function value, of dimension (1)
        - K == number of samples, i.e. size of training data
        - D == number of features per sample
        - M == number of neurons in the hidden layer
        - activation == activation function used in the hidden layer, typically nonlinear
        - dense_layer_weights == hidden layer weights (should be initialized randomly) of shape (D,M)
        - dense_layer_biases == hidden layer biases of shape (M,)
        - linear_layer_weights == output layer weights (should be fit using by solving the linear system) of shape (M,1)
        - linear_layer_biases == output layer biases (should be fit using by solving the linear system) of shape (1,)
        - rcond == solver tolerance for lstsq
    """

    # weights of the hidden layer
    def __init__(self, X, dX, x0, f0, n_hidden, rcond=1e-10, activation='tanh', verbose=False, prune_duplicates=False, repetition_scaler=1, dist_min=1e-10, parameter_sampler="relu", swim=False, random_seed=1):
        # set up traning data
        X = self.prepare_x(X)   # (K,D)
        dX = self.prepare_x(dX) # (K,D)
        x0 = self.prepare_x(x0) # (D,1)
        f0 = self.prepare_x(f0) # (1,1)
        self.validate_inputs(X, dX, x0, f0)

        # get the dimensions of the function that we want to approximate: (D) -> (1)
        self.K, self.D = X.shape
        self.x_train = (X,dX,x0,f0)
        self.M = n_hidden
        self.regularization_scale = rcond
        # set the activation and its derivative
        # the derivativeIs needed for derivative calculation of the hidden layer
        # for solving the linear system
        match activation:
            case "relu":
                self.activation = Activations.relu
                self.d_activation = Activations.d_relu
            case "leaky_relu":
                self.activation = Activations.leaky_relu
                self.d_activation = Activations.leaky_relu
            case "parametric_relu":
                self.activation = Activations.parametric_relu
                self.d_activation = Activations.d_parametric_relu
            case "elu":
                self.activation = Activations.elu
                self.d_activation = Activations.d_elu
            case "tanh":
                self.activation = Activations.tanh
                self.d_activation = Activations.d_tanh
            case "sigmoid":
                self.activation = Activations.sigmoid
                self.d_activation = Activations.d_sigmoid
            case _:
                self.activation = Activations.identity
                self.d_activation = Activations.d_identity

        self.verbose = verbose

        # swim related set up
        self.swim=swim
        self.prune_duplicates=prune_duplicates
        self.repetition_scaler=repetition_scaler
        self.dist_min=dist_min
        self.idx_from = None
        self.idx_to = None
        self.n_pruned_neurons = 0
        self.rng = np.random.default_rng(random_seed)

        if parameter_sampler == "relu":
            self.parameter_sampler = self.sample_parameters_relu
        elif parameter_sampler == "tanh":
            self.parameter_sampler = self.sample_parameters_tanh
        elif parameter_sampler == "random":
            self.parameter_sampler = self.sample_parameters_randomly
        else:
            raise ValueError(f"Unknown parameter sampler {self.parameter_sampler}.")

        # random initialization of the weights
        self.linear_layer_weights = self.rng.normal(loc=0, scale=1, size=(self.M, 1)) # we solve Aw=b in the fit function for this
        self.linear_layer_biases = self.rng.normal(loc=0, scale=1, size=(1,)) # final linear layer bias
        self.dense_layer_weights = self.rng.normal(loc=0, scale=1, size=(self.D, self.M)) # in EML we randomly intialize this
        self.dense_layer_biases = self.rng.uniform(low=-3, high=3, size=(self.M,)) # in EML we randomly initialize this, bias applied in the hidden layer

        # TODO choose biases from uniformly -a,a , -8,8 in this data
        if verbose:
            print('------------------------')
            print('ELM is ready, model info:')
            print('X shape (K,D)', X.shape)
            print('dX shape (K,D)', dX.shape)
            print('x0 shape (D,1)', x0.shape)
            print('f0 shape (1,1)', f0.shape)
            print('linear_layer_weights shape (M,1)', self.linear_layer_weights.shape)
            print('dense_layer_weights shape (D,M)', self.dense_layer_weights.shape)

    def forward(self, X):
        """
        X : numpy array of shape (N,D) where N is the number of samples and D the number of features
        """
        X = self.prepare_x(X)
        phi_0 = X # (N,D)
        phi_1 = self.activation(np.subtract(phi_0 @ self.dense_layer_weights, self.dense_layer_biases)) # (N,D) x (D,M) = (N,M)
        phi_2 = np.subtract(phi_1 @ self.linear_layer_weights, self.linear_layer_biases) # (N,M) x (M,1) = (N,1)
        return phi_2.reshape(X.shape[0]) # output (number of samples) instead of (K,1) I think it is more convenient

    def fit(self):
        """
        here we want to solve the linear equation Aw=b for w to find the optimal weights w
        setup a linear system to approximate the weights: A*w=b

        See the notebook for reference
        A : (DK+1,M)
        w : (M)
        b : (DK+1)
        """
        # X  : (K,D)
        # dX : (K,D)
        # x0 : (D,1)
        # f0 : (1,1)
        (X,dX,x0,f0) = self.x_train

        if self.swim:
            # SWIM? then sample the weights of the first dense layer "dense_layer_weights" and "dense_layer_biases"
            # TODO: integrate this
            if self.M is None:
                raise ValueError("M must be set.")

            # do not stack the x0
            x = self.prepare_x(X)

            weights, biases, idx_from, idx_to = self.parameter_sampler(x, self.rng)

            self.idx_from = idx_from
            self.idx_to = idx_to
            self.dense_layer_weights = weights.reshape(self.D, self.M)
            self.dense_layer_biases = biases.reshape(self.M)
            self.n_parameters = np.prod(weights.shape) + np.prod(biases.shape)

        self.solve_lstsq()

    def evaluate(self, X, Y_true):
        """
        Given the true labels y_true=f(X) evaluate the model's prediction
        """
        Y_pred = self.forward(X).flatten() # (K)
        Y_true = Y_true.flatten() # (K)

        assert Y_pred.shape == Y_true.shape

        return np.sum((Y_true- Y_pred)**2) / X.shape[0]

    def prepare_x(self, x):
        """
        prepares x to be of shape (K,D)
        """
        x = x.reshape(x.shape[0], -1)
        return x

    def prepare_y(self, y):
        """Prepares labels for the sampling.

        For the classification problem, applies one-hot encoding for the labels.
        For the regression problem, adds a dimension to the labels if neccesary.
        """
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        return y, y

    def clean_inputs(self, x, y):
        x = self.prepare_x(x)
        y, _ = self.prepare_y(y)
        return x, y

    def validate_inputs(self, X, dX, x0, f0):
        assert len(X.shape) == 2
        # print('assertion passed! len(X.shape) == 2')
        assert X.shape == dX.shape
        # print('assertion passed! X.shape == dX.shape')
        D = X.shape[1] # num. of features
        assert len(x0.shape) == 2
        # print('assertion passed! len(x0.shape) == 2')
        assert x0.shape[0] == D # num. of features
        # print('assertion passed! x0.shape[0] == 1')
        assert x0.shape[1] == 1 # num. of samples, x0 should be a single point
        # print('assertion passed! x0.shape[1] == D = 2')
        assert f0.shape == (1,1) # should always be (1,1) for Hamiltonian functions
        # print('assertion passed! f0.shape == (1,1)')

    def sample_parameters_tanh(self, x, rng):
        scale = 0.5 * (np.log(1 + 1/2) - np.log(1 - 1/2))

        directions, dists, idx_from, idx_to = self.sample_parameters(x, rng)
        weights = (2 * scale * directions / dists).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1) - scale

        return weights, biases, idx_from, idx_to

    def sample_parameters_relu(self, x, rng):
        scale = 1.0

        directions, dists, idx_from, idx_to = self.sample_parameters(x, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)

        return weights, biases, idx_from, idx_to

    def sample_parameters_randomly(self, x, rng):
        weights = rng.normal(loc=0, scale=1, size=(self.M, x.shape[1])).T
        biases = rng.uniform(low=-np.pi, high=np.pi, size=(self.M, 1)).T
        idx0 = None
        idx1 = None
        return weights, biases, idx0, idx1

    def sample_parameters(self, x, rng):
        """
        Sample directions from points to other points in the given dataset (x, y).
        """

        # n_repetitions repeats the sampling procedure to find better directions.
        # If we require more samples than data points, the repetitions will cause more pairs to be drawn.
        # n_repetitions = max(1, int(np.ceil(self.M / x.shape[0]))) * self.repetition_scaler
        # print("n_repetitions:", n_repetitions)

        # This guarantees that:
        # (a) we draw from all the N(N-1)/2 - N possible pairs (minus the exact idx_from=idx_to case)
        # (b) no indices appear twice at the same position (never idx0[k]==idx1[k] for all k)
        # print("high is: ", x.shape[0])
        # candidates_idx_from = rng.integers(low=0, high=x.shape[0], size=x.shape[0]*n_repetitions)
        # FIXME
        candidates_idx_from = rng.integers(low=0, high=x.shape[0], size=x.shape[0])
        # print("candidates_idx_from.shape:", candidates_idx_from.shape)
        delta = rng.integers(low=1, high=x.shape[0]-1, size=candidates_idx_from.shape[0])
        # print("delta.shape:", delta.shape)
        candidates_idx_to = (candidates_idx_from + delta) % x.shape[0]
        # print("candidates_idx_to:", candidates_idx_to.shape)

        directions = x[candidates_idx_to, ...] - x[candidates_idx_from, ...]
        # print("directions.shape:", directions.shape)
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.clip(dists, a_min=self.dist_min, a_max=None)
        directions = directions / dists

        # dy = y[candidates_idx_to, :] - y[candidates_idx_from, :]

        # We always sample with replacement to avoid forcing to sample low densities
        # probabilities = self.weight_probabilities(dy, dists)

        # FIXME
        # compute the maximum over all changes in all y directions to sample good gradients for all outputs
        # gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()
        probabilities = np.ones_like(x[:,0]) / len(x[:,0])
        # print("dists.shape[0]:", dists.shape[0])
        # print("M:", self.M)
        # print("probabilities.shape:", probabilities.shape)
        selected_idx = rng.choice(dists.shape[0],
                                  size=self.M,
                                  replace=True,
                                  p=probabilities)

        if self.prune_duplicates:
            selected_idx = np.unique(selected_idx)
            self.n_pruned_neurons = self.M - len(selected_idx)
            self.M = len(selected_idx)

        directions = directions[selected_idx]
        dists = dists[selected_idx]
        idx_from = candidates_idx_from[selected_idx]
        idx_to = candidates_idx_to[selected_idx]

        return directions, dists, idx_from, idx_to


    def weight_probabilities(self, dy, dists):
        """Compute probability that a certain weight should be chosen as part of the network.
        This method computes all probabilities at once, without removing the new weights one by one.

        Args:
            dy: function difference
            dists: distance between the base points
            rng: random number generator

        Returns:
            probabilities: probabilities for the weights.
        """
        # gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()
        # When all gradients are small, avoind dividing by a small number
        # and default to uniform distribution.
        # probabilities = np.ones_like(gradients) / len(gradients)
        # compute the maximum over all changes in all y directions to sample good gradients for all outputs
        # return probabilities

    def solve_lstsq(self):
        (X,dX,x0,f0) = self.x_train
        # calculate dense layer derivative w.r.t. x => of shape (KD,M)
        d_activation_wrt_x = self.d_activation(np.subtract(X @ self.dense_layer_weights, self.dense_layer_biases)) # (K,M)
        phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * self.dense_layer_weights) for i in range(self.K)]) # (KD,M)

        # evaluate at x0
        x0 = x0.reshape(1, self.D)

        phi_1_of_x0 = self.activation(np.subtract(x0 @ self.dense_layer_weights, self.dense_layer_biases))
        # set up matrix A
        A =  np.vstack([
            phi_1_derivs,
            phi_1_of_x0
        ])
        # add the bias term to the weights
        bias_term = np.concatenate([np.zeros(phi_1_derivs.shape[0]), np.ones(phi_1_of_x0.shape[0])])
        A = np.column_stack([A, bias_term])
        # (KD + 1) x (M + 1)
        assert A.shape == (self.D*self.K + 1, self.M+ 1)

        # set up b (KD+1)
        b = np.concatenate([
            dX.flatten(), # [[x11,x12],[x21,x22],[x31,x32]...[xK1,xK2]] does match the way we build the matrix A
            f0.flatten()
        ])
        assert b.shape == (self.K*self.D + 1,)
        # solve the linear equations
        c = np.linalg.lstsq(A, b, rcond=self.regularization_scale)[0]

        assert c.shape == (self.M + 1,)

        self.linear_layer_weights = c[:-1].reshape((self.M, 1))
        self.linear_layer_biases = c[-1].reshape((1,))
