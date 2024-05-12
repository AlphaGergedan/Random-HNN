"""
src/utils/swim.py

Bridges the gap between SWIM and HNNs by implementing
the SWIM networks for solving Hamiltonian PDEs.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from swimnetworks.swimnetworks import Linear, Dense
from activations.utils import parse_activation


def backward(model, activation, x):
    """
    Gives derivatives of the model w.r.t x

    @param model        : should be a sklearn pipeline with hidden and linear layers
    @param activation   : string e.g. "relu", "tanh", "sigmoid", "elu", "identity"
    @param x            : input data (K, D)

    @returns dx         : derivatives of NN w.r.t x
    """
    # get dense layers and linear layer
    _, linear_layer = model.steps[-1]

    f_activation, df_activation = parse_activation(activation)

    # derivative of the last hidden layer w.r.t. input
    phi_L_derivs = compute_phi_L_derivs(model, f_activation, df_activation, x)

    # network derivative w.r.t. input
    phi_derivs = phi_L_derivs @ linear_layer.weights

    return phi_derivs.reshape(x.shape) # (K, D)


def compute_phi_L_derivs(model, f_activation, df_activation, x, x_next=None, integration_scheme=None):
    """
    Gives gradients of the last hidden layer output w.r.t x, of shape (KD,N_last)

    @param model        : sklearn Pipeline with hidden layers and last linear layer
    @param activation   : string e.g. "relu", "tanh", "sigmoid", "elu", "identity"
    @param x            : input data points in the phase space (K,D)
    @param x_next       : next state after a time step h of the input data (K,D),
                          if this is given then we use the finite differences method in the
                          hswim, use an integration scheme in the gradient computation
    @param integration_scheme
                        : either "forward-euler", "midpoint-rule", "symplectic-euler"

    @returns dx         : derivatives of hidden layer output w.r.t x of shape (KD,N_last)
    """
    # get dense and linear layer
    assert len(x.shape) == 2
    K,D = x.shape
    first_hidden_layer = model[0]

    # calculate first dense layer derivative w.r.t. x => of shape (KD,M) where M is the last hidden layer size
    first_hidden_layer.activation = df_activation

    # realizes a geometric integration scheme
    if x_next is not None:
        x = select_integration(x, x_next, integration_scheme)

    d_activation_wrt_x = first_hidden_layer.transform(x) # (K,N_1)
    first_hidden_layer.activation = f_activation

    phi_derivs = np.einsum('ij,kj->ikj', d_activation_wrt_x, first_hidden_layer.weights) # (K,D,N_1)

    # aggregate other layers, FIXME: adapt to element-wise multiplication as above
    current_hidden_layer = 2
    for hidden_layer in model[1:-1]:
        hidden_layer.activation = df_activation
        d_activation_wrt_x = model[:current_hidden_layer].transform(x) # (K,N)
        hidden_layer.activation = f_activation

        current_phi_derivs = np.einsum('ij,kj->ikj', d_activation_wrt_x, hidden_layer.weights) # (K,N_prev,N)

        # aggregate the derivatives: (K, D, N_prev) @ (K, N_prev, N) => (K, D, N)
        phi_derivs = phi_derivs @ current_phi_derivs

        current_hidden_layer += 1

    return phi_derivs.reshape(K*D, -1)


def select_integration(x, x_next, method):
    """
    Given x and the next state x_next with a time step h
    returns the constructed input which realizes the given integration
    scheme

    @param x:      current state (K,D)
    @param x_next: next state after a time step h (K,D)
    @param method: 'symplectic_euler', 'forward_euler', 'implicit_midpoint_rule'
    """
    (_,D) = x.shape
    assert x.shape == x_next.shape
    match method:
        case "symplectic_euler":
            # note that this realizes the symplectic euler with q_next and p
            # although other scheme also exists with q and p_next
            x = np.column_stack([x_next[:,:D//2], x[:, D//2:]])
        case "forward_euler":
            # x = np.column_stack([x[:,:D//2], x[:, D//2:]])
            x = x
        case "implicit_midpoint_rule":
            x = (x + x_next) / 2
        case _:
            # default to identity
            raise ValueError(f"Method {method} not implemented")
    return x


def fit_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs_true, f0_true, rcond, include_bias=True):
    """
    Fits the last layer of the model by solving least squares,
    builds the matrix A and vector b and solves the linear equation for x (weights)

    @param phi_1_derivs        : derivatives of hidden layer output w.r.t. input (K*D,M)
    @param phi_1_of_x0         : hidden layer output of x0 (1,M)
    @param y_train_derivs_true : derivatives of target function w.r.t. X (K*D)
    @param f0_true             : true function value at input x0
    @param rcond               : how approximately to solve the least squares
    @include_bias              : whether to include bias in the weights

    @returns                   : solved x (weights of the final linear layer)
    """
    # set up matrix A with shape (ND,M)
    A =  np.concatenate(( phi_1_derivs, phi_1_of_x0 ), axis=0)

    # set up b (ND+1)
    b = np.concatenate((
        y_train_derivs_true.ravel(), # [[x11,x12],[x21,x22],[x31,x32]...[xK1,xK2]] e.g. for D=2
        f0_true.ravel()
        ))

    if include_bias:
        # add the bias term to the weights (should be 0 for derivatives to not take it into account, but for phi_1_of_x0 it counts)
        # bias_term = np.concatenate([np.zeros(phi_1_derivs.shape[0]), np.ones(phi_1_of_x0.shape[0])])

        bias_term = np.zeros(A.shape[0])
        bias_term[-phi_1_of_x0.shape[0]:] = 1

        # (ND + 1, M + 1)
        A = np.column_stack((A, bias_term))

    # solve the linear equations (if bias is included then shape is (M+1,))
    c = np.linalg.lstsq(A, b, rcond=rcond)[0]
    return c.reshape(-1, 1) # final shape (M+1, 1) == [weights, bias] of shapes (M,1) and (1,1)

def hswim(x_train, y_train_derivs_true, x0, f0_true,
          n_hidden_layers, n_neurons, f_activation, df_activation, parameter_sampler, sample_uniformly, rcond, elm_bias_start, elm_bias_end,
          y_train_true=None, random_seed=1, include_bias=True, resample_duplicates=False, x_train_next=None, train_integration_scheme="forward_euler"):
    """
    Hamiltonian SWIM Implementation

    @param x_train             : train set for the Hamiltonian of shape (K,D) where number of trainin points is K,
                                 and DOF of the system is D/2 (q-position and p-momentum for a degree of freedom).
                                 So for n DOF system we should have D=2n dimension for the training set (q,p)
    @param y_train_true        : function values of x_train of shape (K,1), if given then usual SWIM is used to sample the network
    @param y_train_derivs_true : derivatives of the Hamiltonian w.r.t x_train, therefore of shape (K,D)
    @param x0                  : a reference point within the phase space of the target function
    @param f0_true             : true function value of a single point, can be any point, usually initial state of the system, used when
                                 setting the bias in the linear layer to shift the function to the correct values
    @param n_hidden            : number of hidden neurons in the hidden layer
    @param f_activation        : activation function used in the hidden layer
    @param df_activation       : derivative of the activation function used in the hidden layer, needed to calculate phi_1_derivs
    @param parameter_sampler   : either 'relu', 'tanh', 'random', sampling method of the weights in SWIM algorithm
    @param sample_uniformly    : if true, data point picking distribution is uniform, meaning that we have same probability of picking any data point when sampling the weights
                                 if false, initial guess is done using uniform sampling since we do not have access to function values in the training set, then
                                 we can use the first approximation to compute y_train_values and use it to define the data point picking distribution in the SWIM algorithm
                                 and rerun the approximation (A-SWIM)
    @param rcond               : how approximately to solve the least squares
    @param random_seed         : used in SWIM
    @param include_bias        : whether to include bias in the weights
    @param resample_duplicates : if set to True, then the parameters are resampled if they are not unique in the SWIM algorithm
    @param x_train_next        : whether the y_train_derivs_true are calculated using finite differences with x_next, without the knowledge of true
                                 vector fields or gradients. In this case we use an integration scheme in the gradient computation
    @param train_integration_scheme
                               : None, 'forward_euler', 'implicit_midpoint_euler', 'symplectic_euler'

    @returns                   : model (sklearn pipeline of the sampled network)
    """
    assert len(x_train.shape) == 2
    K,D = x_train.shape
    assert y_train_derivs_true.shape == (K,D)
    assert x0.shape == (1,D)
    assert f0_true.shape == (1,1)
    assert y_train_true is None or y_train_true.shape == (K,1)

    steps = []
    if y_train_true is None:
        # in the first approximation we always use uniform data point picking probability so sample_uniformly is set to True
        for k_layer in range(n_hidden_layers):
            # random_seed is set as 'random_seed + k_layer * 12345'
            steps.append((f"dense{k_layer+1}", Dense(layer_width=n_neurons[k_layer], activation=f_activation, elm_bias_start=elm_bias_start, elm_bias_end=elm_bias_end,
                                                     parameter_sampler=parameter_sampler, sample_uniformly=True, resample_duplicates=resample_duplicates, random_seed=random_seed + k_layer * 12345)))
    else:
        for k_layer in range(n_hidden_layers):
            # random_seed is set as 'random_seed + k_layer * 12345'
            steps.append((f"dense{k_layer+1}", Dense(layer_width=n_neurons[k_layer], activation=f_activation,
                                                     parameter_sampler=parameter_sampler, sample_uniformly=False, random_seed=random_seed + k_layer * 12345)))

    # add the last linear layer and build the model
    steps.append(("linear", Linear(regularization_scale=rcond)))
    model = Pipeline(steps)

    # sample hidden layer weights
    model[:-1].fit(x_train, y_train_true)

    phi_1_derivs = compute_phi_L_derivs(model, f_activation, df_activation, x_train, x_next=x_train_next, integration_scheme=train_integration_scheme)

    # evaluate at x0, transform till the last layer (including all hidden layers, not including the linear layer)
    phi_1_of_x0 = model[:-1].transform(x0)

    # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
    c = fit_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs_true, f0_true, rcond=rcond, include_bias=include_bias).reshape(-1,1)

    _, linear_layer = model.steps[-1]

    if include_bias:
        linear_layer.weights = c[:-1].reshape((-1,1))
        linear_layer.biases = c[-1].reshape((1,1))
    else:
        linear_layer.weights = c.reshape((-1,1))
        linear_layer.biases = np.zeros((1,1))

    # set the info for linear layer, this will be set only once
    linear_layer.layer_width = linear_layer.weights.shape[1]
    linear_layer.n_parameters = linear_layer.weights.size + linear_layer.biases.size

    # return in case of ELM or SWIM, A-SWIM (second step)
    if sample_uniformly or y_train_true is not None:
        return model

    # approximate the Hamiltonian values (target function values) which we need in other sampling methods
    y_train_pred = model.transform(x_train)

    # recursive call with the y_train_pred values
    return hswim(x_train, y_train_derivs_true, x0, f0_true,
                 n_hidden_layers, n_neurons, f_activation, df_activation, parameter_sampler, False, rcond, # False stands for sample_uniformly=False
                 elm_bias_start, elm_bias_end,
                 y_train_true=y_train_pred, random_seed=random_seed, include_bias=include_bias,
                 resample_duplicates=resample_duplicates, x_train_next=x_train_next, train_integration_scheme=train_integration_scheme)


def swim(x_train, y_train_true, n_hidden_layers, n_neurons, f_activation,
         parameter_sampler, sample_uniformly, rcond, elm_bias_start, elm_bias_end,
         random_seed=1):
    """
    SWIM Implementation for fitting the function directly using the true function values.
    This is a supervised method, and without solving the PDE, but directly fitting the function.

    @param x_train          : train set points in the phase space
    @param y_train_true     : true function values (true Hamiltonians)
    ...
    @param paramter_sampler : which parameter sampler to use in the SWIM algorithm,
                              in our case always 'tanh'
    @param sample_uniformly : whether to sample the data points uniformly in the SWIM algorithm
                              weight construction
    @param rcond            : regularization parameter for the lstsq
    @param elm_bias_*       : bias range of the ELM method, in ELM bias is sampled uniformly given this range
    @param random_seed      : for reproducibility

    @return fitted model
    """
    K,_ = x_train.shape # (K,D)
    assert y_train_true.shape == (K,1)
    assert len(n_neurons) == n_hidden_layers

    steps = []
    for k_layer in range(n_hidden_layers):
        # random_seed is set as 'random_seed + k_layer * 12345'
        steps.append((f"dense{k_layer+1}", Dense(layer_width=n_neurons[k_layer], activation=f_activation, elm_bias_start=elm_bias_start, elm_bias_end=elm_bias_end,
                                                 parameter_sampler=parameter_sampler, sample_uniformly=sample_uniformly, random_seed=random_seed + k_layer * 12345)))
    steps.append(("linear", Linear(regularization_scale=rcond)))
    model = Pipeline(steps=steps, verbose=False)
    model.fit(x_train, y_train_true)
    return model
