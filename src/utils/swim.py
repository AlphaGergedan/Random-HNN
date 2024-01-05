import numpy as np
from activations import Activations
from sklearn.pipeline import Pipeline
from swimnetworks import Linear, Dense

def parse_activation(activation):
    """
    Given activation function string, returns the activation function and its derivative
    """
    match activation:
        case "relu":
            return Activations.relu, Activations.d_relu
        case "leaky_relu":
            return Activations.leaky_relu, Activations.d_leaky_relu
        case "elu":
            return Activations.elu, Activations.d_elu
        case "tanh":
            return Activations.tanh, Activations.d_tanh
        case "sigmoid":
            return Activations.sigmoid, Activations.d_sigmoid
        case _:
            # default to identity
            return Activations.identity, Activations.d_identity

def approximate_linear_layer(phi_1_derivs, phi_1_of_x0, x_train_derivs, f0, rcond, bias=True):
    """
    phi_1_derivs  : derivatives of hidden layer output w.r.t. input (N*D,M)
    phi_1_of_x0   : hidden layer output of x0 (1,M)
    x_train_derivs: derivatives of target function w.r.t. X (N*D)
    f0            : output of the target function with input x0 (1,1)
    bias          : whether to include bias in the weights

    Builds the matrix A and vector b and solves the linear equation for w
    """
    # set up matrix A with shape (ND,M)
    A =  np.vstack([
        phi_1_derivs,
        phi_1_of_x0
    ])
    # set up b (ND+1)
    b = np.concatenate([
        x_train_derivs.flatten(), # [[x11,x12],[x21,x22],[x31,x32]...[xK1,xK2]]
        f0
    ])

    if bias:
        # add the bias term to the weights
        bias_term = np.concatenate([np.zeros(phi_1_derivs.shape[0]), np.ones(phi_1_of_x0.shape[0])])
        # (ND + 1, M + 1)
        A = np.column_stack([A, bias_term])

    # solve the linear equations (if bias is included then shape is (M+1,))
    c = np.linalg.lstsq(A, b, rcond=rcond)[0]
    return c


# solving using swim
def solve_swim_hamiltonian(
        # dataset training
        x_train_values, x_train_derivs, x0, f0,
        # model parameters
        n_hidden, activation, parameter_sampler, sample_uniformly, rcond, random_seed=1, include_bias=True
        ):
    """
    Given hamiltonian related input, predicts the Hamiltonian function of the dynamical system
    parameter_sampler must be from the set { 'relu', 'tanh', 'random' }
    """
    f_activation, df_activation = parse_activation(activation)

    # number of data points and features
    K,D = x_train_values.shape
    assert (K,D) == x_train_derivs.shape

    # build the pipeline for having dense + linear layer set up
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=sample_uniformly, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    # set up the linear system to solve the outer coefficients
    model_ansatz.fit(x_train_values, np.ones((K, 1))) # output has one feature dimension

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train_values) # (K,M)
    # the following stacks the derivatives in the matrix A explained as above
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)

    # evaluate at x0
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)

    c = approximate_linear_layer(phi_1_derivs, phi_1_of_x0, x_train_derivs, f0, rcond=rcond, bias=include_bias)

    if include_bias:
        linear_layer.weights = c[:-1]
        linear_layer.biases = c[-1]
    else:
        linear_layer.weights = c
        linear_layer.biases = 0

    return model_ansatz

def weight_sampling(
        # dataset training
        x_train_values, x_train_derivs, x0, f0,
        # model parameters
        n_hidden, activation, parameter_sampler, rcond, random_seed=1,
        include_bias=True
        ):
    """
    Given hamiltonian related input, predicts the Hamiltonian function of the dynamical system
    parameter_sampler must be from the set { 'relu', 'tanh', 'random' }
    """
    f_activation, df_activation = parse_activation(activation)

    # number of data points and features
    K,D = x_train_values.shape
    assert (K,D) == x_train_derivs.shape

    # # build the pipeline for having ELM for the first approximation of H(x)
    # model_ansatz = Pipeline([
        # ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler='random', sample_uniformly=True, random_seed=random_seed)),
        # ("linear", Linear(regularization_scale=rcond))
    # ])

    # build the pipeline for initial approximation with random data point picking probabilty
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=True, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    # set up the linear system to solve the outer coefficients
    model_ansatz.fit(x_train_values, x_train_derivs[:,0])

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train_values) # (K,M)
    # the following stacks the derivatives in the matrix A explained as above
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)

    # evaluate at x0
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)

    # STEP 1: approximate the target function with uniform sampling of the weights
    c = approximate_linear_layer(phi_1_derivs, phi_1_of_x0, x_train_derivs, f0, rcond=rcond, bias=include_bias)

    if include_bias:
        linear_layer.weights = c[:-1]
        linear_layer.biases = c[-1]
    else:
        linear_layer.weights = c
        linear_layer.biases = 0

    # STEP 2: predict
    y_train_values_pred = model_ansatz.transform(x_train_values)

    # STEP 3: approximate target function using the weight distribution
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=False, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    # set up the linear system to solve the outer coefficients
    # use weight probabilities with the distances of the predicted function values
    model_ansatz.fit(x_train_values, y_train_values_pred)

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train_values) # (K,M)
    # the following stacks the derivatives in the matrix A explained as above
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)

    # evaluate at x0
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)

    # STEP 1: approximate the target function with uniform sampling of the weights
    c = approximate_linear_layer(phi_1_derivs, phi_1_of_x0, x_train_derivs, f0, rcond=rcond, bias=include_bias)

    if include_bias:
        linear_layer.weights = c[:-1]
        linear_layer.biases = c[-1]
    else:
        linear_layer.weights = c
        linear_layer.biases = 0

    return model_ansatz
