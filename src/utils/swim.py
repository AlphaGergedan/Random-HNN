# Import necessary modules

# common modules
import os
import sys
import numpy as np
from time import time
from sklearn.pipeline import Pipeline

# # local modules
# directory_to_prepend = os.path.abspath("..")
# if directory_to_prepend not in sys.path:
    # sys.path = [directory_to_prepend] + sys.path

from swimnetworks.swimnetworks import Linear, Dense
from activations.utils import parse_activation
from utils.plot import plot_2d

from error_functions.index import *

from utils.grid import generate_grid, generate_train_test_grid

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# PRIVATE

def compute_phi_1_derivs(model, f_activation, df_activation, x):
    """
    Gives derivatives of the model w.r.t x

    @param model        : sklearn Pipeline with hidden layers and last linear layer
    @param activation   : string e.g. "relu", "tanh", "sigmoid", "elu", "identity"
    @param x            : input data (K, D)

    @returns dx         : derivatives of hidden layer output w.r.t x
    """
    # get dense and linear layer
    assert len(x.shape) == 2
    K,D = x.shape
    first_hidden_layer = model[0]
    _, N_1 = first_hidden_layer.weights.shape

    # calculate first dense layer derivative w.r.t. x => of shape (KD,M) where M is the last hidden layer size
    first_hidden_layer.activation = df_activation
    d_activation_wrt_x = first_hidden_layer.transform(x) # (K,N_1)
    first_hidden_layer.activation = f_activation

    # the following stacks the derivatives in the matrix A for the linear system
    phi_derivs = np.empty((K*D, N_1))
    for i in range(K):
        # phi_1_derivs[i * D: (i+1)*D, :] = d_activation_wrt_x[i,:] * hidden_layer.weights
        phi_derivs[i*D:(i+1)*D, :] = d_activation_wrt_x[i, :] * first_hidden_layer.weights # (N_1) x (D,N_1)
    phi_derivs = phi_derivs.reshape((K,D,N_1))

    # aggregate other layers
    current_hidden_layer = 2
    for hidden_layer in model[1:-1]:
        input_size, output_size = hidden_layer.weights.shape

        hidden_layer.activation = df_activation
        d_activation_wrt_x = model[:current_hidden_layer].transform(x) # (K, output_size)
        hidden_layer.activation = f_activation

        current_phi_1_derivs = np.empty((K*input_size, output_size))
        print(f"computing derivs for layer {current_hidden_layer}")
        for i in range(K):
            current_phi_1_derivs[i*input_size:(i+1)*input_size, :] = d_activation_wrt_x[i, :] * hidden_layer.weights # (hidden_size) x (D,hidden_size)
        current_phi_1_derivs = current_phi_1_derivs.reshape((K, input_size, output_size))

        # aggregate the derivatives: (K, D, N_old) @ (K, N_old, N_new) => (K, D, N_new)

        print(f"aggregating result..")
        phi_derivs = phi_derivs @ current_phi_1_derivs

        current_hidden_layer += 1
        raise RuntimeError('Support for more layers is not tested yet. If you want to use it anyways then remove this line.')

    # d_activation_wrt_x * hidden_layer.weights.T == (K,M)x(M,D) = (K,D)

    # phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(d_activation_wrt_x.shape[0])]) # (KD,M)

    # perform element-wise multiplication of the arrays
    # phi_1_derivs = d_activation_wrt_x[:, np.newaxis, :] * hidden_layer.weights[np.newaxis, :, :]

    return phi_derivs.reshape(K*D, -1)



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

# PUBLIC

def hswim(x_train, y_train_derivs_true, x0, f0_true,
          n_hidden_layers, n_neurons, f_activation, df_activation, parameter_sampler, sample_uniformly, rcond,
          y_train_true=None, random_seed=1, include_bias=True):
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

    @returns                   : model (sklearn pipeline of the sampled shallow network)
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
            steps.append((f"dense{k_layer+1}", Dense(layer_width=n_neurons[k_layer], activation=f_activation,
                                                     parameter_sampler=parameter_sampler, sample_uniformly=True, random_seed=random_seed + k_layer * 12345)))
    else:
        for k_layer in range(n_hidden_layers):
            # random_seed is set as 'random_seed + k_layer * 12345'
            steps.append((f"dense{k_layer+1}", Dense(layer_width=n_neurons[k_layer], activation=f_activation,
                                                     parameter_sampler=parameter_sampler, sample_uniformly=False, random_seed=random_seed + k_layer * 12345)))

    # add the last linear layer and build the model
    steps.append(("linear", Linear(regularization_scale=rcond)))
    model = Pipeline(steps)

    # sample hidden layer weights
    print(f"model has size {len(model)}")
    model[:-1].fit(x_train, y_train_true)

    phi_1_derivs = compute_phi_1_derivs(model, f_activation, df_activation, x_train)

    # evaluate at x0, transform till the last layer (including all hidden layers, not including the linear layer)
    phi_1_of_x0 = model[:-1].transform(x0)

    # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
    print("fitting linear layer inside hswim..")
    t_start = time()
    c = fit_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs_true, f0_true, rcond=rcond, include_bias=include_bias).reshape(-1,1)
    t_end = time()
    print(f"fitting linear layer inside hswim took {t_end-t_start} seconds")

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
                 n_hidden_layers, n_neurons, f_activation, df_activation, parameter_sampler, False, rcond,
                 y_train_true=y_train_pred, random_seed=random_seed, include_bias=True)


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

    # TODO
    # hidden_layers = [hidden_layer for _, hidden_layer in model.steps[:-1]]
    # _, linear_layer = model.steps[1][1]

    f_activation, df_activation = parse_activation(activation)

    phi_1_derivs = compute_phi_1_derivs(model, f_activation, df_activation, x)

    # linear layer
    phi_2_derivs = phi_1_derivs @ linear_layer.weights # avoid bias! it has no effect on derivative w.r.t x

    return phi_2_derivs.reshape(x.shape) # (K, D)










############### CLEAN THE CODE BELOW

def swim(x_train, y_train_true, n_hidden_layers, n_neurons, f_activation,
         parameter_sampler, sample_uniformly, rcond, random_seed=1):
    """
    SWIM Implementation
    """
    K,D = x_train.shape
    assert y_train_true.shape == (K,1)
    assert len(n_neurons) == n_hidden_layers
    assert y_train_true is not None

    steps = []
    for k_layer in range(n_hidden_layers):
        # random_seed is set as 'random_seed + k_layer * 12345'
        steps.append((f"dense{k_layer+1}", Dense(layer_width=n_neurons[k_layer], activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=sample_uniformly, random_seed=random_seed + k_layer * 12345)))
    steps.append(("linear", Linear(regularization_scale=rcond)))
    model = Pipeline(steps=steps, verbose=False)
    model.fit(x_train, y_train_true)

    return model

def approximate_hamiltonian(
        # dataset training
        x_train, y_train_derivs_true, x0, f0,
        # model parameters
        n_hidden, f_activation, df_activation, parameter_sampler, sample_uniformly, rcond, random_seed=1, include_bias=True,
        ):
    """
    Given hamiltonian related input, predicts the Hamiltonian function of the dynamical system

    Training set == {x_train, y_train_derivs_true, x0, f0}
    TODO: Evaluation set == {x_test_values}

    - x_train: data points of the hamiltonian system, for n DOF system we should have 2n dimension for the training set (q,p)
    - y_train_derivs_true: derivatives of the hamiltonian w.r.t. x_train
    - x0: a reference point for the hamiltonian, can be any point, usually the initial state of the system
    - f0: Hamiltonian value at x0
    - n_hidden: number of hidden nodes
    - f_activation: activation function of the hidden layer
    - df_activation: derivative of the activation function of the hidden layer
    - parameter_sampler: either 'relu', 'tanh', 'random', sampling method of the weights in SWIM algorithm
    - sample_uniformly: if true, data point picking distribution is uniform, meaning that we have same probability of picking any data point when sampling the weights
                        if false, initial guess is done using uniform sampling since we do not have access to function values in the training set, then
                                  we use the first approximation to compute y_train_values and use it to define the data point picking distribution in the SWIM algorithm
                                  and rerun the approximation
    - rcond: regularization parameter for the linear layer in the least square solution
    - random_seed: random seed for reproducibility
    - include_bias: whether to include bias in the weights
    """
    # number of data points and features
    K,D = x_train.shape
    assert (K,D) == y_train_derivs_true.shape

    # build the pipeline for having dense + linear layer set up, in the first approximation we always use uniform data point picking probability so sample_uniformly is set to True
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=True, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    # model_ansatz.fit(x_train, np.ones((K, 1))) # output has one feature dimension
    # use SWIM algorithm with uniform data point picking probability to sample the hidden layer weights
    model_ansatz.fit(x_train)

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train) # (K,M)
    # the following stacks the derivatives in the matrix A for the linear system
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)

    # evaluate at x0
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)

    # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
    c = fit_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs_true, f0, rcond=rcond, include_bias=include_bias).reshape(-1,1)

    if include_bias:
        linear_layer.weights = c[:-1].reshape((-1,1))
        linear_layer.biases = c[-1].reshape((1,1))
    else:
        linear_layer.weights = c.reshape((-1,1))
        linear_layer.biases = np.zeros((1,1))

    # set the info for linear layer, this will be set only once
    linear_layer.layer_width = linear_layer.weights.shape[1]
    linear_layer.n_parameters = np.prod(linear_layer.weights.shape) + np.prod(linear_layer.biases.shape)

    # return the model if uniform data picking probability is used
    if sample_uniformly:
        return model_ansatz

    # approximate the Hamiltonian values (target function values) which we need in other sampling methods
    y_train_hat = model_ansatz.transform(x_train)

    # build the pipeline for having dense + linear layer set up, in the first approximation we always use uniform data point picking probability so sample_uniformly is set to True
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=False, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    # model_ansatz.fit(x_train, np.ones((K, 1))) # output has one feature dimension
    # use SWIM algorithm with uniform data point picking probability to sample the hidden layer weights
    model_ansatz.fit(x_train, y_train_hat)

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train) # (K,M)
    # the following stacks the derivatives in the matrix A for the linear system
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)

    # evaluate at x0
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)

    # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
    c = fit_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs_true, f0, rcond=rcond, include_bias=include_bias).reshape(-1,1)

    if include_bias:
        linear_layer.weights = c[:-1].reshape((-1,1))
        linear_layer.biases = c[-1].reshape((1,1))
    else:
        linear_layer.weights = c.reshape((-1,1))
        linear_layer.biases = np.zeros((1,1))

    # set the info for linear layer, this will be set only once
    linear_layer.layer_width = linear_layer.weights.shape[1]
    linear_layer.n_parameters = np.prod(linear_layer.weights.shape) + np.prod(linear_layer.biases.shape)

    return model_ansatz



def model_layer_params(model):
    return [ step[1].n_parameters for step in model.steps ]


def validate(model, domain_params, model_params, verbose=False):
    # prepare the validation data
    *_, q_val_grids, p_val_grids = generate_grid(domain_params["V_qs"], domain_params["V_ps"], domain_params["q_val_lims"], domain_params["p_val_lims"], domain_params["dof"])
    x_val = np.column_stack([ q_val_grid.flatten() for q_val_grid in q_val_grids ] + [ p_val_grid.flatten() for p_val_grid in p_val_grids ])
    del q_val_grids, p_val_grids

    t_start = time()
    y_val_true = domain_params["H"](x_val)
    t_end = time()
    print("evaluating H(x_val) took:", t_end - t_start, "seconds")

    t_start = time()
    y_val_pred = model.transform(x_val)
    t_end = time()
    print("evaluating H_hat(x_val) took:", t_end - t_start, "seconds")

    val_mse_error = mean_squared_error(y_val_true, y_val_pred)
    val_l2_error_relative = l2_error_relative(y_val_true, y_val_pred)

    val_name_tokens = [domain_params["system_name"], "V_q1" + str(domain_params["V_qs"][0]), "V_p1" + str(domain_params["V_ps"][0]), "q1_v_lim" + str(domain_params["q_val_lims"][0]), "p1_v_lim" + str(domain_params["p_val_lims"][0]) + "noise" + str(domain_params["noise"])]

    del x_val

    if verbose:
        model_name_tokens = [ k + str(v) for k, v in model_params.items() ]
        print("VAL RESULTS")
        print("-> val mse error on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_mse_error))
        print("-> val l2 relative error on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_l2_error_relative))

    return val_mse_error, val_l2_error_relative


def fit(domain_params, model_params, verbose=False, save=False):
    """
    @param domain_params: {
                            "dof": degree of freedom of the system, must be in [1,2,3,4],
                            "K_qs": number of grid points for q in each dimension as a list used in fitting,
                            "K_ps": number of grid points for p in each dimension as a list used in fitting,
                            "q_train_lims": list of [q_train_min, q_train_max] in each dimension
                            "p_train_lims": list of [p_train_min, p_train_max] in each dimension
                          }
    """
    # prepare the train data (X, dX, x0, f0) of dimension ( (sum(K_qs), 2*dof), (sum(K_qs), 2*dof), (1,2*dof), (1) )
    *_, q_train_grids, p_train_grids = generate_grid(domain_params["K_qs"], domain_params["K_ps"], domain_params["q_train_lims"], domain_params["p_train_lims"], domain_params["dof"])
    x_train = np.column_stack([ q_train_grid.flatten() for q_train_grid in q_train_grids ] + [ p_train_grid.flatten() for p_train_grid in p_train_grids ])
    del q_train_grids, p_train_grids

    y_train_derivs_true = domain_params["dH"](x_train)
    x0 = np.array(np.zeros(2*domain_params["dof"]))
    f0 = domain_params["H"](x0.reshape(-1,2*domain_params["dof"]))

    f_activation, df_activation = parse_activation(model_params["activation"])
    t_start = time()
    model = approximate_hamiltonian(x_train, y_train_derivs_true, x0, f0, model_params["M"], f_activation, df_activation, model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
    t_end = time()

    del y_train_derivs_true, x0, f0

    model_name_tokens = [ k + str(v) for k, v in model_params.items() ]

    train_mse_error = mean_squared_error(domain_params["H"](x_train), model.transform(x_train))
    train_l2_error_relative = l2_error_relative(domain_params["H"](x_train), model.transform(x_train))
    del x_train

    train_name_tokens = [domain_params["system_name"], "K_q1" + str(domain_params["K_qs"][0]), "K_p1" + str(domain_params["K_ps"][0]), "q1_t_lim" + str(domain_params["q_train_lims"][0]), "p1_t_lim" + str(domain_params["p_train_lims"][0]) + "noise" + str(domain_params["noise"])]

    model_params = model_layer_params(model)

    if verbose:
        print("TRAIN RESULTS")
        print("-> train mse error on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_mse_error))
        print("-> train l2 relative error on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_l2_error_relative))
        print("took", t_end - t_start, "seconds to fit")
        print("model size in bytes", sys.getsizeof(model))
        print("number of params in layers:", str(model_params), "in total is", sum(model_params))
        print()

    if save:
        # TODO save the model in /models
        pass

    return model, train_mse_error, train_l2_error_relative, t_end-t_start, model_name_tokens




def experiment_2d_hamiltonian_system(domain_params, model_params, verbose=False, save=False):
    """
    Evaluates the given model in the given 2d domain

    @param domain_params:   {
                                "system_name": name of the system, e.g. "single_pendulum"
                                "H": hamiltonian function of the system
                                "dH": derivative of H w.r.t. q and p
                                "N_q", "N_p", "q_test_lim", "p_test_lim": test data parameters
                                "V_q", "V_p", "q_val_lim", "p_val_lim": validation data parameters
                                "K_q", "K_p", "q_train_lim", "p_train_lim": training data parameters
                                "random_seed": seed for reproducibility
                                "noise": noise level in the data
                            }
    @param model_params:    {
                                "name": discriminative name for the model
                                "M": hidden nodes
                                "activation": activation function, should be string e.g. "relu", "tanh", "sigmoid"
                                "parameter_sampler": weight sampling strategy
                                "sample_uniformly": whether to use uniform distribution for data point picking when sampling the weights
                                                    this must be set to True for ELM and U-SWIM; and False for A-SWIM
                                "rcond": regularization in lstsq in the linear layer
                                "random_seed": for reproducability
                                "include_bias": whether to include bias in linear layer
                                "clip": whether to clip to min/max values when plotting
                            }
    @param verbose: whether to plot the validation and test approximations
    @param save: whether to save the plots

    @returns train_errors, train_losses, test_errors, test_losses, train_time
    """
    # if model name is H, then we plot the ground truth for validation and test
    if model_params["name"] == "H":
        # plot the ground truth
        [q_plot_range], [p_plot_range], [q_plot_grid], [p_plot_grid] = generate_grid([domain_params["q_plot"]], [domain_params["p_plot"]], [domain_params["q_plot_lim"]], [domain_params["p_plot_lim"]], 1, linspace=True)
        x_plot = np.column_stack([q_plot_grid.flatten(), p_plot_grid.flatten()])
        plot_name_tokens = [domain_params["system_name"], "q" + str(domain_params["q_plot"]), "p" + str(domain_params["p_plot"]), "q_lim" + str(domain_params["q_plot_lim"]), "p_lim" + str(domain_params["p_plot_lim"])]
        # q diff / p diff
        q_diff = domain_params["q_plot_lim"][1] - domain_params["q_plot_lim"][0]
        p_diff = domain_params["p_plot_lim"][1] - domain_params["p_plot_lim"][0]
        if q_diff == p_diff:
            aspect = 1
        else:
            aspect = 2.5
        if save:
            file_location = "../../plots/" + "_".join(plot_name_tokens) + "GROUND_TRUTH" + ".pdf"
        else:
            file_location = '' # which defaults to not saving the plot
        plot_2d(domain_params["H"](x_plot).reshape((domain_params["p_plot"],domain_params["q_plot"])), q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Ground Truth", verbose=verbose, save=file_location)
        del q_plot_range, p_plot_range, q_plot_grid, p_plot_grid, x_plot

        return { "mean_absolute_error": 0, "mean_squared_error": 0, "l2_error": 0, "l2_error_relative": 0 }, { "mean_absolute_error": 0, "mean_squared_error": 0, "l2_error": 0, "l2_error_relative": 0 }, { "mean_absolute_error": 0, "mean_squared_error": 0, "l2_error": 0, "l2_error_relative": 0 }, { "mean_absolute_error": 0, "mean_squared_error": 0, "l2_error": 0, "l2_error_relative": 0 }, 0

    # first we train the model with the train data (X, dX, x0, f0) then evaluate
    train_rng = np.random.default_rng(domain_params["train_random_seed"])
    test_rng = np.random.default_rng(domain_params["test_random_seed"])

    # repeat 10 times and take the average
    train_errors = [] # measures performance of H_hat against ground truth H with training data
    test_errors = [] # measures performance of H_hat against ground truth H with test data
    train_losses = [] # measures performance of dH_hat against ground truth dH with training data
    test_losses = [] # measures performance of dH_hat against ground truth dH with test data
    train_times = []
    for _ in range(domain_params["repeat"]):
        [q_train_range], [p_train_range], [q_train_grid], [p_train_grid], [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_train_test_grid([domain_params["q_train"]], [domain_params["p_train"]], [domain_params["q_train_lim"]], [domain_params["p_train_lim"]],[domain_params["q_test"]], [domain_params["p_test"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], test_rng=test_rng, dof=1, linspace=domain_params["training_set_linspaced"], train_rng=train_rng)
        x_train = np.column_stack([q_train_grid.flatten(), p_train_grid.flatten()])
        y_train_derivs_true = domain_params["dH"](x_train)
        x0 = np.array([0,0])
        f0 = domain_params["H"](x0.reshape(-1,2))

        f_activation, df_activation = parse_activation(model_params["activation"])

        if model_params["name"] == "SWIM":
            t_start = time()
            print("SWIM")
            model = swim(x_train, domain_params["H"](x_train), y_train_derivs_true, x0, f0, model_params["M"], f_activation, df_activation, model_params["parameter_sampler"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
            t_end = time()
        else:
            t_start = time()
            model = approximate_hamiltonian(x_train, y_train_derivs_true, x0, f0, model_params["M"], f_activation, df_activation, model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
            t_end = time()

        train_times.append(t_end-t_start)

        ##############################################################  PLOT #######################################################

        if verbose:
            # for plotting resample validation set linearly spaced
            [q_plot_range], [p_plot_range], [q_plot_grid], [p_plot_grid] = generate_grid([domain_params["q_plot"]], [domain_params["p_plot"]], [domain_params["q_plot_lim"]], [domain_params["p_plot_lim"]], dof=1, linspace=True)
            print(f'q_plot_range shape {q_plot_range.shape}')
            print(f'p_plot_range shape {p_plot_range.shape}')
            print(f'q_plot_grid shape {q_plot_grid.shape}')
            print(f'p_plot_grid shape {p_plot_grid.shape}')
            x_plot  = np.column_stack([q_plot_grid.flatten(), p_plot_grid.flatten()]) # (N,2)
            y_plot_true = domain_params["H"](x_plot).reshape(-1) # (N)
            y_plot_pred = model.transform(x_plot).reshape(-1) # (N)
            neuron_number = 0

            # plot where the first training point is placed in the first hidden neuron
            hidden_layer = model.steps[0][1]
            print(f"avg weight: {np.mean(hidden_layer.weights)}")
            print(f"avg bias: {np.mean(hidden_layer.biases)}")
            x_plot_activation = hidden_layer.transform(x_plot)
            print(f'avg. activation: {np.mean(x_plot_activation)}')
            print(f'avg. activation for first 8 neurons:')
            print(f'Neuron 1: {np.mean(x_plot_activation[:,0])}')
            print(f'Neuron 2: {np.mean(x_plot_activation[:,1])}')
            print(f'Neuron 3: {np.mean(x_plot_activation[:,2])}')
            print(f'Neuron 4: {np.mean(x_plot_activation[:,3])}')
            print(f'Neuron 5: {np.mean(x_plot_activation[:,4])}')
            print(f'Neuron 6: {np.mean(x_plot_activation[:,5])}')
            print(f'Neuron 7: {np.mean(x_plot_activation[:,6])}')
            print(f'Neuron 8: {np.mean(x_plot_activation[:,7])}')
            print(f'x_plot_activation shape {x_plot_activation.shape}')
            activation_plot = x_plot_activation[:,neuron_number].reshape((domain_params["p_plot"],domain_params["q_plot"]))
            print(f'activation_plot shape {activation_plot.shape}')

            # q diff / p diff
            q_diff = domain_params["q_plot_lim"][1] - domain_params["q_plot_lim"][0]
            p_diff = domain_params["p_plot_lim"][1] - domain_params["p_plot_lim"][0]
            if q_diff == p_diff:
                aspect = 1
            else:
                aspect = 2.5

            # here plot the approximation
            y_plot = y_plot_pred.reshape((domain_params["p_plot"],domain_params["q_plot"]))
            plot_2d(y_plot, q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, verbose=True)

            # here plot the real function to see where the function changes quickly
            y_plot = y_plot_true.reshape((domain_params["p_plot"],domain_params["q_plot"]))
            # ACTIVATION PLOT
            plt.clf()
            x = np.linspace(-2, 2, 400)
            # sigmoid
            inverse_sigmoid = (lambda x: np.log((1 / (x)) - 1))
            inverse_tanh = (lambda x: np.arctanh(x))
            y = f_activation(x)
            plt.grid()
            plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line at y=0
            plt.axvline(0, color='black', linewidth=0.5)  # Vertical line at x=0
            plt.plot(x, y)
            # arctanh is the inverse of tanh, shows where the point is placed in the activation function
            plt.plot(inverse_tanh(x_plot_activation[:][neuron_number]), np.zeros_like(x_plot_activation[:][neuron_number]), c="r", marker="o", alpha=0.2, markersize=3)
            # plt.plot(np.arctanh(x_plot_activation[:][1]), np.zeros_like(x_plot_activation[:][1]), c="r", marker="o", alpha=0.2, markersize=3)
            plt.title('Neuron ' + str(neuron_number + 1))
            plt.show()
            plt.clf()

            # 2D PLOT
            fig = plt.figure()
            ax = plt.gca()
            im = ax.imshow(y_plot, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"])
            ax.contour(q_plot_range, p_plot_range, y_plot, 10, colors='white', linewidths=0.5)
            sc = ax.scatter(q_plot_grid, p_plot_grid, c=activation_plot, cmap='coolwarm')
            cbar_im = plt.colorbar(im, ax=ax, fraction=0.049*(p_plot_range.shape[0] / q_plot_range.shape[0]), pad=0.04)
            cbar_sc = plt.colorbar(sc, ax=ax, fraction=0.049*(p_plot_range.shape[0] / q_plot_range.shape[0]), pad=0.04)
            plt.subplots_adjust(right=0.8)
            cbar_sc.set_ticks([-1, -0.5, 0, 0.5, 1])
            cbar_im.ax.set_position([0.85, 0.15, 0.02, 0.7])
            ax.set_aspect(aspect)
            plt.title('Neuron ' + str(neuron_number + 1))
            plt.show()
            plt.clf()

            # 3D PLOT
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf_function = ax.plot_surface(q_plot_grid, p_plot_grid, y_plot, cmap=cm.viridis, linewidth=0, antialiased=False)
            surf_activation = ax.plot_surface(q_plot_grid, p_plot_grid, activation_plot, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            cbar = fig.colorbar(surf_activation, shrink=0.5, aspect=5)
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            plt.show()
            plt.clf()

            # 2D PLOT ACTIVATION STDDEV
            fig = plt.figure()
            ax = plt.gca()
            im = ax.imshow(y_plot, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"])
            ax.contour(q_plot_range, p_plot_range, y_plot, 10, colors='white', linewidths=0.5)
            sc = ax.scatter(q_plot_grid, p_plot_grid, c=np.std(x_plot_activation, axis=1).reshape((domain_params["p_plot"],domain_params["q_plot"])), cmap='coolwarm')
            cbar_im = plt.colorbar(im, ax=ax, fraction=0.049*(p_plot_range.shape[0] / q_plot_range.shape[0]), pad=0.04)
            cbar_sc = plt.colorbar(sc, ax=ax, fraction=0.049*(p_plot_range.shape[0] / q_plot_range.shape[0]), pad=0.04)
            plt.subplots_adjust(right=0.8)
            # cbar_sc.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar_im.ax.set_position([0.85, 0.15, 0.02, 0.7])
            ax.set_aspect(aspect)
            plt.title('Deviation of Activation')
            plt.show()
            plt.clf()

            # 3D PLOT ACTIVATION STDDEV
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf_function = ax.plot_surface(q_plot_grid, p_plot_grid, y_plot, cmap=cm.viridis, linewidth=0, antialiased=False)
            surf_activation = ax.plot_surface(q_plot_grid, p_plot_grid, np.std(x_plot_activation, axis=1).reshape((domain_params["p_plot"],domain_params["q_plot"])), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            fig.colorbar(surf_activation, shrink=0.5, aspect=5)
            plt.show()
            plt.clf()

            x_plot_logits = inverse_tanh(x_plot_activation)
            # replace inf values with max min possible values
            x_plot_logits[np.isinf(x_plot_logits)] = 18.714973875118524
            x_plot_logits[np.isneginf(x_plot_logits)] = -18.714973875118524
            print(f'x_plot_logits {x_plot_logits.shape}')
            print(f'whether includes inf: {np.any(x_plot_logits == np.inf)}')
            print(f'whether includes -inf: {np.any(x_plot_logits == -np.inf)}')
            print(f'whether includes nan: {np.any(x_plot_logits == np.nan)}')
            print(f'max value is: {np.max(x_plot_logits)}')
            print(f'min value is: {np.min(x_plot_logits)}')

            # 2D PLOT STD on logits
            print(f"stddev on logits: {np.std(x_plot_logits.reshape(-1))}")
            fig = plt.figure()
            ax = plt.gca()
            im = ax.imshow(y_plot, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"])
            ax.contour(q_plot_range, p_plot_range, y_plot, 10, colors='white', linewidths=0.5)
            sc = ax.scatter(q_plot_grid, p_plot_grid, c=np.std(x_plot_logits, axis=1).reshape((domain_params["p_plot"],domain_params["q_plot"])), cmap='coolwarm')
            cbar_im = plt.colorbar(im, ax=ax, fraction=0.049*(p_plot_range.shape[0] / q_plot_range.shape[0]), pad=0.04)
            cbar_sc = plt.colorbar(sc, ax=ax, fraction=0.049*(p_plot_range.shape[0] / q_plot_range.shape[0]), pad=0.04)
            plt.subplots_adjust(right=0.8)
            # cbar_sc.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar_im.ax.set_position([0.85, 0.15, 0.02, 0.7])
            ax.set_aspect(aspect)
            plt.title('Deviation of Logits')
            plt.show()
            plt.clf()

            # 3D PLOT STD on logits
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf_function = ax.plot_surface(q_plot_grid, p_plot_grid, y_plot, cmap=cm.viridis, linewidth=0, antialiased=False)
            surf_activation = ax.plot_surface(q_plot_grid, p_plot_grid, np.std(x_plot_logits, axis=1).reshape((domain_params["p_plot"],domain_params["q_plot"])), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # Customize the z axis.
            # ax.set_zlim(-1.01, 1.01)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # # A StrMethodFormatter is used automatically
            # ax.zaxis.set_major_formatter('{x:.02f}')
            # Add a color bar which maps values to colors.
            fig.colorbar(surf_activation, shrink=0.5, aspect=5)
            plt.show()
            plt.clf()















        # print(f"Has shape.. {y_plot.shape}") # (100,240), need (100, 240, )
        # plot_2d(y_plot, q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="", verbose=True)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # img = ax.scatter(q_plot_range, p_plot_range, np.ones_like(q_plot_range), c=y_hat, cmap=plt.hot())
        # fig.colorbar(img)
        # plt.show()
        # plt.imshow(y_plot, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"])
        # plt.clf()


        # fig = plt.figure()
        # ax = plt.gca()
        # im = ax.imshow(grid, extent=extent, vmin=vmin, vmax=vmax)
        # ax.contour(xrange, yrange, grid, contourlines, colors='white', linewidths=0.5)
        # fig.colorbar(im, ax=ax, fraction=0.049*(yrange.shape[0] / xrange.shape[0]), pad=0.04)
        # ax.set_ylabel(ylabel)
        # ax.set_xlabel(xlabel)
        # ax.set_aspect(aspect)
        # plt.show()
        # plt.show()
        # plt.clf()


        ##############################################################  PLOT #######################################################

        y_train_derivs_pred = backward(model, model_params["activation"], x_train)
        train_losses.append([mean_absolute_error(y_train_derivs_true, y_train_derivs_pred), mean_squared_error(y_train_derivs_true, y_train_derivs_pred), l2_error(y_train_derivs_true, y_train_derivs_pred), l2_error_relative(y_train_derivs_true, y_train_derivs_pred)])
        y_true = domain_params["H"](x_train)
        y_pred = model.transform(x_train)
        train_errors.append([mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), l2_error(y_true, y_pred), l2_error_relative(y_true, y_pred)])

        x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])

        y_test_derivs_true = domain_params["dH"](x_test)
        y_test_derivs_pred = backward(model, model_params["activation"], x_test)
        test_losses.append([mean_absolute_error(y_test_derivs_true, y_test_derivs_pred), mean_squared_error(y_test_derivs_true, y_test_derivs_pred), l2_error(y_test_derivs_true, y_test_derivs_pred), l2_error_relative(y_test_derivs_true, y_test_derivs_pred)])

        y_true = domain_params["H"](x_test).reshape(-1)
        y_pred = model.transform(x_test).reshape(-1)
        test_errors.append([mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), l2_error(y_true, y_pred), l2_error_relative(y_true, y_pred)])

    # if verbose: FIXME remove after iterating runs
    if False:
        # run one more time to plot
        [q_train_range], [p_train_range], [q_train_grid], [p_train_grid], [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_train_test_grid([domain_params["q_train"]], [domain_params["p_train"]], [domain_params["q_train_lim"]], [domain_params["p_train_lim"]],[domain_params["q_test"]], [domain_params["p_test"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], test_rng=test_rng, dof=1, linspace=domain_params["training_set_linspaced"], train_rng=train_rng)
        x_train = np.column_stack([q_train_grid.flatten(), p_train_grid.flatten()])
        y_train_derivs_true = domain_params["dH"](x_train)
        x0 = np.array([0,0])
        f0 = domain_params["H"](x0.reshape(-1,2))

        f_activation, df_activation = parse_activation(model_params["activation"])

        model = approximate_hamiltonian(x_train, y_train_derivs_true, x0, f0, model_params["M"], f_activation, df_activation, model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
        hidden_layer = model.steps[0][1]

        x_train_activation = hidden_layer.transform(x_train)
        y_train_derivs_pred = backward(model, model_params["activation"], x_train)
        y_train_pred = model.transform(x_train)

        x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])

        x_test_activation = hidden_layer.transform(x_test)
        y_test_derivs_pred = backward(model, model_params["activation"], x_test)
        y_test_true = domain_params["H"](x_test).reshape(-1)

        # for plotting resample validation set linearly spaced
        [q_plot_range], [p_plot_range], [q_plot_grid], [p_plot_grid] = generate_grid([domain_params["q_plot"]], [domain_params["p_plot"]], [domain_params["q_plot_lim"]], [domain_params["p_plot_lim"]], dof=1, linspace=True)
        x_plot  = np.column_stack([q_plot_grid.flatten(), p_plot_grid.flatten()])
        y = domain_params["H"](x_plot).reshape(-1)
        y_hat = model.transform(x_plot).reshape(-1)

        plot_name_tokens = [domain_params["system_name"], "q" + str(domain_params["q_plot"]), "p" + str(domain_params["p_plot"]), "q_lim" + str(domain_params["q_plot_lim"]), "p_plot_lim" + str(domain_params["p_plot_lim"]) + "noise" + str(domain_params["noise"])]
        model_name_tokens = [ k + str(v) for k, v in model_params.items()]

        # q diff / p diff
        q_diff = domain_params["q_plot_lim"][1] - domain_params["q_plot_lim"][0]
        p_diff = domain_params["p_plot_lim"][1] - domain_params["p_plot_lim"][0]
        if q_diff == p_diff:
            aspect = 1
        else:
            aspect = 2.5

        if save:
            file_location = "../../plots/" + "_".join(plot_name_tokens) + "_" + "_".join(model_name_tokens) + ".pdf"
        else:
            file_location = '' # which defaults to not saving the plot

        if model_params["clip"]:
            y_plot = y_hat.reshape((domain_params["p_plot"],domain_params["q_plot"])).clip(np.min(y), np.max(y))
        else:
            y_plot = y_hat.reshape((domain_params["p_plot"],domain_params["q_plot"]))

        plot_2d(y_plot, q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)
        plot_2d((y-y_hat).reshape((domain_params["p_plot"],domain_params["q_plot"])), q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Error, "+" ".join(model_name_tokens), verbose=verbose, save=file_location)

        # plot the direction of the weights


    # mean of
    # train_errors, train_losses, test_errors, test_losses, train_times
    return {
            "mean_absolute_error": np.mean(np.asarray(train_errors)[:,0]),
             "mean_squared_error": np.mean(np.asarray(train_errors)[:,1]),
             "l2_error": np.mean(np.asarray(train_errors)[:,2]),
             "l2_error_relative": np.mean(np.asarray(train_errors)[:,3])
           }, {
            "mean_absolute_error": np.mean(np.asarray(train_losses)[:,0]),
            "mean_squared_error": np.mean(np.asarray(train_losses)[:,1]),
            "l2_error": np.mean(np.asarray(train_losses)[:,2]),
            "l2_error_relative": np.mean(np.asarray(train_losses)[:,3])
           }, {
            "mean_absolute_error": np.mean(np.asarray(test_errors)[:,0]),
            "mean_squared_error": np.mean(np.asarray(test_errors)[:,1]),
            "l2_error": np.mean(np.asarray(test_errors)[:,2]),
            "l2_error_relative": np.mean(np.asarray(test_errors)[:,3])
           }, {
            "mean_absolute_error": np.mean(np.asarray(test_losses)[:,0]),
            "mean_squared_error": np.mean(np.asarray(test_losses)[:,1]),
            "l2_error": np.mean(np.asarray(test_losses)[:,2]),
            "l2_error_relative": np.mean(np.asarray(test_losses)[:,3])
           }, np.mean(np.asarray(train_times))

    [q_train_range], [p_train_range], [q_train_grid], [p_train_grid], [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_train_test_grid([domain_params["q_train"]], [domain_params["p_train"]], [domain_params["q_train_lim"]], [domain_params["p_train_lim"]],[domain_params["q_test"]], [domain_params["p_test"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], test_rng=test_rng, dof=1, linspace=domain_params["training_set_linspaced"], train_rng=train_rng)
    x_train = np.column_stack([q_train_grid.flatten(), p_train_grid.flatten()])
    y_train_derivs_true = domain_params["dH"](x_train)
    x0 = np.array([0,0])
    f0 = domain_params["H"](x0.reshape(-1,2))

    f_activation, df_activation = parse_activation(model_params["activation"])

    t_start = time()
    model = approximate_hamiltonian(x_train, y_train_derivs_true, x0, f0, model_params["M"], f_activation, df_activation, model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
    t_end = time()

    model_name_tokens = [ k + str(v) for k, v in model_params.items()]

    y = domain_params["H"](x_train)
    y_hat = model.transform(x_train)
    train_mse_error = MSE(y, y_hat)
    train_l2_error_relative = l2_error_relative(y, y_hat)

    train_name_tokens = [domain_params["system_name"], "train", "q" + str(domain_params["q_train"]), "p" + str(domain_params["p_train"]), "q_lim" + str(domain_params["q_train_lim"]), "p_lim" + str(domain_params["p_train_lim"]) + "noise" + str(domain_params["noise"])]

    if verbose:
        print("-------------")
        print("TRAIN RESULTS")
        print("-> train mse error on domain:\n", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_mse_error))
        print("-> train l2 relative error on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_l2_error_relative))
        print("-------------")

    # TODO plot where the inputs are placed in the activation function, e.g. 3d plot where we see the output of activation functions

    del q_train_range, p_train_range, q_train_grid, p_train_grid, x_train, y_train_derivs_true, x0, f0

    # test set, sampled distinct from training set
    x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])

    y = domain_params["H"](x_test).reshape(-1)
    y_hat = model.transform(x_test).reshape(-1)
    assert y.shape == y_hat.shape
    test_mse_error = mean_squared_error(y, y_hat)
    test_l2_error_relative = l2_error_relative(y, y_hat)

    test_name_tokens = [domain_params["system_name"], "test", "q" + str(domain_params["q_test"]), "p" + str(domain_params["p_test"]), "q_lim" + str(domain_params["q_test_lim"]), "p_test_lim" + str(domain_params["p_test_lim"]) + "noise" + str(domain_params["noise"])]

    if verbose:
        print("-------------")
        print("TEST RESULTS")
        print("-> val mse error on domain:\n", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_mse_error))
        print("-> val l2 relative error on domain\n:", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_l2_error_relative))
        print("-------------")

    # skip plotting if not verbose
    if not verbose:
        return { "mse": train_mse_error, "l2_error_relative": train_l2_error_relative }, { "mse": test_mse_error, "l2_error_relative": test_l2_error_relative }, t_end-t_start

    # for plotting resample validation set linearly spaced
    [q_plot_range], [p_plot_range], [q_plot_grid], [p_plot_grid] = generate_grid([domain_params["q_plot"]], [domain_params["p_plot"]], [domain_params["q_plot_lim"]], [domain_params["p_plot_lim"]], dof=1, linspace=True)
    x_plot  = np.column_stack([q_plot_grid.flatten(), p_plot_grid.flatten()])
    y = domain_params["H"](x_plot).reshape(-1)
    y_hat = model.transform(x_plot).reshape(-1)

    # q diff / p diff
    q_diff = domain_params["q_plot_lim"][1] - domain_params["q_plot_lim"][0]
    p_diff = domain_params["p_plot_lim"][1] - domain_params["p_plot_lim"][0]
    if q_diff == p_diff:
        aspect = 1
    else:
        aspect = 2.5

    plot_name_tokens = [domain_params["system_name"], "q" + str(domain_params["q_plot"]), "p" + str(domain_params["p_plot"]), "q_lim" + str(domain_params["q_plot_lim"]), "p_plot_lim" + str(domain_params["p_plot_lim"]) + "noise" + str(domain_params["noise"])]

    if save:
        file_location = "../../plots/" + "_".join(plot_name_tokens) + "_" + "_".join(model_name_tokens) + ".pdf"
    else:
        file_location = '' # which defaults to not saving the plot

    if model_params["clip"]:
        y_plot = y_hat.reshape((domain_params["p_plot"],domain_params["q_plot"])).clip(np.min(y), np.max(y))
    else:
        y_plot = y_hat.reshape((domain_params["p_plot"],domain_params["q_plot"]))

    plot_2d(y_plot, q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)
    plot_2d((y-y_hat).reshape((domain_params["p_plot"],domain_params["q_plot"])), q_plot_range, p_plot_range, extent=domain_params["q_plot_lim"] + domain_params["p_plot_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Error, "+" ".join(model_name_tokens), verbose=verbose, save=file_location)

    # plot the direction of the weights

    del q_plot_range, p_plot_range, q_plot_grid, p_plot_grid, x_plot, y_plot, y, y_hat

    # early retrurn to avoid test set
    return { "mse": train_mse_error, "l2_error_relative": train_l2_error_relative }, { "mse": test_mse_error, "l2_error_relative": test_l2_error_relative }, t_end-t_start

    # test with a broader range as the train and val data
    [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_grid([domain_params["N_q"]], [domain_params["N_p"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], 1, linspace=True, random_seed=domain_params["random_seed"])
    x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])

    y = domain_params["H"](x_test).reshape(-1)
    y_hat = model.transform(x_test).reshape(-1)
    assert y.shape == y_hat.shape
    test_mse_error = mean_squsred_error(y, y_hat)
    test_l2_error_relative = l2_error_relative(y, y_hat)

    test_name_tokens = [domain_params["system_name"], "N_q" + str(domain_params["N_q"]), "N_p" + str(domain_params["N_p"]), "q_test_lim" + str(domain_params["q_test_lim"]), "p_test_lim" + str(domain_params["p_test_lim"]) + "noise" + str(domain_params["noise"])]

    if verbose:
        print("TEST RESULTS")
        print("-> test mse error on domain:", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_mse_error))
        print("-> test l2 relative error on domain:", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_l2_error_relative))

    # q diff / p diff
    q_diff = domain_params["q_test_lim"][1] - domain_params["q_test_lim"][0]
    p_diff = domain_params["p_test_lim"][1] - domain_params["p_test_lim"][0]
    if q_diff == p_diff:
        aspect = 1
    else:
        aspect = 2.5

    if save:
        file_location = "../../plots/" + "_".join(test_name_tokens) + "_" + "_".join(model_name_tokens) + "_test_l2_" + "{:.6f}".format(test_l2_error_relative) + ".pdf"
    else:
        file_location = '' # which defaults to not saving the plot

    if model_params["clip"]:
        y_plot = model.transform(x_test).reshape((domain_params["N_p"],domain_params["N_q"])).clip(np.min(y), np.max(y))
    else:
        y_plot = model.transform(x_test).reshape((domain_params["N_p"],domain_params["N_q"]))

    plot_2d(y_plot, q_test_range, p_test_range, extent=domain_params["q_test_lim"] + domain_params["p_test_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)
    plot_2d((y-y_hat).reshape((domain_params["N_p"],domain_params["N_q"])), q_test_range, p_test_range, extent=domain_params["q_test_lim"] + domain_params["p_test_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="TESTERROR"+" ".join(model_name_tokens), verbose=verbose, save=file_location)

    del q_test_range, p_test_range, q_test_grid, p_test_grid, x_test, y_plot

    return (train_mse_error,  train_l2_error_relative), (val_mse_error,  val_l2_error_relative), (test_mse_error, test_l2_error_relative), t_end - t_start
