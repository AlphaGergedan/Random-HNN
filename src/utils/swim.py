# Import necessary modules

# common modules
import os
import sys
import numpy as np
from time import time
from sklearn.pipeline import Pipeline

# local modules
directory_to_prepend = os.path.abspath("..")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from swimnetworks.swimnetworks import Linear, Dense
from activations import relu, tanh, sigmoid, elu, identity
from utils.plot import plot_2d
from utils.mse import MSE
from utils.l2_error_relative import l2_error_relative
from utils.grid import generate_grid


def model_layer_params(model):
    return [ step[1].n_parameters for step in model.steps ]


def validate(model, domain_params, model_params, verbose=False):
    # prepare the validation data
    *_, q_val_grids, p_val_grids = generate_grid(domain_params["V_qs"], domain_params["V_ps"], domain_params["q_val_lims"], domain_params["p_val_lims"], domain_params["dof"])
    x_val = np.column_stack([ q_val_grid.flatten() for q_val_grid in q_val_grids ] + [ p_val_grid.flatten() for p_val_grid in p_val_grids ])
    del q_val_grids, p_val_grids

    t_start = time()
    y_val = domain_params["H"](x_val)
    t_end = time()
    print("evaluating H(x_val) took:", t_end - t_start, "seconds")

    t_start = time()
    y_val_hat = model.transform(x_val)
    t_end = time()
    print("evaluating H_hat(x_val) took:", t_end - t_start, "seconds")

    val_mse_loss = MSE(y_val, y_val_hat)
    val_l2_error_relative = l2_error_relative(y_val, y_val_hat)

    val_name_tokens = [domain_params["system_name"], "V_q1" + str(domain_params["V_qs"][0]), "V_p1" + str(domain_params["V_ps"][0]), "q1_v_lim" + str(domain_params["q_val_lims"][0]), "p1_v_lim" + str(domain_params["p_val_lims"][0]) + "noise" + str(domain_params["noise"])]

    del x_val

    if verbose:
        model_name_tokens = [ k + str(v) for k, v in model_params.items() ]
        print("VAL RESULTS")
        print("-> val mse loss on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_mse_loss))
        print("-> val l2 relative error on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_l2_error_relative))

    return val_mse_loss, val_l2_error_relative


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

    y_train_derivs = domain_params["dH"](x_train)
    x0 = np.array(np.zeros(2*domain_params["dof"]))
    f0 = domain_params["H"](x0.reshape(-1,2*domain_params["dof"]))

    t_start = time()
    model = approximate_hamiltonian(x_train, y_train_derivs, x0, f0, model_params["M"], model_params["activation"], model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
    t_end = time()

    del y_train_derivs, x0, f0

    model_name_tokens = [ k + str(v) for k, v in model_params.items() ]

    train_mse_loss = MSE(domain_params["H"](x_train), model.transform(x_train))
    train_l2_error_relative = l2_error_relative(domain_params["H"](x_train), model.transform(x_train))
    del x_train

    train_name_tokens = [domain_params["system_name"], "K_q1" + str(domain_params["K_qs"][0]), "K_p1" + str(domain_params["K_ps"][0]), "q1_t_lim" + str(domain_params["q_train_lims"][0]), "p1_t_lim" + str(domain_params["p_train_lims"][0]) + "noise" + str(domain_params["noise"])]

    model_params = model_layer_params(model)

    if verbose:
        print("TRAIN RESULTS")
        print("-> train mse loss on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_mse_loss))
        print("-> train l2 relative error on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_l2_error_relative))
        print("took", t_end - t_start, "seconds to fit")
        print("model size in bytes", sys.getsizeof(model))
        print("number of params in layers:", str(model_params), "in total is", sum(model_params))
        print()

    if save:
        # TODO save the model in /models
        pass

    return model, train_mse_loss, train_l2_error_relative, t_end-t_start, model_name_tokens


def parse_activation(activation):
    """
    Given activation function string, returns the activation function and its derivative
    """
    match activation:
        case "relu":
            return relu.relu, relu.d_relu
        case "leaky_relu":
            return relu.leaky_relu, relu.d_leaky_relu
        case "elu":
            return elu.elu, elu.d_elu
        case "tanh":
            return tanh.tanh, tanh.d_tanh
        case "sigmoid":
            return sigmoid.sigmoid, sigmoid.d_sigmoid
        case _:
            # default to identity
            return identity.identity, identity.d_identity

def approximate_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs, f0, rcond, bias=True):
    """
    phi_1_derivs  : derivatives of hidden layer output w.r.t. input (N*D,M)
    phi_1_of_x0   : hidden layer output of x0 (1,M)
    y_train_derivs: derivatives of target function w.r.t. X (N*D)
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
        y_train_derivs.flatten(), # [[x11,x12],[x21,x22],[x31,x32]...[xK1,xK2]]
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
def approximate_hamiltonian(
        # dataset training
        x_train, y_train_derivs, x0, f0,
        # model parameters
        n_hidden, activation, parameter_sampler, sample_uniformly, rcond, random_seed=1, include_bias=True,
        ):
    """
    Given hamiltonian related input, predicts the Hamiltonian function of the dynamical system

    Training set == {x_train, y_train_derivs, x0, f0}
    TODO: Evaluation set == {x_test_values}

    - x_train: data points of the hamiltonian system, for n DOF system we should have 2n dimension for the training set (q,p)
    - y_train_derivs: derivatives of the hamiltonian w.r.t. x_train
    - x0: a reference point for the hamiltonian, can be any point, usually the initial state of the system
    - f0: Hamiltonian value at x0
    - n_hidden: number of hidden nodes
    - activation: activation function of the hidden layer
    - parameter_sampler: either 'relu', 'tanh', 'random', sampling method of the weights in SWIM algorithm
    - sample_uniformly: if true, data point picking distribution is uniform, meaning that we have same probability of picking any data point when sampling the weights
                        if false, initial guess is done using uniform sampling since we do not have access to function values in the training set, then
                                  we use the first approximation to compute y_train_values and use it to define the data point picking distribution in the SWIM algorithm
                                  and rerun the approximation
    - rcond: regularization parameter for the linear layer in the least square solution
    - random_seed: random seed for reproducibility
    - include_bias: whether to include bias in the weights
    """
    f_activation, df_activation = parse_activation(activation)

    # number of data points and features
    K,D = x_train.shape
    assert (K,D) == y_train_derivs.shape
    # TODO: assert (K,D) == x_test_values

    # build the pipeline for having dense + linear layer set up, in the first approximation we always use uniform data point picking probability so sample_uniformly is set to True
    t_start = time()
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=True, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])
    t_end = time()

    print("--> pipeline built with dense + linear layers in:", t_end - t_start, "seconds")

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    # model_ansatz.fit(x_train, np.ones((K, 1))) # output has one feature dimension
    # use SWIM algorithm with uniform data point picking probability to sample the hidden layer weights
    t_start = time()
    model_ansatz.fit(x_train)
    t_end = time()
    print("--> hidden layer weights+biases sampled using random sampling of the data points in:", t_end - t_start, "seconds")

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    t_start = time()
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train) # (K,M)
    # the following stacks the derivatives in the matrix A for the linear system
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)
    t_end = time()
    print("--> derivatives w.r.t. inputs are taken in:", t_end - t_start, "seconds")


    # evaluate at x0
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)

    # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
    t_start = time()
    c = approximate_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs, f0, rcond=rcond, bias=include_bias).reshape(-1,1)
    t_end = time()
    print("--> linear layer approximated by solving linear system in:", t_end - t_start, "seconds")

    if include_bias:
        linear_layer.weights = c[:-1]
        linear_layer.biases = c[-1]
    else:
        linear_layer.weights = c
        linear_layer.biases = np.zeros_like((1,1))

    # set the info for linear layer, this will be set only once
    linear_layer.layer_width = linear_layer.weights.shape[1]
    linear_layer.n_parameters = np.prod(linear_layer.weights.shape) + np.prod(linear_layer.biases.shape)

    # return the model if uniform data picking probability is used
    if sample_uniformly:
        return model_ansatz

    # approximate the Hamiltonian values (target function values) which we need in other sampling methods
    t_start = time()
    y_train_hat = model_ansatz.transform(x_train)
    t_end = time()
    print("y_train_hat calculated using first pipeline in:", t_end - t_start, "seconds")

    # build the pipeline for having dense + linear layer set up, this is the second approximation with weight probabilities, so sample_uniformly is set to False
    model_ansatz = Pipeline([
        ("dense", Dense(layer_width=n_hidden, activation=f_activation, parameter_sampler=parameter_sampler, sample_uniformly=False, random_seed=random_seed)),
        ("linear", Linear(regularization_scale=rcond))
    ])
    print("*--> second pipeline built with dense + linear layers in:", t_end - t_start, "seconds")

    # get dense and linear layer
    hidden_layer = model_ansatz.steps[0][1]
    linear_layer = model_ansatz.steps[1][1]

    t_start = time()
    # set up the linear system to solve the outer coefficients
    # use weight probabilities with the distances of the predicted function values
    model_ansatz.fit(x_train, y_train_hat)
    t_end = time()
    print("*--> hidden layer weights+biases sampled using weight probabiliities using the approximated y_train_hat of the data points in:", t_end - t_start, "seconds")

    # calculate dense layer derivative w.r.t. x => of shape (KD,M)
    hidden_layer.activation = df_activation
    d_activation_wrt_x = hidden_layer.transform(x_train) # (K,M)
    # the following stacks the derivatives in the matrix A explained as above
    phi_1_derivs = np.row_stack([(d_activation_wrt_x[i,:] * hidden_layer.weights) for i in range(K)]) # (KD,M)

    # evaluate at x0
    t_start = time()
    hidden_layer.activation = f_activation
    x0 = x0.reshape(1, D)
    phi_1_of_x0 = hidden_layer.transform(x0)
    t_end = time()
    print("*--> derivatives w.r.t. inputs are taken in:", t_end - t_start, "seconds")

    # STEP 1: approximate the target function with uniform sampling of the weights
    t_start = time()
    c = approximate_linear_layer(phi_1_derivs, phi_1_of_x0, y_train_derivs, f0, rcond=rcond, bias=include_bias).reshape(-1,1)
    t_end = time()
    print("*--> linear layer approximated by solving linear system in:", t_end - t_start, "seconds")

    if include_bias:
        linear_layer.weights = c[:-1]
        linear_layer.biases = c[-1]
    else:
        linear_layer.weights = c
        linear_layer.biases = np.zeros_like((1,1))

    return model_ansatz

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

    @returns train_losses, val_losses, test_losses
    """
    # if model name is H, then we plot the ground truth for validation and test
    if model_params["name"] == "H":
        N = domain_params["N_q"] * domain_params["N_p"]
        V = domain_params["V_q"] * domain_params["V_p"]
        # validate with a broader data in the same range as the training data
        [q_val_range], [p_val_range], [q_val_grid], [p_val_grid] = generate_grid([domain_params["V_q"]], [domain_params["V_p"]], [domain_params["q_val_lim"]], [domain_params["p_val_lim"]], 1)
        x_val = np.column_stack([q_val_grid.flatten(), p_val_grid.flatten()])
        val_name_tokens = [domain_params["system_name"], "V_q" + str(domain_params["V_q"]), "V_p" + str(domain_params["V_p"]), "q_val_lim" + str(domain_params["q_val_lim"]), "p_val_lim" + str(domain_params["p_val_lim"])]
        # q diff / p diff
        q_diff = domain_params["q_val_lim"][1] - domain_params["q_val_lim"][0]
        p_diff = domain_params["p_val_lim"][1] - domain_params["p_val_lim"][0]
        if q_diff == p_diff:
            aspect = 1
        else:
            aspect = 2.5
        if save:
            file_location = "../../plots/" + "_".join(val_name_tokens) + "GROUND_TRUTH" + ".pdf"
        else:
            file_location = '' # which defaults to not saving the plot
        plot_2d(domain_params["H"](x_val).reshape((domain_params["V_p"],domain_params["V_q"])), q_val_range, p_val_range, extent=domain_params["q_val_lim"] + domain_params["p_val_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Ground Truth", verbose=verbose, save=file_location)
        del q_val_range, p_val_range, q_val_grid, p_val_grid, x_val


        # test with a broader range as the train and val data
        [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_grid([domain_params["N_q"]], [domain_params["N_p"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], 1)
        x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])
        test_name_tokens = [domain_params["system_name"], "N_q" + str(domain_params["N_q"]), "N_p" + str(domain_params["N_p"]), "q_test_lim" + str(domain_params["q_test_lim"]), "p_test_lim" + str(domain_params["p_test_lim"])]
        # q diff / p diff
        q_diff = domain_params["q_test_lim"][1] - domain_params["q_test_lim"][0]
        p_diff = domain_params["p_test_lim"][1] - domain_params["p_test_lim"][0]
        if q_diff == p_diff:
            aspect = 1
        else:
            aspect = 2.5
        if save:
            file_location = "../../plots/" + "_".join(test_name_tokens) + "GROUND_TRUTH" + ".pdf"
        else:
            file_location = '' # which defaults to not saving the plot

        plot_2d(domain_params["H"](x_test).reshape((domain_params["N_p"],domain_params["N_q"])), q_test_range, p_test_range, extent=domain_params["q_test_lim"] + domain_params["p_test_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Ground Truth", verbose=verbose, save=file_location)
        del q_test_range, p_test_range, q_test_grid, p_test_grid, x_test
        return 0,0,0,0


    # calculate domain sizes
    N = domain_params["N_q"] * domain_params["N_p"]
    V = domain_params["V_q"] * domain_params["V_p"]
    K = domain_params["K_q"] * domain_params["K_p"]

    # first we train the model with the train data (X, dX, x0, f0)
    [q_train_range], [p_train_range], [q_train_grid], [p_train_grid] = generate_grid([domain_params["K_q"]], [domain_params["K_p"]], [domain_params["q_train_lim"]], [domain_params["p_train_lim"]], 1)
    x_train = np.column_stack([q_train_grid.flatten(), p_train_grid.flatten()])
    y_train_derivs = domain_params["dH"](x_train)
    x0 = np.array([0,0])
    f0 = domain_params["H"](x0.reshape(-1,2))

    t_start = time()
    model = approximate_hamiltonian(x_train, y_train_derivs, x0, f0, model_params["M"], model_params["activation"], model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
    t_end = time()

    model_name_tokens = [ k + str(v) for k, v in model_params.items()]

    train_mse_loss = MSE(domain_params["H"](x_train), model.transform(x_train))
    train_l2_error_relative = l2_error_relative(domain_params["H"](x_train), model.transform(x_train))

    train_name_tokens = [domain_params["system_name"], "K_q" + str(domain_params["K_q"]), "K_p" + str(domain_params["K_p"]), "q_train_lim" + str(domain_params["q_train_lim"]), "p_train_lim" + str(domain_params["p_train_lim"]) + "noise" + str(domain_params["noise"])]

    if verbose:
        print("TRAIN RESULTS")
        print("-> train mse loss on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_mse_loss))
        print("-> train l2 relative error on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_l2_error_relative))
        print()

    del q_train_range, p_train_range, q_train_grid, p_train_grid, x_train, y_train_derivs, x0, f0

    # validate with a broader data in the same range as the training data
    [q_val_range], [p_val_range], [q_val_grid], [p_val_grid] = generate_grid([domain_params["V_q"]], [domain_params["V_p"]], [domain_params["q_val_lim"]], [domain_params["p_val_lim"]], 1)
    x_val = np.column_stack([q_val_grid.flatten(), p_val_grid.flatten()])

    val_mse_loss = MSE(domain_params["H"](x_val), model.transform(x_val))
    val_l2_error_relative = l2_error_relative(domain_params["H"](x_val), model.transform(x_val))

    val_name_tokens = [domain_params["system_name"], "V_q" + str(domain_params["V_q"]), "V_p" + str(domain_params["V_p"]), "q_val_lim" + str(domain_params["q_val_lim"]), "p_val_lim" + str(domain_params["p_val_lim"]) + "noise" + str(domain_params["noise"])]

    if verbose:
        print("VAL RESULTS")
        print("-> val mse loss on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_mse_loss))
        print("-> val l2 relative error on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_l2_error_relative))

    # q diff / p diff
    q_diff = domain_params["q_val_lim"][1] - domain_params["q_val_lim"][0]
    p_diff = domain_params["p_val_lim"][1] - domain_params["p_val_lim"][0]
    if q_diff == p_diff:
        aspect = 1
    else:
        aspect = 2.5

    if save:
        file_location = "../../plots/" + "_".join(val_name_tokens) + "_" + "_".join(model_name_tokens) + "_train_l2_" + "{:.6f}".format(train_l2_error_relative) + "_val_l2_" + "{:.6f}".format(val_l2_error_relative) + ".pdf"
    else:
        file_location = '' # which defaults to not saving the plot

    if model_params["clip"]:
        y_plot = model.transform(x_val).reshape((domain_params["V_p"],domain_params["V_q"])).clip(np.min(domain_params["H"](x_val)), np.max(domain_params["H"](x_val)))
    else:
        y_plot = model.transform(x_val).reshape((domain_params["V_p"],domain_params["V_q"]))

    plot_2d(y_plot, q_val_range, p_val_range, extent=domain_params["q_val_lim"] + domain_params["p_val_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)

    del q_val_range, p_val_range, q_val_grid, p_val_grid, x_val, y_plot

    # test with a broader range as the train and val data
    [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_grid([domain_params["N_q"]], [domain_params["N_p"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], 1)
    x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])

    test_mse_loss = MSE(domain_params["H"](x_test), model.transform(x_test))
    test_l2_error_relative = l2_error_relative(domain_params["H"](x_test), model.transform(x_test))

    test_name_tokens = [domain_params["system_name"], "N_q" + str(domain_params["N_q"]), "N_p" + str(domain_params["N_p"]), "q_test_lim" + str(domain_params["q_test_lim"]), "p_test_lim" + str(domain_params["p_test_lim"]) + "noise" + str(domain_params["noise"])]

    if verbose:
        print("TEST RESULTS")
        print("-> test mse loss on domain:", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_mse_loss))
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
        y_plot = model.transform(x_test).reshape((domain_params["N_p"],domain_params["N_q"])).clip(np.min(domain_params["H"](x_test)), np.max(domain_params["H"](x_test)))
    else:
        y_plot = model.transform(x_test).reshape((domain_params["N_p"],domain_params["N_q"]))

    plot_2d(y_plot, q_test_range, p_test_range, extent=domain_params["q_test_lim"] + domain_params["p_test_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)

    del q_test_range, p_test_range, q_test_grid, p_test_grid, x_test, y_plot

    return train_l2_error_relative, val_l2_error_relative, test_l2_error_relative, t_end - t_start


# def experiment_4d_hamiltonian_system(domain_params, model_params, verbose=False, save=False):
    # """
    # Evaluates the given model in the given 2d domain
#
    # @param domain_params:   {
                                # "system_name": name of the system, e.g. "double_pendulum"
                                # "H": hamiltonian function of the system
                                # "dH": derivative of H w.r.t. q and p
                                # "N_q1", "N_q2", "N_p1", "N_p2", "q1_test_lim", "q2_test_lim", "p1_test_lim", "p2_test_lim": test data parameters
                                # "V_q1", "V_q2", "V_p1", "V_p2", "q1_val_lim", "q2_val_lim", "p1_val_lim", "p2_val_lim": val validation data parameters
                                # "K_q1", "K_q2", "K_p1", "K_p2", "q1_train_lim", "q2_train_lim", "p1_train_lim", "p2_train_lim": training data parameters
                                # "random_seed": seed for reproducibility
                                # "noise": noise level in the data
                            # }
    # @param model_params:    {
                                # "name": discriminative name for the model
                                # "M": hidden nodes
                                # "activation": activation function, should be string e.g. "relu", "tanh", "sigmoid"
                                # "parameter_sampler": weight sampling strategy
                                # "sample_uniformly": whether to use uniform distribution for data point picking when sampling the weights
                                                    # this must be set to True for ELM and U-SWIM; and False for A-SWIM
                                # "rcond": regularization in lstsq in the linear layer
                                # "random_seed": for reproducability
                                # "include_bias": whether to include bias in linear layer
                                # "clip": whether to clip to min/max values when plotting
                            # }
    # @param poincare_params: {[
                                #
                             #
                            #
                            # ]}
    # @param verbose: whether to plot the validation and test approximations
    # @param save: whether to save the plots
#
    # @returns train_losses, val_losses, test_losses
    # """
    # # if model name is H, then we plot the ground truth for validation and test
    # if model_params["name"] == "H":
        # N = domain_params["N_q1"] * domain_params["N_q2"] * domain_params["N_p1"] * domain_params["N_p2"]
        # V = domain_params["V_q1"] * domain_params["V_q2"] * domain_params["V_p1"] * domain_params["V_p2"]
        # # validate with a broader data in the same range as the training data
        # q1_val_range, q2_val_range, p1_val_range, p2_val_range, q1_val_grid, q2_val_grid, p1_val_grid, p2_val_grid = generate_grid_4d(
            # domain_params["V_q1"], domain_params["V_q2"], domain_params["V_p1"], domain_params["V_p2"],
            # domain_params["q1_val_lim"], domain_params["q2_val_lim"], domain_params["p1_val_lim"], domain_params["p2_val_lim"]
        # )
        # x_val = np.column_stack([q1_val_grid.flatten(), q2_val_grid.flatten(), p1_val_grid.flatten(), p2_val_grid.flatten()])
        # val_name_tokens = [domain_params["system_name"], "V_q" + str(domain_params["V_q"]), "V_p" + str(domain_params["V_p"]), "q_val_lim" + str(domain_params["q_val_lim"]), "p_val_lim" + str(domain_params["p_val_lim"])]
        # # q diff / p diff
        # q_diff = domain_params["q_val_lim"][1] - domain_params["q_val_lim"][0]
        # p_diff = domain_params["p_val_lim"][1] - domain_params["p_val_lim"][0]
        # if q_diff == p_diff:
            # aspect = 1
        # else:
            # aspect = 2.5
        # if save:
            # file_location = "../../plots/" + "_".join(val_name_tokens) + "GROUND_TRUTH" + ".pdf"
        # else:
            # file_location = '' # which defaults to not saving the plot
        # plot_2d(domain_params["H"](x_val).reshape((domain_params["V_p"],domain_params["V_q"])), q_val_range, p_val_range, extent=domain_params["q_val_lim"] + domain_params["p_val_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Ground Truth", verbose=verbose, save=file_location)
        # del q_val_range, p_val_range, q_val_grid, p_val_grid, x_val
#
#
        # # test with a broader range as the train and val data
        # q_test_range, p_test_range, q_test_grid, p_test_grid = generate_grid_2d(domain_params["N_q"], domain_params["N_p"], domain_params["q_test_lim"], domain_params["p_test_lim"])
        # x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])
        # test_name_tokens = [domain_params["system_name"], "N_q" + str(domain_params["N_q"]), "N_p" + str(domain_params["N_p"]), "q_test_lim" + str(domain_params["q_test_lim"]), "p_test_lim" + str(domain_params["p_test_lim"])]
        # # q diff / p diff
        # q_diff = domain_params["q_test_lim"][1] - domain_params["q_test_lim"][0]
        # p_diff = domain_params["p_test_lim"][1] - domain_params["p_test_lim"][0]
        # if q_diff == p_diff:
            # aspect = 1
        # else:
            # aspect = 2.5
        # if save:
            # file_location = "../../plots/" + "_".join(test_name_tokens) + "GROUND_TRUTH" + ".pdf"
        # else:
            # file_location = '' # which defaults to not saving the plot
#
        # plot_2d(domain_params["H"](x_test).reshape((domain_params["N_p"],domain_params["N_q"])), q_test_range, p_test_range, extent=domain_params["q_test_lim"] + domain_params["p_test_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title="Ground Truth", verbose=verbose, save=file_location)
        # del q_test_range, p_test_range, q_test_grid, p_test_grid, x_test
        # return 0,0,0
#
#
    # # calculate domain sizes
    # N = domain_params["N_q"] * domain_params["N_p"]
    # V = domain_params["V_q"] * domain_params["V_p"]
    # K = domain_params["K_q"] * domain_params["K_p"]
#
    # # first we train the model with the train data (X, dX, x0, f0)
    # q_train_range, p_train_range, q_train_grid, p_train_grid = generate_grid_2d(domain_params["K_q"], domain_params["K_p"], domain_params["q_train_lim"], domain_params["p_train_lim"])
    # x_train = np.column_stack([q_train_grid.flatten(), p_train_grid.flatten()])
    # y_train_derivs = domain_params["dH"](x_train)
    # x0 = np.array([0,0])
    # f0 = domain_params["H"](x0.reshape(-1,2))
#
    # model = approximate_hamiltonian(x_train, y_train_derivs, x0, f0, model_params["M"], model_params["activation"], model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"], model_params["random_seed"], model_params["include_bias"])
    # model_name_tokens = [ k + str(v) for k, v in model_params.items()]
#
    # train_mse_loss = MSE(domain_params["H"](x_train), model.transform(x_train))
    # train_l2_error_relative = l2_error_relative(domain_params["H"](x_train), model.transform(x_train))
#
    # train_name_tokens = [domain_params["system_name"], "K_q" + str(domain_params["K_q"]), "K_p" + str(domain_params["K_p"]), "q_train_lim" + str(domain_params["q_train_lim"]), "p_train_lim" + str(domain_params["p_train_lim"]) + "noise" + str(domain_params["noise"])]
#
    # if verbose:
        # print("TRAIN RESULTS")
        # print("-> train mse loss on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_mse_loss))
        # print("-> train l2 relative error on domain:", " ".join(train_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(train_l2_error_relative))
        # print()
#
    # del q_train_range, p_train_range, q_train_grid, p_train_grid, x_train, y_train_derivs, x0, f0
#
    # # validate with a broader data in the same range as the training data
    # q_val_range, p_val_range, q_val_grid, p_val_grid = generate_grid_2d(domain_params["V_q"], domain_params["V_p"], domain_params["q_val_lim"], domain_params["p_val_lim"])
    # x_val = np.column_stack([q_val_grid.flatten(), p_val_grid.flatten()])
#
    # val_mse_loss = MSE(domain_params["H"](x_val), model.transform(x_val))
    # val_l2_error_relative = l2_error_relative(domain_params["H"](x_val), model.transform(x_val))
#
    # val_name_tokens = [domain_params["system_name"], "V_q" + str(domain_params["V_q"]), "V_p" + str(domain_params["V_p"]), "q_val_lim" + str(domain_params["q_val_lim"]), "p_val_lim" + str(domain_params["p_val_lim"]) + "noise" + str(domain_params["noise"])]
#
    # if verbose:
        # print("VAL RESULTS")
        # print("-> val mse loss on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_mse_loss))
        # print("-> val l2 relative error on domain:", " ".join(val_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(val_l2_error_relative))
#
    # # q diff / p diff
    # q_diff = domain_params["q_val_lim"][1] - domain_params["q_val_lim"][0]
    # p_diff = domain_params["p_val_lim"][1] - domain_params["p_val_lim"][0]
    # if q_diff == p_diff:
        # aspect = 1
    # else:
        # aspect = 2.5
#
    # if save:
        # file_location = "../../plots/" + "_".join(val_name_tokens) + "_" + "_".join(model_name_tokens) + "_train_l2_" + "{:.6f}".format(train_l2_error_relative) + "_val_l2_" + "{:.6f}".format(val_l2_error_relative) + ".pdf"
    # else:
        # file_location = '' # which defaults to not saving the plot
#
    # if model_params["clip"]:
        # y_plot = model.transform(x_val).reshape((domain_params["V_p"],domain_params["V_q"])).clip(np.min(domain_params["H"](x_val)), np.max(domain_params["H"](x_val)))
    # else:
        # y_plot = model.transform(x_val).reshape((domain_params["V_p"],domain_params["V_q"]))
#
    # plot_2d(y_plot, q_val_range, p_val_range, extent=domain_params["q_val_lim"] + domain_params["p_val_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)
#
    # del q_val_range, p_val_range, q_val_grid, p_val_grid, x_val, y_plot
#
    # # test with a broader range as the train and val data
    # q_test_range, p_test_range, q_test_grid, p_test_grid = generate_grid_2d(domain_params["N_q"], domain_params["N_p"], domain_params["q_test_lim"], domain_params["p_test_lim"])
    # x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])
#
    # test_mse_loss = MSE(domain_params["H"](x_test), model.transform(x_test))
    # test_l2_error_relative = l2_error_relative(domain_params["H"](x_test), model.transform(x_test))
#
    # test_name_tokens = [domain_params["system_name"], "N_q" + str(domain_params["N_q"]), "N_p" + str(domain_params["N_p"]), "q_test_lim" + str(domain_params["q_test_lim"]), "p_test_lim" + str(domain_params["p_test_lim"]) + "noise" + str(domain_params["noise"])]
#
    # if verbose:
        # print("TEST RESULTS")
        # print("-> test mse loss on domain:", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_mse_loss))
        # print("-> test l2 relative error on domain:", " ".join(test_name_tokens) + ", with model:", " ".join(model_name_tokens) + "\n" + str(test_l2_error_relative))
#
    # # q diff / p diff
    # q_diff = domain_params["q_test_lim"][1] - domain_params["q_test_lim"][0]
    # p_diff = domain_params["p_test_lim"][1] - domain_params["p_test_lim"][0]
    # if q_diff == p_diff:
        # aspect = 1
    # else:
        # aspect = 2.5
#
    # if save:
        # file_location = "../../plots/" + "_".join(test_name_tokens) + "_" + "_".join(model_name_tokens) + "_test_l2_" + "{:.6f}".format(test_l2_error_relative) + ".pdf"
    # else:
        # file_location = '' # which defaults to not saving the plot
#
    # if model_params["clip"]:
        # y_plot = model.transform(x_test).reshape((domain_params["N_p"],domain_params["N_q"])).clip(np.min(domain_params["H"](x_test)), np.max(domain_params["H"](x_test)))
    # else:
        # y_plot = model.transform(x_test).reshape((domain_params["N_p"],domain_params["N_q"]))
#
    # plot_2d(y_plot, q_test_range, p_test_range, extent=domain_params["q_test_lim"] + domain_params["p_test_lim"], contourlines=20, xlabel='q', ylabel='p', aspect=aspect, title=" ".join(model_name_tokens), verbose=verbose, save=file_location)
#
    # del q_test_range, p_test_range, q_test_grid, p_test_grid, x_test, y_plot
#
    # return (train_l2_error_relative), (val_l2_error_relative), (test_l2_error_relative)
