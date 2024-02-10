###################################
## Sample NNs using SWIM method! ##
###################################
import os, sys
directory_to_prepend = os.path.abspath("./src")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from time import time
from joblib import dump
import argparse
import numpy as np
from utils.utils import parse_system_name, parse_activation, get_errors
from utils.grid import generate_train_test_grid
from utils.swim import hswim, backward


################################
## READ DOMAIN AND MODEL ARGS ##
################################

parser = argparse.ArgumentParser(prog='HSWIM', description='Hamiltonian-SWIM-Networks')
# Read domain params
parser.add_argument('system_name', type=str, help='Hamiltonian system')
parser.add_argument('q_train', type=int, help='Number of train points in q')
parser.add_argument('p_train', type=int, help='Number of train points in p')
parser.add_argument('q_train_lim_start', type=float, help='Train q range start')
parser.add_argument('q_train_lim_end', type=float, help='Train q range end')
parser.add_argument('p_train_lim_start', type=float, help='Train p range start')
parser.add_argument('p_train_lim_end', type=float, help='Train p range end')
parser.add_argument('train_set_linspaced', type=bool, help='Whether train set is linspaced')
parser.add_argument('q_test', type=int, help='Number of test points in q')
parser.add_argument('p_test', type=int, help='Number of test points in p')
parser.add_argument('q_test_lim_start', type=float, help='Test q range start')
parser.add_argument('q_test_lim_end', type=float, help='Test q range end')
parser.add_argument('p_test_lim_start', type=float, help='Test p range start')
parser.add_argument('p_test_lim_end', type=float, help='Test p range end')
parser.add_argument('repeat', type=int, help='Number of runs to experiment')

# Read model params
parser.add_argument('M', type=int, help='Number of hidden neurons')
parser.add_argument('activation', type=str, help='Activation function')
parser.add_argument('parameter_sampler', type=str, help='Param sampler for SWIM: random, tanh, relu')
parser.add_argument('rcond', type=float, help='How precise the lstsq is')
parser.add_argument('include_bias', type=bool, help='Whether to include bias in the network')

parser.add_argument('output_dir', type=str, help='Output directory for the experiment results')

# OUTPUT DIRECTORY FOR THE RESULTS

args = parser.parse_args()

# parse system
H, dH = parse_system_name(args.system_name)

domain_params = {
    "system_name": args.system_name,
    "H": H, "dH": dH,
    "q_train": args.q_train, "p_train": args.p_train, "q_train_lim": [args.q_train_lim_start,args.q_train_lim_end], "p_train_lim": [args.p_train_lim_start,args.p_train_lim_end],
    "q_test": args.q_test, "p_test": args.p_test, "q_test_lim": [args.q_test_lim_start,args.q_test_lim_end], "p_test_lim": [args.p_test_lim_start,args.p_test_lim_end],
    # Recommended is to have equal number of points to the number of parameters of the network
    # in our case number of params are = 4M + 1
    "train_set_linspaced": args.train_set_linspaced,
    "train_random_seed": None, # 23451
    "test_random_seed": None, # 54321
    "repeat": args.repeat,
}
elm_params = {
    "name": "ELM",                                  # discriminative name for the model
    "M": args.M,                                    # hidden nodes
    "activation": args.activation,                  # activation function
    "parameter_sampler": "random",                      # weight sampling strategy, random for ELM
    "sample_uniformly": True,                       # whether to use uniform distribution for data point picking when sampling the weights
    "rcond": args.rcond,                            # regularization in lstsq in the linear layer
    "random_seed": None,#98765                      # for reproducability
    "include_bias": args.include_bias,              # bias in linear layer
}
uswim_params = {
    "name": "U-SWIM",                               # discriminative name for the model
    "M": args.M,                                    # hidden nodes
    "activation": args.activation,                  # activation function
    "parameter_sampler": args.parameter_sampler,            # weight sampling strategy
    "sample_uniformly": True,                       # whether to use uniform distribution for data point picking when sampling the weights
    "rcond": args.rcond,                            # regularization in lstsq in the linear layer
    "random_seed": None,#98765                      # for reproducability
    "include_bias": args.include_bias,              # bias in linear layer
}
aswim_params = {
    "name": "A-SWIM",                               # discriminative name for the model
    "M": args.M,                                    # hidden nodes
    "activation": args.activation,                  # activation function
    "parameter_sampler": args.parameter_sampler,            # weight sampling strategy
    "sample_uniformly": False,                      # whether to use uniform distribution for data point picking when sampling the weights, if set to false, we use approximate values to compute the prob. distribution
    "rcond": args.rcond,                            # regularization in lstsq in the linear layer
    "random_seed": None,#98765                      # for reproducability
    "include_bias": args.include_bias,              # bias in linear layer
}
swim_params = {
    "name": "SWIM",                                 # discriminative name for the model
    "M": args.M,                                    # hidden nodes
    "activation": args.activation,                  # actiation function
    "parameter_sampler": args.parameter_sampler,            # weight sampling strategy
    "sample_uniformly": False,                      # whether to use uniform distribution for data point picking when sampling the weights
    "rcond": args.rcond,                            # regularization in lstsq in the linear layer
    "random_seed": None,#98765                      # for reproducability
    "include_bias": args.include_bias               # bias in linear layer
}

models = [elm_params, uswim_params, aswim_params, swim_params]

################
## EXPERIMENT ##
################

## DEBUG OUTPUT TO SEE THE DOMAIN AND MODEL PARAMS
print('-- DOMAIN PARAMS --')
print(str(domain_params))
for model in models:
    print('-- ' + model['name'] + ' PARAMS --')
    print(str(model))
print()
print('-> Results will be saved under: ' + args.output_dir)

print()
print('-> Experiment Start')
print()

# current seeds, will be incremented on each run for randomness
# train_random_seed = 3943
# test_random_seed = 29548
# model_random_seed = 992472
train_random_seed = None
test_random_seed = None
model_random_seed = None

# file to save in the end of experiment to document the results
# each run item will document an iteration, there are in total domain_params["repeat"] iterations
# each run item will contain "train_random_seed", "test_random_seed", "model_random_seed", "train_times", "train_errors", "train_losses", "test_errors", "test_losses"
experiment = {
    "domain_params": domain_params,
    "elm_params": elm_params, "uswim_params": uswim_params, "aswim_params": aswim_params, "swim_params": swim_params,
    "runs": []
}

for i in range(domain_params['repeat']):
    print(f'-> iterating {i}')

    # save random seeds
    current_run = {
        # trained models will be saved too
        "train_random_seed": train_random_seed, "test_random_seed": test_random_seed, "model_random_seed": model_random_seed,
        "train_errors": {}, "train_losses": {}, "test_errors": {}, "test_losses": {},
        "train_times": {},
    }

    for model_params in models:
        print(f'---> model {model_params["name"]}')

        # TRAIN TEST DATA: first we train the model with the train data (X, dX, x0, f0) then evaluate
        train_rng = np.random.default_rng(train_random_seed)
        test_rng = np.random.default_rng(test_random_seed)
        [q_train_range], [p_train_range], [q_train_grid], [p_train_grid], [q_test_range], [p_test_range], [q_test_grid], [p_test_grid] = generate_train_test_grid([domain_params["q_train"]], [domain_params["p_train"]], [domain_params["q_train_lim"]], [domain_params["p_train_lim"]],[domain_params["q_test"]], [domain_params["p_test"]], [domain_params["q_test_lim"]], [domain_params["p_test_lim"]], test_rng=test_rng, dof=1, linspace=domain_params["train_set_linspaced"], train_rng=train_rng)
        x_train = np.column_stack([q_train_grid.flatten(), p_train_grid.flatten()])
        y_train_derivs_true = domain_params["dH"](x_train)
        x0 = np.array([[0,0]])
        f0 = domain_params["H"](x0)

        f_activation, df_activation = parse_activation(model_params["activation"])

        # TRAIN MODEL
        y_train_true = None
        if model_params["name"] == "SWIM":
            y_train_true = domain_params["H"](x_train)
        t_start = time()
        model = hswim(x_train, y_train_derivs_true, x0, f0,
                      model_params["M"], f_activation, df_activation, model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"],
                      y_train_true=y_train_true, random_seed=model_random_seed, include_bias=model_params["include_bias"])
        t_end = time()

        # save train time
        current_run["train_times"][model_params['name']] = t_end-t_start

        # save the trained models
        current_run[model_params['name']] = model

        # EVALUATE LOSS (error on derivative approximation)
        y_train_derivs_pred = backward(model, model_params["activation"], x_train)
        current_run['train_losses'][model_params['name']] = get_errors(y_train_derivs_true, y_train_derivs_pred)

        x_test = np.column_stack([q_test_grid.flatten(), p_test_grid.flatten()])
        y_test_derivs_true = domain_params["dH"](x_test)
        y_test_derivs_pred = backward(model, model_params["activation"], x_test)
        current_run['test_losses'][model_params['name']] = get_errors(y_test_derivs_true, y_test_derivs_pred)

        y_train_true = domain_params["H"](x_train)
        y_train_pred = model.transform(x_train)
        current_run['train_errors'][model_params['name']] = get_errors(y_train_true, y_train_pred)

        y_test_true = domain_params["H"](x_test)
        y_test_pred = model.transform(x_test)
        current_run['test_errors'][model_params['name']] = get_errors(y_test_true, y_test_pred)

        # update seeds
        # train_random_seed += 1
        # test_random_seed += 1
        # model_random_seed += 1

    experiment['runs'].append(current_run)

print('-----------------------------')
print('-> Runs finished')
dump(experiment, os.path.join(args.output_dir, f'{args.q_train}q_train{args.p_train}p_train_{args.system_name}_NOSEED.pkl'))
print('-> Saved experiment results under: ' + args.output_dir)

print()
print('-> Experiment End')
print()
