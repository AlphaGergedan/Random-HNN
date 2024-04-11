###################################
## Sample NNs using SWIM method! ##
###################################
import os, sys
if os.path.exists("/dss/dsshome1/0B/ge49rev3/master-thesis/src"):
    directory_to_prepend = os.path.abspath("/dss/dsshome1/0B/ge49rev3/master-thesis/src")
elif os.path.exists("./src"):
    directory_to_prepend = os.path.abspath("./src")
else:
    raise RuntimeError("'src' is not in path")

print(f"-> src path found in: '{directory_to_prepend}")

if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from time import time
from joblib import dump, load
from pathlib import Path
import argparse
import numpy as np
from utils.utils import parse_system_name, parse_activation, get_errors
from utils.grid import generate_train_test_grid
from utils.swim import swim, hswim, backward


################################
## READ DOMAIN AND MODEL ARGS ##
################################

parser = argparse.ArgumentParser(prog='HSWIM', description='Hamiltonian-SWIM-Networks')
# Read domain params
parser.add_argument('system_name', type=str, help='Hamiltonian system')
# list (for each dimension)
parser.add_argument('-dof', type=int, help='Degree of freedom of the given system', required=True)
parser.add_argument('-qtrain', type=int, nargs='+', help='Number of train points in q', required=True)
parser.add_argument('-ptrain', type=int, nargs='+', help='Number of train points in p', required=True)
parser.add_argument('-qtrainlimstart', type=float, nargs='+', help='Train q range start', required=True)
parser.add_argument('-qtrainlimend', type=float, nargs='+', help='Train q range end', required=True)
parser.add_argument('-ptrainlimstart', type=float, nargs='+', help='Train p range start', required=True)
parser.add_argument('-ptrainlimend', type=float, nargs='+', help='Train p range end', required=True)
parser.add_argument('-trainsetlinspaced', help='Whether train set is linspaced', action='store_true', default=False)
parser.add_argument('-qtest', nargs='+', type=int, help='Number of test points in q', required=True)
parser.add_argument('-ptest', nargs='+', type=int, help='Number of test points in p', required=True)
parser.add_argument('-qtestlimstart', type=float, nargs='+', help='Test q range start', required=True)
parser.add_argument('-qtestlimend', type=float, nargs='+', help='Test q range end', required=True)
parser.add_argument('-ptestlimstart', type=float, nargs='+', help='Test p range start', required=True)
parser.add_argument('-ptestlimend', type=float, nargs='+', help='Test p range end', required=True)
parser.add_argument('-repeat', type=int, help='Number of runs to experiment', required=True)

# Read model params
parser.add_argument('-nneurons', type=int, nargs='+', help='Number of hidden neurons in the hidden layers', required=True)
parser.add_argument('-nhiddenlayers', type=int, help='Number of hidden layers', required=True)
parser.add_argument('-activation', type=str, help='Activation function', required=True)
parser.add_argument('-parametersampler', type=str, help='Param sampler for SWIM: random, tanh, relu', required=True)
parser.add_argument('-rcond', type=float, help='How precise the lstsq is', required=True)
parser.add_argument('-trainrandomseedstart', type=int, help='Start seed for train set generation', required=True)
parser.add_argument('-testrandomseedstart', type=int, help='Start seed for test set generation', required=True)
parser.add_argument('-modelrandomseedstart', type=int, help='Start seed for model param generation', required=True)
parser.add_argument('-includebias', help='Whether to include bias in the network', action='store_true', default=False)
parser.add_argument('-elmbiasstart', type=float, help='Bias start for ELM hidden layers, it is sampled uniformly from [start,end]', required=True)
parser.add_argument('-elmbiasend', type=float, help='Bias end for ELM hidden layers, it is sampled uniformly from [start,end]', required=True)
parser.add_argument('-noise', type=float, help='Noise to add to the train set. Noise is added as gaussian noise to the points before evaluating the Hamiltonian value.', required=False)
parser.add_argument('-resampleduplicates', help='Whether to resample from data if duplicate weights are detected until getting unique weights.', action='store_true', default=False)

parser.add_argument('-solvetrue', help='Solves using true function values', action='store_true', default=False)

parser.add_argument('output_dir', type=str, help='Output directory for the experiment results')

# OUTPUT DIRECTORY FOR THE RESULTS

args = parser.parse_args()

# check dimensions train
assert args.dof == len(args.qtrain) == len(args.ptrain) == len(args.qtrainlimstart) == len(args.qtrainlimend) == len(args.ptrainlimstart) == len(args.ptrainlimend)
assert args.dof == len(args.qtest) == len(args.ptest) == len(args.qtestlimstart) == len(args.qtestlimend) == len(args.ptestlimstart) == len(args.ptestlimend)

# assert dimensions of nneurons matching nhiddenlayers
assert len(args.nneurons) == args.nhiddenlayers

# parse system
H, dH = parse_system_name(args.system_name)


NOISE_SEED = 9922381


# parse domain boundaries
q_train_lim, p_train_lim, q_test_lim, p_test_lim = [], [], [], []
for d in range(args.dof):
    current_q_train_lim = [args.qtrainlimstart[d], args.qtrainlimend[d]]
    q_train_lim.append(current_q_train_lim)
    current_p_train_lim = [args.ptrainlimstart[d], args.ptrainlimend[d]]
    p_train_lim.append(current_p_train_lim)
    current_q_test_lim = [args.qtestlimstart[d], args.qtestlimend[d]]
    q_test_lim.append(current_q_test_lim)
    current_p_test_lim = [args.ptestlimstart[d], args.ptestlimend[d]]
    p_test_lim.append(current_p_test_lim)
    print(f"iteration {d}, q_train_lim is {q_train_lim}")

print(f"q_train_lim: {q_train_lim}")
print(f"p_train_lim: {p_train_lim}")
print(f"q_test_lim: {q_test_lim}")
print(f"p_test_lim: {p_test_lim}")

domain_params = {
    "system_name": args.system_name, "H": H, "dH": dH, "dof": args.dof,
    "q_train": args.qtrain, "p_train": args.ptrain, "q_train_lim": q_train_lim, "p_train_lim": p_train_lim,
    "q_test": args.qtest, "p_test": args.ptest,  "q_test_lim": q_test_lim, "p_test_lim": p_test_lim,
    "train_set_linspaced": args.trainsetlinspaced,
    "train_random_seed_start": args.trainrandomseedstart,
    "test_random_seed_start": args.testrandomseedstart,
    "repeat": args.repeat,
    "solvetrue": args.solvetrue,
    "elm_bias_start": args.elmbiasstart,
    "elm_bias_end": args.elmbiasend,
    "noise": args.noise,
    "noise_seed": NOISE_SEED,
}

elm_params = {
    "name": "ELM",
    "n_neurons": args.nneurons,
    "n_hidden_layers": args.nhiddenlayers,
    "activation": args.activation,
    "parameter_sampler": "random",                  # weight sampling strategy, random for ELM
    "sample_uniformly": True,                       # whether to use uniform distribution for data point picking when sampling the weights
    "rcond": args.rcond,
    "resample_duplicates": args.resampleduplicates,  # this parameter is ignored in ELM, since in normal distribution it is almost impossible to get the same parameter
    "model_random_seed_start": args.modelrandomseedstart, # for reproducability, will be set uniquely for each run
    "include_bias": args.includebias,
}
uswim_params = {
    "name": "U-SWIM",
    "n_neurons": args.nneurons,
    "n_hidden_layers": args.nhiddenlayers,
    "activation": args.activation,
    "parameter_sampler": args.parametersampler,
    "sample_uniformly": True,                       # whether to use uniform distribution for data point picking when sampling the weights, True for uniform SWIM
    "rcond": args.rcond,
    "resample_duplicates": args.resampleduplicates,
    #"model_random_seed_start": None,#98765                      # for reproducability, will be set uniquely for each run
    "model_random_seed_start": args.modelrandomseedstart,       # for reproducability, will be set uniquely for each run, is set to 98765
    "include_bias": args.includebias,
}
aswim_params = {
    "name": "A-SWIM",                               # discriminative name for the model
    "n_neurons": args.nneurons,                     # number of hidden nodes as a list, where each entry corresponds to a hidden layer width
    "n_hidden_layers": args.nhiddenlayers,          # number of hidden layers
    "activation": args.activation,                  # activation function
    "parameter_sampler": args.parametersampler,     # weight sampling strategy
    "sample_uniformly": False,                      # whether to use uniform distribution for data point picking when sampling the weights, if set to false, we use approximate values to compute the prob. distribution
    "rcond": args.rcond,                            # regularization in lstsq in the linear layer
    "resample_duplicates": args.resampleduplicates,
    "model_random_seed_start": args.modelrandomseedstart,                            # for reproducability, will be set uniquely for each run
    "include_bias": args.includebias,               # bias in linear layer
}
swim_params = {
    "name": "SWIM",                                 # discriminative name for the model
    "n_neurons": args.nneurons,                     # number of hidden nodes as a list, where each entry corresponds to a hidden layer width
    "n_hidden_layers": args.nhiddenlayers,          # number of hidden layers
    "activation": args.activation,                  # actiation function
    "parameter_sampler": args.parametersampler,     # weight sampling strategy
    "sample_uniformly": False,                      # whether to use uniform distribution for data point picking when sampling the weights
    "rcond": args.rcond,                            # regularization in lstsq in the linear layer
    "resample_duplicates": args.resampleduplicates,
    "model_random_seed_start": args.modelrandomseedstart,                            # for reproducability, will be set uniquely for each run
    "include_bias": args.includebias                # bias in linear layer
}

models = [elm_params, uswim_params, aswim_params, swim_params]

# following is for experimenting with true function values
solvetrue_models = [elm_params, uswim_params, swim_params]

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
train_random_seed = domain_params["train_random_seed_start"]
test_random_seed = domain_params["test_random_seed_start"]

assert elm_params["model_random_seed_start"] == uswim_params["model_random_seed_start"] == aswim_params["model_random_seed_start"] == swim_params["model_random_seed_start"]
model_random_seed = elm_params["model_random_seed_start"]

# file to save in the end of experiment to document the results
# each run item will document an iteration, there are in total domain_params["repeat"] iterations
# each run item will contain "train_random_seed", "test_random_seed", "model_random_seed", "train_times", "train_function_errors", "train_gradient_errors", "test_function_errors", "test_gradient_errors"
experiment = {
    "domain_params": domain_params,
    "elm_params": elm_params, "uswim_params": uswim_params, "aswim_params": aswim_params, "swim_params": swim_params,
    "runs": []
}

# solve the true function
if args.solvetrue:
    for i in range(domain_params['repeat']):
        print(f'-> iterating {i}')

        # save random seeds
        current_run = {
            # trained models will be saved too
            "train_random_seed": train_random_seed, "test_random_seed": test_random_seed, "model_random_seed": model_random_seed,
            "train_function_errors": {}, "train_gradient_errors": {}, "test_function_errors": {}, "test_gradient_errors": {},
            "train_times": {},
        }

        # TODO: you can also generate train and test sets here, this would consume more memory but would be a little faster

        for model_params in solvetrue_models:
            print(f'---> model {model_params["name"]}')

            # TRAIN TEST DATA: first we train the model with the train data (X, dX, x0, f0) then evaluate
            train_rng = np.random.default_rng(train_random_seed)
            test_rng = np.random.default_rng(test_random_seed)
            _, _, q_train_grids, p_train_grids, _, _, _, _ = generate_train_test_grid(domain_params["q_train"], domain_params["p_train"], domain_params["q_train_lim"], domain_params["p_train_lim"], domain_params["q_test"], domain_params["p_test"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng=test_rng, dof=domain_params["dof"], linspace=domain_params["train_set_linspaced"], train_rng=train_rng)

            # column stacked (q_i, p_i): (N, 2*dof)
            x_train = np.column_stack([ q_train_grid.flatten() for q_train_grid in q_train_grids ] + [ p_train_grid.flatten() for p_train_grid in p_train_grids ])
            # x_train = x_train.astype(np.float16)
            del q_train_grids, p_train_grids
            y_train_true = domain_params["H"](x_train)

            f_activation, df_activation = parse_activation(model_params["activation"])

            # TRAIN MODEL ELM, USWIM and SWIM
            print('Entering swim..')
            t_start = time()
            model = swim(x_train, y_train_true, model_params["n_hidden_layers"], model_params["n_neurons"], f_activation,
                         model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"],
                         domain_params["elm_bias_start"], domain_params["elm_bias_end"], random_seed=model_random_seed)
            t_end = time()
            print(f'swim took {t_end - t_start} seconds')

            # save train time
            current_run["train_times"][model_params['name']] = t_end-t_start

            # save the trained models
            current_run[model_params['name']] = model

            print(f'calculating forward pass of x_train..')
            t_start = time()
            y_train_pred = model.transform(x_train)
            t_end = time()
            print(f'forward pass of x_train took {t_end-t_start} seconds')
            current_run['train_function_errors'][model_params['name']] = get_errors(y_train_true, y_train_pred)

            # to save some memory we sample again for the test set using the same seeds
            train_rng = np.random.default_rng(train_random_seed)
            test_rng = np.random.default_rng(test_random_seed)
            _, _, _, _, _, _, q_test_grids, p_test_grids = generate_train_test_grid(domain_params["q_train"], domain_params["p_train"], domain_params["q_train_lim"], domain_params["p_train_lim"], domain_params["q_test"], domain_params["p_test"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng=test_rng, dof=domain_params["dof"], linspace=domain_params["train_set_linspaced"], train_rng=train_rng)
            x_test = np.column_stack([ q_test_grid.flatten() for q_test_grid in q_test_grids ] + [ p_test_grid.flatten() for p_test_grid in p_test_grids ])
            del q_test_grids, p_test_grids

            y_test_true = domain_params["H"](x_test)
            print(f'calculating forward pass of x_test..')
            t_start = time()
            y_test_pred = model.transform(x_test)
            t_end = time()
            print(f'forward pass of x_test took {t_end-t_start} seconds')
            current_run['test_function_errors'][model_params['name']] = get_errors(y_test_true, y_test_pred)

        # update seeds
        train_random_seed += 1
        test_random_seed += 1
        model_random_seed += 1

        experiment['runs'].append(current_run)

    print('-----------------------------')
    print('-> Runs finished')

    # remove tmp files
    if os.path.exists(os.path.join(args.output_dir, f'TMP_Q_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl')):
        os.remove(os.path.join(args.output_dir, f'TMP_Q_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl'))
    if os.path.exists(os.path.join(args.output_dir, f'TMP_P_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl')):
        os.remove(os.path.join(args.output_dir, f'TMP_P_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl'))

    dump(experiment, os.path.join(args.output_dir, f'{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl'))
    print('-> Saved experiment results under: ' + args.output_dir)

    print()
    print('-> Experiment End')
    print()
    exit(0)


# solve the PDE
for i in range(domain_params['repeat']):
    print(f'-> iterating {i}')

    # save random seeds
    current_run = {
        # trained models will be saved too
        "train_random_seed": train_random_seed, "test_random_seed": test_random_seed, "model_random_seed": model_random_seed,
        "train_function_errors": {}, "train_gradient_errors": {}, "test_function_errors": {}, "test_gradient_errors": {},
        "train_times": {},
    }

    print(f"-> Generating train and test data for the current run {i}")
    # # TRAIN TEST DATA: first we train the model with the train data (X, dX, x0, f0) then evaluate
    # train_rng = np.random.default_rng(train_random_seed)
    # test_rng = np.random.default_rng(test_random_seed)
    # _, _, q_train_grids, p_train_grids, _, _, q_test_grids, p_test_grids = generate_train_test_grid(domain_params["q_train"], domain_params["p_train"], domain_params["q_train_lim"], domain_params["p_train_lim"], domain_params["q_test"], domain_params["p_test"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng=test_rng, dof=domain_params["dof"], linspace=domain_params["train_set_linspaced"], train_rng=train_rng)
#
    # # column stacked (q_i, p_i): (N, 2*dof)
    # x_train = np.column_stack([ q_train_grid.flatten() for q_train_grid in q_train_grids ] + [ p_train_grid.flatten() for p_train_grid in p_train_grids ])
    # del q_train_grids, p_train_grids
#
    # # you can specify dtype=np.float16 for half precision: x_train = x_train.astype(np.float16)
    # Path(os.path.join(args.output_dir, f'TMP_X_TRAIN_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')).touch()
    # np.save(os.path.join(args.output_dir, f'TMP_X_TRAIN_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'), x_train)
#
    # y_train_derivs_true = domain_params["dH"](x_train)
    # Path(os.path.join(args.output_dir, f'TMP_Y_TRAIN_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')).touch()
    # np.save(os.path.join(args.output_dir, f'TMP_Y_TRAIN_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'), y_train_derivs_true)
    # del y_train_derivs_true
#
    # y_train_true = domain_params["H"](x_train)
    # Path(os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')).touch()
    # np.save(os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'), y_train_true)
    # del y_train_true
#
    # del x_train
#
    # x_test = np.column_stack([ q_test_grid.flatten() for q_test_grid in q_test_grids ] + [ p_test_grid.flatten() for p_test_grid in p_test_grids ])
    # del q_test_grids, p_test_grids
#
    # # you can specify dtype=np.float16 for half precision: x_test = x_test.astype(np.float16)
    # Path(os.path.join(args.output_dir, f'TMP_X_TEST_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')).touch()
    # np.save(os.path.join(args.output_dir, f'TMP_X_TEST_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'), x_test)
#
    # y_test_derivs_true = domain_params["dH"](x_test)
    # Path(os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')).touch()
    # np.save(os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'), y_test_derivs_true)
    # del y_test_derivs_true
#
    # y_test_true = domain_params["H"](x_test)
    # Path(os.path.join(args.output_dir, f'TMP_Y_TEST_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')).touch()
    # np.save(os.path.join(args.output_dir, f'TMP_Y_TEST_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'), y_test_true)
    # del y_test_true
#
    # del x_test

    for model_params in models:
        print(f'---> model {model_params["name"]}')
        # TRAIN TEST DATA: first we train the model with the train data (X, dX, x0, f0) then evaluate
        train_rng = np.random.default_rng(train_random_seed)
        test_rng = np.random.default_rng(test_random_seed)
        _, _, q_train_grids, p_train_grids, _, _, _, _ = generate_train_test_grid(domain_params["q_train"], domain_params["p_train"], domain_params["q_train_lim"], domain_params["p_train_lim"], domain_params["q_test"], domain_params["p_test"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng=test_rng, dof=domain_params["dof"], linspace=domain_params["train_set_linspaced"], train_rng=train_rng)

        # load x_test and y_train_derivs_true
        # x_train = np.load(os.path.join(args.output_dir, f'TMP_X_TRAIN_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
        # column stacked (q_i, p_i): (N, 2*dof)
        x_train = np.column_stack([ q_train_grid.flatten() for q_train_grid in q_train_grids ] + [ p_train_grid.flatten() for p_train_grid in p_train_grids ])

        # add noise
        if args.noise:
            noise_rng = np.random.default_rng(NOISE_SEED)
            y_train_derivs_true = domain_params["dH"](x_train + noise_rng.normal(0, args.noise, x_train.shape))
        else:
            y_train_derivs_true = domain_params["dH"](x_train)
        # y_train_derivs_true = np.load(os.path.join(args.output_dir, f'TMP_Y_TRAIN_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))

        # input is of shape 2*dof, x0 = [[q_1, q_2, .., q_dof, p_1, p_2, .., p_dof]]
        x0 = np.zeros(2 * domain_params["dof"]).reshape(1, -1)
        f0 = domain_params["H"](x0)

        f_activation, df_activation = parse_activation(model_params["activation"])

        # TRAIN MODEL
        y_train_true = None
        if model_params["name"] == "SWIM":
            # y_train_true = np.load(os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
            y_train_true = domain_params["H"](x_train)
        print('Entering hswim..')
        t_start = time()
        # TODO specify layers too
        model = hswim(x_train, y_train_derivs_true, x0, f0,
                      model_params["n_hidden_layers"], model_params["n_neurons"], f_activation, df_activation,
                      model_params["parameter_sampler"], model_params["sample_uniformly"], model_params["rcond"],
                      domain_params["elm_bias_start"], domain_params["elm_bias_end"],
                      y_train_true=y_train_true, random_seed=model_random_seed, include_bias=model_params["include_bias"], resample_duplicates=model_params["resample_duplicates"])
        t_end = time()
        print(f'hswim took {t_end - t_start} seconds')

        # save train time
        current_run["train_times"][model_params['name']] = t_end-t_start

        # save the trained models
        current_run[model_params['name']] = model

        # EVALUATE LOSS (error on derivative approximation)
        print(f'calculating backward pass of x_train..')
        t_start = time()
        y_train_derivs_pred = backward(model, model_params["activation"], x_train)
        t_end = time()
        print(f'backward pass of x_train took {t_end - t_start} seconds')
        current_run['train_gradient_errors'][model_params['name']] = get_errors(y_train_derivs_true, y_train_derivs_pred)
        del y_train_derivs_true, y_train_derivs_pred

        print(f'calculating forward pass of x_train..')
        t_start = time()
        y_train_pred = model.transform(x_train)
        t_end = time()
        print(f'forward pass of x_train took {t_end-t_start} seconds')

        if y_train_true is None:
            # y_train_true = np.load(os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
            y_train_true = domain_params["H"](x_train)
        del x_train

        current_run['train_function_errors'][model_params['name']] = get_errors(y_train_true, y_train_pred)
        del y_train_true, y_train_pred

        # load q,p test grids, TODO load x_test
        # TRAIN TEST DATA: first we train the model with the train data (X, dX, x0, f0) then evaluate
        train_rng = np.random.default_rng(train_random_seed)
        test_rng = np.random.default_rng(test_random_seed)
        _, _, _, _, _, _, q_test_grids, p_test_grids = generate_train_test_grid(domain_params["q_train"], domain_params["p_train"], domain_params["q_train_lim"], domain_params["p_train_lim"], domain_params["q_test"], domain_params["p_test"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng=test_rng, dof=domain_params["dof"], linspace=domain_params["train_set_linspaced"], train_rng=train_rng)
        x_test = np.column_stack([ q_test_grid.flatten() for q_test_grid in q_test_grids ] + [ p_test_grid.flatten() for p_test_grid in p_test_grids ])
        # x_test = np.load(os.path.join(args.output_dir, f'TMP_X_TEST_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))

        print(f'calculating backward pass of x_test..')
        t_start = time()
        y_test_derivs_pred = backward(model, model_params["activation"], x_test)
        t_end = time()
        print(f'backward pass of x_test took {t_end-t_start} seconds')

        # y_test_derivs_true = np.load(os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
        y_test_derivs_true = domain_params["dH"](x_test)
        current_run['test_gradient_errors'][model_params['name']] = get_errors(y_test_derivs_true, y_test_derivs_pred)
        del y_test_derivs_true, y_test_derivs_pred

        print(f'calculating forward pass of x_test..')
        t_start = time()
        y_test_pred = model.transform(x_test)
        t_end = time()
        print(f'forward pass of x_test took {t_end-t_start} seconds')

        # y_test_true = np.load(os.path.join(args.output_dir, f'TMP_Y_TEST_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
        y_test_true = domain_params["H"](x_test)
        current_run['test_function_errors'][model_params['name']] = get_errors(y_test_true, y_test_pred)
        del y_test_true, y_test_pred, x_test

    # clear the saved train and test data

    # train
    # if os.path.exists(os.path.join(args.output_dir, f'TMP_X_TRAIN_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')):
        # os.remove(os.path.join(args.output_dir, f'TMP_X_TRAIN_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
    # else:
        # raise RuntimeError(f"File {os.path.join(args.output_dir, f'TMP_X_TRAIN_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')} does not exist")
#
    # if os.path.exists(os.path.join(args.output_dir, f'TMP_Y_TRAIN_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')):
        # os.remove(os.path.join(args.output_dir, f'TMP_Y_TRAIN_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
    # else:
        # raise RuntimeError(f"File {os.path.join(args.output_dir, f'TMP_Y_TRAIN_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')} does not exist")
#
    # if os.path.exists(os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')):
        # os.remove(os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
    # else:
        # raise RuntimeError(f"File {os.path.join(args.output_dir, f'TMP_Y_TRAIN_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')} does not exist")
#
    # # test
    # if os.path.exists(os.path.join(args.output_dir, f'TMP_X_TEST_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')):
        # os.remove(os.path.join(args.output_dir, f'TMP_X_TEST_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
    # else:
        # raise RuntimeError(f"File {os.path.join(args.output_dir, f'TMP_X_TEST_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')} does not exist")
#
    # if os.path.exists(os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')):
        # os.remove(os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
    # else:
        # raise RuntimeError(f"File {os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')} does not exist")
#
    # if os.path.exists(os.path.join(args.output_dir, f'TMP_Y_TEST_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')):
        # os.remove(os.path.join(args.output_dir, f'TMP_Y_TEST_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
    # else:
        # raise RuntimeError(f"File {os.path.join(args.output_dir, f'TMP_Y_TEST_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy')} does not exist")

    # update seeds
    train_random_seed += 1
    test_random_seed += 1
    model_random_seed += 1
    NOISE_SEED += 1

    experiment['runs'].append(current_run)

print('-----------------------------')
print('-> Runs finished')

# remove tmp files
# if os.path.exists(os.path.join(args.output_dir, f'TMP_Q_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl')):
    # os.remove(os.path.join(args.output_dir, f'TMP_Q_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl'))
# if os.path.exists(os.path.join(args.output_dir, f'TMP_P_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl')):
    # os.remove(os.path.join(args.output_dir, f'TMP_P_TEST_GRIDS_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}.pkl'))

# append the total num of points to the name of the file
total_q = 1
total_p = 1
for qtrain in args.qtrain:
    total_q *= qtrain
for ptrain in args.ptrain:
    total_p *= ptrain

dump(experiment, os.path.join(args.output_dir, f'{total_q*total_p}domain{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons{args.elmbiasstart}to{args.elmbiasend}elmbiasnoise{args.noise}resampleduplicates{args.resampleduplicates}_{args.system_name}.pkl'))
print('-> Saved experiment results under: ' + args.output_dir)

print()
print('-> Experiment End')
print()
