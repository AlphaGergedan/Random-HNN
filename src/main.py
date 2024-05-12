"""
src/main.py

File to run the experiments, and saves them using joblib, mainly used in the cluster, additionally
requires plotting files in `src/scripts` to be run to generate the plots

To see some example runs see `README.md`
"""

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
from joblib import dump
import argparse
import numpy as np
from utils.utils import parse_system_name, parse_activation, get_errors
from utils.grid import generate_uniform_train_test_set
from utils.swim import hswim, backward
from utils.trajectories import flow_map_symp_euler, flow_map_rk45

################################
## READ DOMAIN AND MODEL ARGS ##
################################

parser = argparse.ArgumentParser(prog='HSWIM', description='Hamiltonian-SWIM-Networks')
# Read domain params
parser.add_argument('system_name', type=str, help='Hamiltonian system')
# list (for each dimension)
parser.add_argument('-dof', type=int, help='Degree of freedom of the given system', required=True)
parser.add_argument('-trainsetsize', type=int, help='Number of points in the train set', required=True)
parser.add_argument('-qtrainlimstart', type=float, nargs='+', help='Train q range start', required=True)
parser.add_argument('-qtrainlimend', type=float, nargs='+', help='Train q range end', required=True)
parser.add_argument('-ptrainlimstart', type=float, nargs='+', help='Train p range start', required=True)
parser.add_argument('-ptrainlimend', type=float, nargs='+', help='Train p range end', required=True)
parser.add_argument('-testsetsize', type=int, help='Number of points in the test set', required=True)
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

parser.add_argument('-usefd', help='Use finite-differences instead of true derivative values', action='store_true', default=False)
parser.add_argument('-trainintegrator', type=str, help='Integrator to used in case of finite-differences for training', required=False)
parser.add_argument('-trueflowintegrator', type=str, help="Integrator for simulating the true flow, either 'rk45' or 'symplectic_euler'", required=False)
parser.add_argument('-timestepobservations', type=float, help='Time-step of the observed trajectories', required=False)
parser.add_argument('-timestepflowtrue', type=float, help='Time-step used for ground-truth generation', required=False)
parser.add_argument('-correct', help='Whether to apply Post-correction when logging the errors', action='store_true', default=False)

parser.add_argument('output_dir', type=str, help='Output directory for the experiment results')

# OUTPUT DIRECTORY FOR THE RESULTS

args = parser.parse_args()

# check dimensions train
assert args.dof == len(args.qtrainlimstart) == len(args.qtrainlimend) == len(args.ptrainlimstart) == len(args.ptrainlimend)
assert args.dof == len(args.qtestlimstart) == len(args.qtestlimend) == len(args.ptestlimstart) == len(args.ptestlimend)

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

domain_params = {
    "system_name": args.system_name, "H": H, "dH": dH, "dof": args.dof, "train_set_size": args.trainsetsize, "test_set_size": args.testsetsize,
    "q_train_lim": q_train_lim, "p_train_lim": p_train_lim, "q_test_lim": q_test_lim, "p_test_lim": p_test_lim,
    "train_random_seed_start": args.trainrandomseedstart, "test_random_seed_start": args.testrandomseedstart,
    "repeat": args.repeat,
    "elm_bias_start": args.elmbiasstart,
    "elm_bias_end": args.elmbiasend,
    "noise": args.noise,
    "noise_seed": NOISE_SEED,
    "use_finite_differences": args.usefd, "train_integrator": args.trainintegrator,
    "dt_obs": args.timestepobservations, "dt_flow_true": args.timestepflowtrue, "true_flow_integrator": args.trueflowintegrator,
    "post_correction": args.correct,
}

elm_params = {
    "name": "ELM",
    "n_neurons": args.nneurons,
    "n_hidden_layers": args.nhiddenlayers,
    "activation": args.activation,
    "parameter_sampler": "random",                        # weight sampling strategy, random for ELM
    "sample_uniformly": True,                             # whether to use uniform distribution for data point picking when sampling the weights
    "rcond": args.rcond,
    "resample_duplicates": args.resampleduplicates,       # this parameter is ignored in ELM, since in normal distribution it is almost impossible to get the same parameter
    "model_random_seed_start": args.modelrandomseedstart, # for reproducability, will be set uniquely for each run
    "include_bias": args.includebias,
}
uswim_params = {
    "name": "U-SWIM",
    "n_neurons": args.nneurons,
    "n_hidden_layers": args.nhiddenlayers,
    "activation": args.activation,
    "parameter_sampler": args.parametersampler,
    "sample_uniformly": True,                               # whether to use uniform distribution for data point picking when sampling the weights, True for uniform SWIM
    "rcond": args.rcond,
    "resample_duplicates": args.resampleduplicates,
    "model_random_seed_start": args.modelrandomseedstart,   # for reproducability, will be set uniquely for each run, is set to 98765
    "include_bias": args.includebias,
}
aswim_params = {
    "name": "A-SWIM",
    "n_neurons": args.nneurons,
    "n_hidden_layers": args.nhiddenlayers,
    "activation": args.activation,
    "parameter_sampler": args.parametersampler,
    "sample_uniformly": False,                              # whether to use uniform distribution for data point picking when sampling the weights, if set to false, we use approximate values to compute the prob. distribution
    "rcond": args.rcond,
    "resample_duplicates": args.resampleduplicates,
    "model_random_seed_start": args.modelrandomseedstart,   # for reproducability, will be set uniquely for each run
    "include_bias": args.includebias,
}
swim_params = {
    "name": "SWIM",
    "n_neurons": args.nneurons,
    "n_hidden_layers": args.nhiddenlayers,
    "activation": args.activation,
    "parameter_sampler": args.parametersampler,
    "sample_uniformly": False,                              # no uniform sampling for SWIM
    "rcond": args.rcond,
    "resample_duplicates": args.resampleduplicates,
    "model_random_seed_start": args.modelrandomseedstart,   # for reproducability, will be set uniquely for each run
    "include_bias": args.includebias
}

models = [elm_params, uswim_params, aswim_params, swim_params]

# following is for experimenting with true function values

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

# begin experiment of solving the PDE!
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

    for model_params in models:
        print(f'---> model {model_params["name"]}')

        train_rng = np.random.default_rng(train_random_seed)
        test_rng = np.random.default_rng(test_random_seed)
        x_train, _ = generate_uniform_train_test_set(domain_params["train_set_size"], domain_params["q_train_lim"], domain_params["p_train_lim"], train_rng,
                                                     domain_params["test_set_size"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng,
                                                     dof=domain_params["dof"])

        assert x_train.shape[0] == domain_params["train_set_size"]

        if domain_params["use_finite_differences"]:
            # compute x_train_next using the true flow (ground truth observations spaced with given time-step dt_obs)
            if domain_params["true_flow_integrator"] == "symplectic_euler":
                x_train_next = np.array([flow_map_symp_euler(x_i, domain_params["dH"], dt_flow_true=domain_params["dt_flow_true"], dt_obs=domain_params["dt_obs"]) for x_i in x_train])
            elif domain_params["true_flow_integrator"] == "rk45":
                x_train_next = np.array([flow_map_rk45(x_i, domain_params["dH"], dt_flow_true=domain_params["dt_flow_true"], dt_obs=domain_params["dt_obs"]) for x_i in x_train])
            else:
                raise RuntimeError(f"does not know integrator {domain_params['true_flow_integrator']} for simulating the true flow, we currently support 'rk45' or 'symplectic_euler'.")

            J_inv = np.array([[0, -1],
                              [1, 0]])

            # applies J_inv @ (x_next-x_prev)/h for each point
            # y_train_derivs_true = np.einsum("ij,kj->ik", ((x_train_next - x_train) / domain_params["dt_obs"]), J_inv)
            if domain_params["noise"]:
                raise NotImplemented("Noise for finite differences is not implemented yet!")
            else:
                y_train_derivs_true = (J_inv @ ((x_train_next - x_train) / domain_params["dt_obs"]).T).T
        else:
            x_train_next = None

            # add noise
            if domain_params["noise"]:
                noise_rng = np.random.default_rng(NOISE_SEED)
                y_train_derivs_true = domain_params["dH"](x_train + noise_rng.normal(0, domain_params["noise"], x_train.shape))
            else:
                y_train_derivs_true = domain_params["dH"](x_train)

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
                      y_train_true=y_train_true, random_seed=model_random_seed, include_bias=model_params["include_bias"],
                      resample_duplicates=model_params["resample_duplicates"], x_train_next=x_train_next, train_integration_scheme=domain_params["train_integrator"])
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
        del y_train_derivs_true

        print(f'calculating forward pass of x_train..')
        t_start = time()
        y_train_pred = model.transform(x_train)
        t_end = time()
        print(f'forward pass of x_train took {t_end-t_start} seconds')

        if y_train_true is None:
            y_train_true = domain_params["H"](x_train)
        del x_train

        if domain_params["post_correction"]:
            (K,D) = y_train_derivs_pred.shape
            # calculate the first-order correction to the Hamiltonian for the SympEuler scheme
            # The einsum realizes a dot product between dH_q and dH_p
            y_train_pred = y_train_pred + (domain_params["dt_obs"]/2) * np.einsum('ij,ij->i', y_train_derivs_pred[:,:D//2], y_train_derivs_pred[:,D//2:]).reshape(-1, 1)

        current_run['train_function_errors'][model_params['name']] = get_errors(y_train_true, y_train_pred)
        del y_train_true, y_train_pred, y_train_derivs_pred

        # sample the test set using the same generators, we sample twice to avoid memory issues
        train_rng = np.random.default_rng(train_random_seed)
        test_rng = np.random.default_rng(test_random_seed)
        _, x_test = generate_uniform_train_test_set(domain_params["train_set_size"], domain_params["q_train_lim"], domain_params["p_train_lim"], train_rng,
                                                    domain_params["test_set_size"], domain_params["q_test_lim"], domain_params["p_test_lim"], test_rng,
                                                    dof=domain_params["dof"])

        assert x_test.shape[0] == domain_params["test_set_size"]

        print(f'calculating backward pass of x_test..')
        t_start = time()
        y_test_derivs_pred = backward(model, model_params["activation"], x_test)
        t_end = time()
        print(f'backward pass of x_test took {t_end-t_start} seconds')

        # y_test_derivs_true = np.load(os.path.join(args.output_dir, f'TMP_Y_TEST_DERIVS_TRUE_{args.qtrain}qtrain{args.ptrain}ptrain{args.nneurons}neurons_{args.system_name}_run{i}.npy'))
        y_test_derivs_true = domain_params["dH"](x_test)
        current_run['test_gradient_errors'][model_params['name']] = get_errors(y_test_derivs_true, y_test_derivs_pred)
        del y_test_derivs_true

        print(f'calculating forward pass of x_test..')
        t_start = time()
        y_test_pred = model.transform(x_test)
        t_end = time()
        print(f'forward pass of x_test took {t_end-t_start} seconds')

        if domain_params["post_correction"]:
            (K,D) = y_test_derivs_pred.shape
            y_test_pred = y_test_pred + (domain_params["dt_obs"]/2) * np.einsum('ij,ij->i', y_test_derivs_pred[:,:D//2], y_test_derivs_pred[:,D//2:]).reshape(-1, 1)

        y_test_true = domain_params["H"](x_test)
        current_run['test_function_errors'][model_params['name']] = get_errors(y_test_true, y_test_pred)
        del y_test_true, y_test_pred, x_test, y_test_derivs_pred

    # update seeds
    train_random_seed += 1
    test_random_seed += 1
    model_random_seed += 1
    NOISE_SEED += 1

    experiment['runs'].append(current_run)

print('-----------------------------')
print('-> Runs finished')

if domain_params["use_finite_differences"]:
    dump(experiment, os.path.join(args.output_dir, f'FD_{domain_params["train_set_size"]}domain{elm_params["n_neurons"]}neurons{elm_params["resample_duplicates"]}resampleduplicates{domain_params["dt_obs"]}dtobs{domain_params["dt_flow_true"]}dtexact{domain_params["post_correction"]}postcorrection_{args.system_name}.pkl'))
else:
    dump(experiment, os.path.join(args.output_dir, f'{domain_params["train_set_size"]}domain{elm_params["n_neurons"]}neurons{elm_params["resample_duplicates"]}resampleduplicates{domain_params["noise"]}noise_{args.system_name}.pkl'))

print('-> Saved experiment results under: ' + args.output_dir)

print()
print('-> Experiment End')
print()
