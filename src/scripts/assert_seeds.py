########################
## FOR CHECKING SEEDS ##
########################
import os, sys
directory_to_prepend = os.path.abspath("./src")
if not os.path.exists(directory_to_prepend):
    raise RuntimeError("src directory not found")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from joblib import load, dump
import numpy as np
import argparse

###############
## READ ARGS ##
###############

parser = argparse.ArgumentParser(prog='ASSERTER', description='asserts experiment results')
parser.add_argument('-f','--file', nargs='+', help='Experiment output file locations', required=True)

args = parser.parse_args()

################################
## COMBINE EXPERIMENT RESULTS ##
################################

# we will fill the table below and aggregate the results of the experiments
# experiment = {
    # "domain_params": None,
    # "elm_params": None, "uswim_params": None, "aswim_params": None, "swim_params": None,
    # "runs": []
# }

for f in args.file:
    print(f"-> loading experiment: {f}")
    experiment = load(f)
    print(f"DOMAIN PARAMS: {experiment["domain_params"]}")

    # assert that seeds start with these values and being incremented in each run
    train_random_seeds_start = 3943
    test_random_seeds_start = 29548
    model_random_seeds_start = 992472

    train_random_seeds = [ run["train_random_seed"] for run in experiment["runs"] ]
    test_random_seeds = [ run["test_random_seed"] for run in experiment["runs"] ]
    model_random_seeds = [ run["model_random_seed"] for run in experiment["runs"] ]

    print()
    print(f"train_random_seeds: {train_random_seeds}")
    print()
    print(f"test_random_seeds: {test_random_seeds}")
    print()
    print(f"model_random_seeds: {model_random_seeds}")
    print()

    for i in range(len(train_random_seeds)):
        assert train_random_seeds[i] == train_random_seeds_start + i
        assert test_random_seeds[i] == test_random_seeds_start + i
        assert model_random_seeds[i] == model_random_seeds_start + i

    # get the model ELM and assert the bias
    elm_models = [ run["ELM"] for run in experiment["runs"] ]

    for i in range(len(experiment["runs"])):
        model = elm_models[i]
        hidden_layer = model.steps[0][1]
        print(f"-> min(biases) = : {np.min(hidden_layer.biases)} and max(biases) = {np.max(hidden_layer.biases)}")

