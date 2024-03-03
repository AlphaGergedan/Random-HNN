######################################
## FOR COMBINING EXPERIMENT RESULTS ##
######################################
import os, sys
directory_to_prepend = os.path.abspath("./src")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from joblib import load, dump
import argparse

###############
## READ ARGS ##
###############

parser = argparse.ArgumentParser(prog='COMBINER', description='Combines experiment results together')
parser.add_argument('output_file', type=str, help='Path to the output file to save the combined experiment results')
parser.add_argument('-f','--file', nargs='+', help='Experiment output file locations', required=True)

args = parser.parse_args()

################################
## COMBINE EXPERIMENT RESULTS ##
################################

# we will fill the table below and aggregate the results of the experiments
combined_experiments = {
    "domain_params": None,
    "elm_params": None, "uswim_params": None, "aswim_params": None, "swim_params": None,
    "runs": []
}

for f in args.file:
    # experiment = {
        # "domain_params": domain_params,
        # "elm_params": elm_params, "uswim_params": uswim_params, "aswim_params": aswim_params, "swim_params": swim_params,
        # "runs": []
    # }
    current_experiment = load(f)

    print(f"keys of the current experiment is {current_experiment.keys()}")
    print(f"DOMAIN PARAMS: {current_experiment["domain_params"]}")

    # if this is the first iterated file then set it as the base
    if combined_experiments["domain_params"] is None:
        combined_experiments = current_experiment
        print(f"Initial set of domain params as {combined_experiments['domain_params']}")
        print(f"Initial set of elm params as {combined_experiments['elm_params']}")
        print(f"Initial set of uswim params as {combined_experiments['uswim_params']}")
        print(f"Initial set of aswim params as {combined_experiments['aswim_params']}")
        print(f"Initial set of swim params as {combined_experiments['swim_params']}")
        print(f"Initial set of runs of length {len(combined_experiments['runs'])}")
        continue

    # assert domain params
    for key in combined_experiments["domain_params"].keys():
        if key != "H" and key != "dH" and key != "train_random_seed_start" and key != "test_random_seed_start" and key != "repeat":
            assert combined_experiments["domain_params"][key] == current_experiment["domain_params"][key]

    # assert model params
    for key in combined_experiments["elm_params"].keys():
        if key != "model_random_seed_start":
            print(f"comparing key {key} in elm param")
            assert combined_experiments["elm_params"][key] == current_experiment["elm_params"][key]
    for key in combined_experiments["uswim_params"].keys():
        if key != "model_random_seed_start":
            assert combined_experiments["uswim_params"][key] == current_experiment["uswim_params"][key]
    for key in combined_experiments["aswim_params"].keys():
        if key != "model_random_seed_start":
            assert combined_experiments["aswim_params"][key] == current_experiment["aswim_params"][key]
    for key in combined_experiments["swim_params"].keys():
        if key != "model_random_seed_start":
            assert combined_experiments["swim_params"][key] == current_experiment["swim_params"][key]

    # current_run = {
        # # trained models will be saved too
        # "train_random_seed": train_random_seed, "test_random_seed": test_random_seed, "model_random_seed": model_random_seed,
        # "train_errors": {}, "train_losses": {}, "test_errors": {}, "test_losses": {},
        # "train_times": {},
    # }
    print(f"Combining runs of length {len(current_experiment['runs'])} into the combined runs of length {len(combined_experiments['runs'])}")

    # append the runs
    combined_experiments["runs"] += current_experiment["runs"]

print(f"Final result run length: {len(combined_experiments['runs'])}")

combined_experiments["domain_params"]["repeat"] = len(combined_experiments["runs"])

# TODO assert random seeds are being incremented in each run
# train_random_seeds = combined_experiments["runs"][0]["train_random_seed"]
train_random_seeds = [ current_run["train_random_seed"] for current_run in combined_experiments["runs"] ]
test_random_seeds = [ current_run["test_random_seed"] for current_run in combined_experiments["runs"] ]
model_random_seeds = [ current_run["model_random_seed"] for current_run in combined_experiments["runs"] ]

print()
print(f"train_random_seeds: {train_random_seeds}")
print()
print(f"test_random_seeds: {test_random_seeds}")
print()
print(f"model_random_seeds: {model_random_seeds}")
print()

# save the aggregated results
dump(combined_experiments, args.output_file)
print('-> Saved experiment results under: ' + args.output_file)
