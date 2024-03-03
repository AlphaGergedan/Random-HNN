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

    # if this is the first iterated file then set it as the base
    if combined_experiments["domain_params"] is None:
        combined_experiments = current_experiment
        continue

    # assert domain and model params are same
    assert current_experiment["domain_params"] == combined_experiments["domain_params"]
    assert current_experiment["elm_params"] == combined_experiments["elm_params"]
    assert current_experiment["uswim_params"] == combined_experiments["uswim_params"]
    assert current_experiment["aswim_params"] == combined_experiments["aswim_params"]
    assert current_experiment["swim_params"] == combined_experiments["swim_params"]

    # append the runs
    combined_experiments["runs"] += current_experiment["runs"]


# save the aggregated results
dump(combined_experiments, args.output_file)
print('-> Saved experiment results under: ' + args.output_file)
