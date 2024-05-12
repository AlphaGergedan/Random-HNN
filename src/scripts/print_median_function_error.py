"""
src/scripts/print_median_function_error.py

Script for printing the errors in a saved experiment, can be configured to print other errors as well 'mean', 'max' etc.
The runs in the saved experiment already includes different type of errors stored. No additional computation is needed,
this file just reads the saved values. See `src/main.py` for details.

One experiment looks like the following:

experiment = {
    "domain_params": domain_params,
    "elm_params": elm_params, "uswim_params": uswim_params, "aswim_params": aswim_params, "swim_params": swim_params,
    "runs": []
}

where each run contains:
    {
        "ELM", "U-SWIM", ...
        "train_random_seed": train_random_seed, "test_random_seed": test_random_seed, "model_random_seed": model_random_seed,
        "train_function_errors": { "ELM": .. }, "train_gradient_errors": { "ELM": .. }, "test_function_errors": {"ELM": ..}, "test_gradient_errors": {"ELM": ..},
        "train_times": {"ELM": ..},
    }

For each model, a run contains the trained model, its errors and train times, under the name of the model: "ELM" or "U-SWIM" or "A-SWIM" or "SWIM"

A script `get_summmary` inside `src/utils/utils.py` does the job of reading these as used below.
"""

#########################################
## FOR RETURNING MEDIAN FUNCTION ERROR ##
#########################################
import os, sys
directory_to_prepend = os.path.abspath("./src")
if not os.path.exists(directory_to_prepend):
    raise RuntimeError("src directory not found")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from joblib import load
import argparse
from utils.utils import get_summary

###############
## READ ARGS ##
###############

parser = argparse.ArgumentParser(prog='ERROR PRINTER', description='Print median of function error')
parser.add_argument('experiment_path', type=str, help='Path to the experiment file.')

args = parser.parse_args()

print(args.experiment_path)

experiment = load(args.experiment_path)
print(f"domain params: {experiment['domain_params']}")
print(f"elm params as example: {experiment['elm_params']}")
print(f"uswim params as example: {experiment['uswim_params']}")
print(f"aswim params as example: {experiment['aswim_params']}")
print(f"swim params as example: {experiment['swim_params']}")
# print(get_summary(experiment, ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], ['test'], ['function_errors', 'gradient_errors'], ['l2_error_relative', 'mean_squared_error'], ['median']))
print(get_summary(experiment, ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], ['test'], ['function_errors'], ['l2_error_relative'], ['mean']))
# print(get_summary(experiment, ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], ['test'], ['function_errors'], ['l2_error_relative'], ['median', 'mean']))
