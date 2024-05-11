#########################################
## FOR RETURNING MEDIAN FUNCTION ERROR ##
#########################################
import os, sys
directory_to_prepend = os.path.abspath("../") # append src
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
