#####################################
## FOR PLOTTING EXPERIMENT RESULTS ##
#####################################
import os, sys
directory_to_prepend = os.path.abspath("./src")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from joblib import load
import numpy as np
import argparse
from utils.utils import get_results, get_summary
from utils.plot import plot_comparison, plot_hists
import matplotlib.pyplot as plt

###############
## READ ARGS ##
###############

parser = argparse.ArgumentParser(prog='PLOTTER', description='Plot experiment results')
parser.add_argument('output_dir', type=str, help='Output directory for the experiment results')
parser.add_argument('-f','--file', nargs='+', help='Experiment output file locations', required=True)
parser.add_argument('-l','--log-scale', help='Plot in log scale', action='store_true')

args = parser.parse_args()

model_names_to_plot = ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM']
types_to_plot = ['losses', 'errors']
error_functions_to_plot = ['l2_error_relative']

#############################
## PLOT EXPERIMENT RESULTS ##
#############################

# we will fill the table below and aggregate the results of the experiments
results = {
    "train": {
        "losses": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
        "errors": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
    },
    "test": {
        "losses": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
        "errors": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
    },
}

# fill the domain sizes here
domain_sizes = []

for f in args.file:
    # { domain_params, elm_params, uswim_params, aswim_params, swim_params, runs }
    experiment = load(f)

    # will be used in the general plots
    domain_sizes.append(experiment['domain_params']['q_train'] * experiment['domain_params']['p_train'])

    print(f'-> Experiment {f} is loaded')
    print(f'-> Domain Params are read as:')
    print(experiment['domain_params'])
    print()

    print(get_summary(experiment, model_names_to_plot, ['train', 'test'], types_to_plot, error_functions_to_plot, stats=['min', 'median', 'mean']))

    n_runs = dict(experiment['domain_params'])['repeat']

    # for each experiment plot train_loss, test_loss, train_error, test_error plots (4)
    # where on the x-axis we see run number and on y-axis we see error function
    for dataset in ['train', 'test']:
        for type in ['losses', 'errors']:
            elm_l2_rel = get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')
            uswim_l2_rel = get_results(experiment, 'U-SWIM', dataset, type, 'l2_error_relative')
            aswim_l2_rel = get_results(experiment, 'A-SWIM', dataset, type, 'l2_error_relative')
            swim_l2_rel = get_results(experiment, 'SWIM', dataset, type, 'l2_error_relative')
            ys = [ elm_l2_rel, uswim_l2_rel, aswim_l2_rel, swim_l2_rel ]
            plot_comparison(range(1, n_runs+1), ys, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [1,n_runs],
                            r'Run', r'$l_2$ error relative', ['ELM ' + dataset + ' ' + type, 'U-SWIM ' + dataset + ' ' + type, 'A-SWIM ' + dataset + ' ' + type, 'SWIM ' + dataset + ' ' + type],
                            logscale=True, verbose=False, save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_l2_rel.pdf'))

            # plot average weight and biases hists
            # plot_hists(experiment, 20, save=os.path.join(args.output_dir, os.path.basename(f)))

            # aggregate experiment results
            for model in model_names_to_plot:
                results[dataset][type][model]['min'].append(np.min(get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')))
                results[dataset][type][model]['median'].append(np.median(get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')))
                results[dataset][type][model]['mean'].append(np.mean(get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')))
                results[dataset][type][model]['max'].append(np.max(get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')))


# create a general plot (medians, means, mins) with train_loss, test_loss, train_error, test_error plots (4, 4, 4)
# where on the x-axis we see domain size and on y-axis we see error function
for dataset in ['train', 'test']:
    for type in ['losses', 'errors']:
        for stat in ['median', 'mean', 'min']:
            ys = [
                results[dataset][type]['ELM'][stat],
                results[dataset][type]['U-SWIM'][stat],
                results[dataset][type]['A-SWIM'][stat],
                results[dataset][type]['SWIM'][stat],
            ]
            plot_comparison(domain_sizes, ys, domain_sizes, [np.min(domain_sizes), np.max(domain_sizes)],
                            r'Train Set Size', r'$l_2$ error relative ' + stat + ' of 100 runs', ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'],
                            logscale=True, verbose=False, save=os.path.join(args.output_dir, '_'.join([stat, dataset, type, 'l2_rel']) + '.pdf'))

