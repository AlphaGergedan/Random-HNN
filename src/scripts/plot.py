"""
src/scripts/plot.py

For creating plots from saved results, all train set size scaling plots are created using this script.
"""

#####################################
## FOR PLOTTING EXPERIMENT RESULTS ##
#####################################
import os, sys
directory_to_prepend = os.path.abspath("./src")
if not os.path.exists(directory_to_prepend):
    raise RuntimeError("src directory not found")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

from joblib import load
import numpy as np
import argparse
from utils.utils import get_results, get_train_times, get_summary
from utils.plot import plot_comparison, plot_hists

###############
## READ ARGS ##
###############

parser = argparse.ArgumentParser(prog='PLOTTER', description='Plot experiment results')
parser.add_argument('output_dir', type=str, help='Output directory for the experiment results')
parser.add_argument('-f','--file', nargs='+', help='Experiment output file locations', required=True)
parser.add_argument('-hist', help='Plot weight and biases mean histogramms', action='store_true', default=False)
parser.add_argument('-logscale', help='Plot in log scale', action='store_true', default=False)

args = parser.parse_args()

model_names_to_plot = ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM']
types_to_plot = ['gradient_errors', 'function_errors']
error_functions_to_plot = ['l2_error_relative']

#############################
## PLOT EXPERIMENT RESULTS ##
#############################

# we will fill the table below and aggregate the results of the experiments
results = {
    "train": {
        "gradient_errors": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
        "function_errors": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
    },
    "test": {
        "gradient_errors": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
        "function_errors": {
            "ELM": { "min": [], "median": [], "mean": [], "max": [], },
            "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
            "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        },
    },
    "train_times": {
        "ELM": { "min": [], "median": [], "mean": [], "max": [], },
        "U-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        "A-SWIM": { "min": [], "median": [], "mean": [], "max": [], },
        "SWIM": { "min": [], "median": [], "mean": [], "max": [], },
    }
}


# fill the domain sizes here
domain_sizes = []
n_runs = 0

for f in args.file:
    # { domain_params, elm_params, uswim_params, aswim_params, swim_params, runs }
    experiment = load(f)

    # will be used in the general plots
    domain_size = experiment['domain_params']['train_set_size']
    domain_sizes.append(domain_size)


    print(f'-> Experiment {f} is loaded')
    print(f'-> Domain Params are read as:')
    print(experiment['domain_params'])
    print()

    print(f'-> Experiment keys: {experiment.keys()}')
    print(f'-> n_runs         : {len(experiment["runs"])}')
    print(f'-> run keys       : {experiment["runs"][0].keys()}')
    print(f'-> run[funcerr]   : {experiment["runs"][0]["train_function_errors"].keys()}')

    print(get_summary(experiment, model_names_to_plot, ['train', 'test'], types_to_plot, ['l2_error_relative', 'mean_squared_error'], stats=['min', 'median', 'mean']))

    n_runs = dict(experiment['domain_params'])['repeat']
    print(f"n_runs: {n_runs}")
    assert n_runs == 100 or n_runs == 50 or n_runs == 10 or n_runs == 5

    # for each experiment plot train_loss, test_loss, train_error, test_error plots (4)
    # where on the x-axis we see run number and on y-axis we see error function
    for dataset in ['train', 'test']:
        for type in types_to_plot:
            elm_l2_rel = get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')
            uswim_l2_rel = get_results(experiment, 'U-SWIM', dataset, type, 'l2_error_relative')
            aswim_l2_rel = get_results(experiment, 'A-SWIM', dataset, type, 'l2_error_relative')
            swim_l2_rel = get_results(experiment, 'SWIM', dataset, type, 'l2_error_relative')
            # elm_l2_rel = get_results(experiment, 'ELM', dataset, type, 'mean_squared_error')
            # uswim_l2_rel = get_results(experiment, 'U-SWIM', dataset, type, 'mean_squared_error')
            # aswim_l2_rel = get_results(experiment, 'A-SWIM', dataset, type, 'mean_squared_error')
            # swim_l2_rel = get_results(experiment, 'SWIM', dataset, type, 'mean_squared_error')
            ys = [ elm_l2_rel, uswim_l2_rel, aswim_l2_rel, swim_l2_rel ]
            if n_runs == 10:
                plot_comparison(range(1, n_runs+1), ys, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1,n_runs],
                                r'$i$th run', r'Rel. $L^2$ error', legends=None,
                                # r'$i$th Run', r'MSE', ['ELM ' + dataset + ' ' + type, 'U-SWIM ' + dataset + ' ' + type, 'A-SWIM ' + dataset + ' ' + type, 'SWIM ' + dataset + ' ' + type],
                                logscale=args.logscale, verbose=False,
                                save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_l2_rel.pdf'))
                                # save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_mse.pdf'))
            elif n_runs == 100:
                plot_comparison(range(1, n_runs+1), ys, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [1,n_runs],
                                r'$i$th run', r'Rel. $L^2$ error', legends=None,
                                # r'$i$th Run', r'MSE', ['ELM ' + dataset + ' ' + type, 'U-SWIM ' + dataset + ' ' + type, 'A-SWIM ' + dataset + ' ' + type, 'SWIM ' + dataset + ' ' + type],
                                logscale=args.logscale, verbose=False,
                                save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_l2_rel.pdf'))
                                # save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_mse.pdf'))
            elif n_runs == 5:
                plot_comparison(range(1, n_runs+1), ys, [0, 1, 2, 3, 4, 5], [1,n_runs],
                                r'$i$th run', r'Rel. $L^2$ error', legends=None,
                                # r'$i$th Run', r'MSE', ['ELM ' + dataset + ' ' + type, 'U-SWIM ' + dataset + ' ' + type, 'A-SWIM ' + dataset + ' ' + type, 'SWIM ' + dataset + ' ' + type],
                                logscale=args.logscale, verbose=False,
                                save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_l2_rel.pdf'))
                                # save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_mse.pdf'))
            else: raise ValueError("Unexpected number of runs")

            if args.hist:
                # plot average weight and biases hists
                plot_hists(experiment, 20, save=os.path.join(args.output_dir, os.path.basename(f)))

            # aggregate experiment results
            for model in model_names_to_plot:
                results[dataset][type][model]['min'].append(np.min(get_results(experiment, model, dataset, type, 'l2_error_relative')))
                results[dataset][type][model]['median'].append(np.median(get_results(experiment, model, dataset, type, 'l2_error_relative')))
                results[dataset][type][model]['mean'].append(np.mean(get_results(experiment, model, dataset, type, 'l2_error_relative')))
                results[dataset][type][model]['max'].append(np.max(get_results(experiment, model, dataset, type, 'l2_error_relative')))
                # results[dataset][type][model]['min'].append(np.min(get_results(experiment, model, dataset, type, 'mean_squared_error')))
                # results[dataset][type][model]['median'].append(np.median(get_results(experiment, model, dataset, type, 'mean_squared_error')))
                # results[dataset][type][model]['mean'].append(np.mean(get_results(experiment, model, dataset, type, 'mean_squared_error')))
                # results[dataset][type][model]['max'].append(np.max(get_results(experiment, model, dataset, type, 'mean_squared_error')))


    # plot train times
    elm_train_times = get_train_times(experiment, 'ELM')
    uswim_train_times = get_train_times(experiment, 'U-SWIM')
    aswim_train_times = get_train_times(experiment, 'A-SWIM')
    swim_train_times = get_train_times(experiment, 'SWIM')
    ys = [ elm_train_times, uswim_train_times, aswim_train_times, swim_train_times ]
    if n_runs == 100:
        plot_comparison(range(1, n_runs+1), ys, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [1,n_runs],
                        r'$i$th run', r'Time in [s]', legends=None,
                        logscale=args.logscale, verbose=False,
                        save=os.path.join(args.output_dir, os.path.basename(f) + '_train_times.pdf'))
    elif n_runs == 10:
        plot_comparison(range(1, n_runs+1), ys, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1,n_runs],
                        r'$i$th run', r'Time in [s]', legends=None,
                        logscale=args.logscale, verbose=False,
                        save=os.path.join(args.output_dir, os.path.basename(f) + '_train_times.pdf'))
    elif n_runs == 5:
        plot_comparison(range(1, n_runs+1), ys, [1, 2, 3, 4, 5], [1,n_runs],
                        r'$i$th run', r'Time in [s]', legends=None,
                        logscale=args.logscale, verbose=False,
                        save=os.path.join(args.output_dir, os.path.basename(f) + '_train_times.pdf'))
    else: raise ValueError("Unexpected number of runs")
    # aggregate training times to plot scaling
    for model in model_names_to_plot:
        results['train_times'][model]['min'].append(np.min(get_train_times(experiment, model)))
        results['train_times'][model]['median'].append(np.median(get_train_times(experiment, model)))
        results['train_times'][model]['mean'].append(np.mean(get_train_times(experiment, model)))
        results['train_times'][model]['max'].append(np.max(get_train_times(experiment, model)))

# create a general plot (medians, means, mins) with train_loss, test_loss, train_error, test_error plots (4, 4, 4)
# where on the x-axis we see domain size and on y-axis we see error function
for dataset in ['train', 'test']:
    for type in types_to_plot:
        for stat in ['median', 'mean', 'min', 'max']:
            ys = [
                results[dataset][type]['ELM'][stat],
                results[dataset][type]['U-SWIM'][stat],
                results[dataset][type]['A-SWIM'][stat],
                results[dataset][type]['SWIM'][stat],
            ]
            plot_comparison(domain_sizes, ys, domain_sizes, [np.min(domain_sizes), np.max(domain_sizes)],
                            r'Train set size', r'Rel. $L^2$ error, ' + stat + f' of {n_runs} runs', ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'],
                            # r'Train Set Size', r'MSE, ' + stat + f' of {n_runs} runs', ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'],
                            logscale=args.logscale, verbose=False,
                            save=os.path.join(args.output_dir, '_'.join([stat, dataset, type, 'l2_rel']) + '.pdf'), rotate_xticks=True)
                            # save=os.path.join(args.output_dir, '_'.join([stat, dataset, type, 'mse']) + '.pdf'), rotate_xticks=True)


# plot training time scaling
for stat in ['median', 'mean', 'min', 'max']:
    ys = [
        results['train_times']['ELM'][stat],
        results['train_times']['U-SWIM'][stat],
        results['train_times']['A-SWIM'][stat],
        results['train_times']['SWIM'][stat],
    ]
    plot_comparison(domain_sizes, ys, domain_sizes, [np.min(domain_sizes), np.max(domain_sizes)],
                    r'Train set size', r'Time in [s], ' + stat + f' of {n_runs} runs', ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'],
                    logscale=args.logscale, verbose=False,
                    save=os.path.join(args.output_dir, '_'.join([stat, 'train_times']) + '.pdf'), rotate_xticks=True)
