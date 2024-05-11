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

# fill the n_neurons here
n_neurons = []
n_runs = 0

for f in args.file:
    # { domain_params, elm_params, uswim_params, aswim_params, swim_params, runs }
    experiment = load(f)

    # will be used in the general plots
    current_n_neurons = experiment['elm_params']['n_neurons'][0]
    n_neurons.append(current_n_neurons)

    print(f'-> Experiment {f} is loaded')
    print(f'-> Domain Params are read as:')
    print(experiment['domain_params'])
    print()

    print(get_summary(experiment, model_names_to_plot, ['train', 'test'], types_to_plot, error_functions_to_plot, stats=['min', 'median', 'mean']))

    n_runs = dict(experiment['domain_params'])['repeat']

    # for each experiment plot train_loss, test_loss, train_error, test_error plots (4)
    # where on the x-axis we see run number and on y-axis we see error function
    for dataset in ['train', 'test']:
        for type in ['gradient_errors', 'function_errors']:
            elm_l2_rel = get_results(experiment, 'ELM', dataset, type, 'l2_error_relative')
            uswim_l2_rel = get_results(experiment, 'U-SWIM', dataset, type, 'l2_error_relative')
            aswim_l2_rel = get_results(experiment, 'A-SWIM', dataset, type, 'l2_error_relative')
            swim_l2_rel = get_results(experiment, 'SWIM', dataset, type, 'l2_error_relative')
            ys = [ elm_l2_rel, uswim_l2_rel, aswim_l2_rel, swim_l2_rel ]
            if n_runs == 10:
                plot_comparison(range(1, n_runs+1), ys, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1,n_runs],
                                r'$i$th run', r'Rel. $L^2$ error', legends=None,
                                logscale=args.logscale, verbose=False,
                                save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_l2_rel.pdf'))
            elif n_runs == 100:
                plot_comparison(range(1, n_runs+1), ys, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [1,n_runs],
                                r'$i$th run', r'Rel. $L^2$ error', legends=None,
                                logscale=args.logscale, verbose=False,
                                save=os.path.join(args.output_dir, os.path.basename(f) + '_' + dataset + '_' + type + '_l2_rel.pdf'))

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
        plot_comparison(range(1, n_runs+1), ys, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [1,n_runs],
                        r'$i$th run', r'Time in [s]', legends=None,
                        logscale=args.logscale, verbose=False,
                        save=os.path.join(args.output_dir, os.path.basename(f) + '_train_times.pdf'))
    elif n_runs == 10:
        plot_comparison(range(1, n_runs+1), ys, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1,n_runs],
                        r'$i$th run', r'Time in [s]', legends=None,
                        logscale=args.logscale, verbose=False,
                        save=os.path.join(args.output_dir, os.path.basename(f) + '_train_times.pdf'))
    # aggregate training times to plot scaling
    for model in model_names_to_plot:
        results['train_times'][model]['min'].append(np.min(get_train_times(experiment, model)))
        results['train_times'][model]['median'].append(np.median(get_train_times(experiment, model)))
        results['train_times'][model]['mean'].append(np.mean(get_train_times(experiment, model)))
        results['train_times'][model]['max'].append(np.max(get_train_times(experiment, model)))

# create a general plot (medians, means, mins) with train_loss, test_loss, train_error, test_error plots (4, 4, 4)
# where on the x-axis we see domain size and on y-axis we see error function
for dataset in ['train', 'test']:
    for type in ['gradient_errors', 'function_errors']:
        for stat in ['median', 'mean', 'min', 'max']:
            ys = [
                results[dataset][type]['ELM'][stat],
                results[dataset][type]['U-SWIM'][stat],
                results[dataset][type]['A-SWIM'][stat],
                results[dataset][type]['SWIM'][stat],
            ]
            plot_comparison(n_neurons, ys, n_neurons, [np.min(n_neurons), np.max(n_neurons)],
                            r'Network Width', r'Rel. $L^2$ error, ' + stat + f' of {n_runs} runs', ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'],
                            logscale=args.logscale, verbose=False,
                            save=os.path.join(args.output_dir, '_'.join([stat, dataset, type, 'l2_rel']) + '.pdf'), rotate_xticks=True)


# plot training time scaling
for stat in ['median', 'mean', 'min', 'max']:
    ys = [
        results['train_times']['ELM'][stat],
        results['train_times']['U-SWIM'][stat],
        results['train_times']['A-SWIM'][stat],
        results['train_times']['SWIM'][stat],
    ]
    plot_comparison(n_neurons, ys, n_neurons, [np.min(n_neurons), np.max(n_neurons)],
                    r'Network Width', r'Time in [s], ' + stat + f' of {n_runs} runs', ['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'],
                    logscale=args.logscale, verbose=False,
                    save=os.path.join(args.output_dir, '_'.join([stat, 'train_times']) + '.pdf'), rotate_xticks=True)
