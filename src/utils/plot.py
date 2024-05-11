import matplotlib.pyplot as plt
import numpy as np
from error_functions.index import mean_absolute_error, mean_squared_error, l2_error, l2_error_relative

COLOR_TRUTH = '#000000'
COLOR_ELM = '#05c9ff' # blue
COLOR_USWIM = '#ff0000' # red
COLOR_ASWIM = '#28fa02' # green
COLOR_SWIM = '#ffff00' # yellow


def plot_1d(x, y,
            label='f(x)', xpredict=None, ypredict=None, labelpredict='Approximation',
            xlim=None, ylim=None, xlabel='x', ylabel='y', title=None, verbose=True, save=''):
    """
    x : x points
    y : y points
    xpredict : x points to predict
    ypredict : predictions at xpredict
    labelpredict : label for predictions
    xlim : (x_start,x_end)
    ylim : (y_start,y_end)
    xlabel : label for x-axis
    ylabel : label for y-axis
    title : title of the figure
    """
    plt.clf()
    fig = plt.figure()
    if title is not None:
        ax = plt.gca()
        ax.set_title(title)
    if xpredict is not None and ypredict is not None:
        plt.plot(xpredict, ypredict,'ro',markevery=1,label=labelpredict);
    plt.plot(x,y,label=label)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save != '':
        fig.savefig(save)
    if verbose:
        plt.show()
        print()
    else:
        plt.close()

def plot_2d(grid, xrange, yrange,
            vmin=None, vmax=None, extent=None, contourlines=10,
            xlabel='x', ylabel='y', aspect=2.5, title=None, verbose=True, save='', xlim=None, dpi=100, colorbar_v=10):
    """
    @param grid: 2D meshgrid
    @param xrange, yrange: value ranges on q and p axes
    @param vmin, vmax: min max value for colorbar
    @param contourlines: number of contour lines
    @param xlabel, ylabel: label for axes
    @param aspect: aspect ratio of the plot
    @param title: title of the plot
    @param verbose: whether to show plot
    @param save: path to save plot, if empty string no plot is saved
    """
    plt.clf()
    # fig, ax = plt.figure(figsize=(5,5) ,dpi=50)
    fig, ax = plt.subplots(dpi=dpi)
    # ax = plt.gca()
    im = ax.imshow(grid, extent=extent, vmin=vmin, vmax=vmax)
    v_colorbar = np.linspace(np.min(grid), np.max(grid), colorbar_v, endpoint=True)
    if contourlines > 0:
        ax.contour(xrange, yrange, grid, contourlines, colors='white', linewidths=0.5, linestyles='solid')
    fig.colorbar(im, ax=ax, fraction=0.049*(yrange.shape[0] / xrange.shape[0]), pad=0.04, label=r'$\nabla_p \mathcal{H}(q,p)$', ticks=v_colorbar)
    # set ticks for the colorbar with min and max values
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim) if xlim else None
    ax.set_aspect(aspect)
    fig.tight_layout()
    if title is not None:
        ax.set_title(title)
    if save != '':
        # fig.savefig(save)
        plt.savefig(save,bbox_inches='tight')
    if verbose:
        plt.show()
        print()
    else:
        plt.close()


def plot_poincare(grid, xrange, yrange, extent, contourlines, xlabel, ylabel, aspect, title,
                  fixed_dims={"q1": 0, "q2":0}, verbose=True, save=''):
    """
    Plots 2d poincare map using the plot_2d function
    This function is just a wrapper

    @param grid: 2D meshgrid
    @param xrange, yrange: value ranges on q and p axes
    @param contourlines: number of contour lines
    @param xlabel, ylabel: label for axes
    @param aspect: aspect ratio of the plot
    @param title: title of the plot
    @param fixed_dims: dict for fixed dimensions to plot for poincare
    @param verbose: whether to show plot
    @param save: path to save plot, if empty string no plot is saved
    """
    fixed_dims_names = list(fixed_dims.keys())
    fixed_dims_values = list(fixed_dims.values())
    title = title + "\n" + "with " + fixed_dims_names[0] + "=" + str(fixed_dims_values[0]) + "," + fixed_dims_names[1] + "=" + str(fixed_dims_values[1])
    plot_2d(grid, xrange, yrange, extent=extent, contourlines=contourlines, xlabel=xlabel, ylabel=ylabel, aspect=aspect, title=title, save=save, verbose=verbose)

def plot_comparison(x, ys, xticks, xlim, xlabel, ylabel, legends, logscale=False, verbose=False, save='', rotate_xticks=False):
    assert len(ys) == 4 # we have 4 models right now
    assert legends is None or len(legends) == 4

    if logscale:
       ys = [ np.log10(y) for y in ys ]

    plt.clf()
    plt.figure(figsize=(4, 4), dpi=100)
    plt.plot(x, ys[0], c=COLOR_ELM, marker="+")    # light blue
    plt.plot(x, ys[1], c=COLOR_USWIM, marker=".")   # light red
    plt.plot(x, ys[2], c=COLOR_ASWIM, marker="x")  # light green
    plt.plot(x, ys[3], c=COLOR_SWIM, marker="*")  # light yellow
    plt.xlabel(xlabel)
    plt.xlim(xlim)

    if rotate_xticks:
        # Rotate x-ticks by 45 degrees and align them to the right
        plt.xticks(xticks, rotation=60, ha='right', fontsize=6)
    else:
        plt.xticks(xticks)

    if logscale:
        plt.ylabel(ylabel + r', on $\log_{10}$ scale')
    else:
        plt.ylabel(ylabel)

    plt.grid()
    if legends:
        plt.legend(legends, fontsize='medium', shadow=True, loc='best')
    plt.tight_layout()
    plt.savefig(save)
    #plt.savefig(save,bbox_inches='tight')
    plt.close()

def plot_comparison_solvetrue(x, ys, xticks, xlim, xlabel, ylabel, legends, logscale=False, verbose=False, save='', rotate_xticks=False):
    assert len(ys) == 3 # we have 4 models right now
    assert legends is None or len(legends) == 3

    if logscale:
       ys = [ np.log10(y) for y in ys ]

    plt.clf()
    plt.plot(x, ys[0], c=COLOR_ELM, marker="+")    # light blue
    plt.plot(x, ys[1], c=COLOR_USWIM, marker=".")   # light red
    plt.plot(x, ys[2], c=COLOR_SWIM, marker="*")  # light yellow
    plt.xlabel(xlabel)
    plt.xlim(xlim)

    if rotate_xticks:
        # Rotate x-ticks by 45 degrees and align them to the right
        plt.xticks(xticks, rotation=60, ha='right', fontsize=6)
    else:
        plt.xticks(xticks)

    if logscale:
        plt.ylabel(ylabel + r', on $log_{10}$ scale')
    else:
        plt.ylabel(ylabel)

    plt.grid()
    plt.legend(legends, fontsize='medium', shadow=True, loc='best')
    plt.tight_layout()
    plt.savefig(save)
    plt.close()

def plot_hists(experiment, nhists, save=''):
    # ELM
    weights = np.mean([ run['ELM'].steps[0][1].weights for run in experiment['runs'] ], axis=1)
    biases = np.mean([ run['ELM'].steps[0][1].biases for run in experiment['runs']], axis=1)
    plt.clf()
    plt.hist(weights, nhists)
    plt.savefig(save + '_elm_weights_hist.png')
    plt.close()
    plt.clf()
    plt.hist(biases, nhists)
    plt.savefig(save + '_elm_biases_hist.png')
    plt.close()
    # U-SWIM
    weights = np.mean([ run['U-SWIM'].steps[0][1].weights for run in experiment['runs'] ], axis=1)
    biases = np.mean([ run['U-SWIM'].steps[0][1].biases for run in experiment['runs']], axis=1)
    plt.clf()
    plt.hist(weights, nhists)
    plt.savefig(save + '_uswim_weights_hist.png')
    plt.close()
    plt.clf()
    plt.hist(biases, nhists)
    plt.savefig(save + '_uswim_biases_hist.png')
    plt.close()
    # A-SWIM
    weights = np.mean([ run['A-SWIM'].steps[0][1].weights for run in experiment['runs'] ], axis=1)
    biases = np.mean([ run['A-SWIM'].steps[0][1].biases for run in experiment['runs']], axis=1)
    plt.clf()
    plt.hist(weights, nhists)
    plt.savefig(save + '_aswim_weights_hist.png')
    plt.close()
    plt.clf()
    plt.hist(biases, nhists)
    plt.savefig(save + '_aswim_biases_hist.png')
    plt.close()
    # SWIM
    weights = np.mean([ run['SWIM'].steps[0][1].weights for run in experiment['runs'] ], axis=1)
    biases = np.mean([ run['SWIM'].steps[0][1].biases for run in experiment['runs']], axis=1)
    plt.clf()
    plt.hist(weights, nhists)
    plt.savefig(save + '_swim_weights_hist.png')
    plt.close()
    plt.clf()
    plt.hist(biases, nhists)
    plt.savefig(save + '_swim_biases_hist.png')
    plt.close()

def plot_ground_truth_trajectory_4d(t_span, t_eval, traj_true, save='', verbose=False, **args):
    # GROUND TRUTH ONLY
    plt.clf()
    fig = plt.figure(figsize=[12,3], dpi=100)
    # t against q1
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:,0], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_1$')
    #plt.legend(['Ground truth'], shadow=True)
    plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # t against q2
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_2$')
    #plt.legend(['Ground truth'], shadow=True)
    plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # t against p1
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:,2], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_1$')
    #plt.legend(['Ground truth'], shadow=True)
    plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # t against p2
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_2$')
    #plt.legend(['Ground truth'], shadow=True)
    plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # if verbose
    plt.show() if verbose else None
    fig.savefig('timeplots_' + save) if save else None
    plt.clf()

    # poincares
    fig = plt.figure(figsize=[12,3], dpi=100)
    # q1 against p1
    plt.subplot(1,4,1)
    plt.plot(traj_true[:,0], traj_true[:,2], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$p_1$')
    #plt.legend(['Ground truth'], shadow=True)
    # plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # q2 against p2
    plt.subplot(1,4,2)
    plt.plot(traj_true[:,1], traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$q_2$')
    plt.ylabel(r'$p_2$')
    #plt.legend(['Ground truth'], shadow=True)
    # plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # q1 against q2
    plt.subplot(1,4,3)
    plt.plot(traj_true[:,0], traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    #plt.legend(['Ground truth'], shadow=True)
    # plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # p1 against p2
    plt.subplot(1,4,4)
    plt.plot(traj_true[:,2], traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$p_1$')
    plt.ylabel(r'$p_2$')
    #plt.legend(['Ground truth'], shadow=True)
    # plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # if verbose
    plt.show() if verbose else None
    fig.savefig('poincares_' + save) if save else None
    plt.clf()

def plot_ground_truth_trajectory_2d(t_span, t_eval, traj_true, save='', verbose=False, **args):
    COLOR_TRUTH = '#000000'
    # GROUND TRUTH ONLY
    plt.clf()
    fig = plt.figure(figsize=[6,3], dpi=100)
    # t against q
    plt.subplot(1,2,1)
    plt.plot(t_eval, traj_true[:,0], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q$')
    #plt.legend(['Ground truth'], shadow=True)
    plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # t against p
    plt.subplot(1,2,2)
    plt.plot(t_eval, traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', **args)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    #plt.legend(['Ground truth'], shadow=True)
    plt.xlim(t_span)
    plt.grid()
    plt.tight_layout()
    # if verbose
    plt.show() if verbose else None
    fig.savefig(save) if save else None
    plt.clf()


def plot_predicted_trajectories_2d(t_span, t_eval, traj_true, traj_pred_elm, traj_pred_uswim, traj_pred_aswim, traj_pred_swim,
                                   zorders, linewidths, verbose=False, save=[]):
    # q against p
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(traj_true[:, 0], traj_true[:, 1], c=COLOR_TRUTH, label='Ground Truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(traj_pred_elm[:, 0], traj_pred_elm[:, 1], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlabel(r'$q$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.subplot(1,4,2)
    plt.plot(traj_true[:, 0], traj_true[:, 1], c=COLOR_TRUTH, label='Ground Truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(traj_pred_uswim[:, 0], traj_pred_uswim[:, 1], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlabel(r'$q$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.subplot(1,4,3)
    plt.plot(traj_true[:, 0], traj_true[:, 1], c=COLOR_TRUTH, label='Ground Truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(traj_pred_aswim[:, 0], traj_pred_aswim[:, 1], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlabel(r'$q$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.subplot(1,4,4)
    plt.plot(traj_true[:, 0], traj_true[:, 1], c=COLOR_TRUTH, label='Ground Truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(traj_pred_swim[:, 0], traj_pred_swim[:, 1], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlabel(r'$q$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[0]) if save else None
    plt.clf()

    # t against q
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:, 0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_elm[:, 0], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:, 0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_uswim[:, 0], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:, 0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_aswim[:, 0], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:, 0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_swim[:, 0], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[1]) if save else None
    plt.clf()

    # t against p
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:, 1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_elm[:, 1], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:, 1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_uswim[:, 1], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:, 1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_aswim[:, 1], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:, 1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_swim[:, 1], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[2]) if save else None
    plt.clf()

def plot_predicted_trajectories_4d(t_span, t_eval, traj_true, traj_pred_elm, traj_pred_uswim, traj_pred_aswim, traj_pred_swim,
                                   zorders, linewidths, verbose=False, save=[]):
    # t against q1
    plt.clf()
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:,0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_elm[:,0], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_1$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:,0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_uswim[:,0], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_1$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:,0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_aswim[:,0], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_1$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:,0], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_swim[:,0], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_1$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[0]) if save else None
    plt.clf()

    # t against q2
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_elm[:,1], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_2$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_uswim[:,1], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_2$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_aswim[:,1], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_2$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:,1], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_swim[:,1], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_2$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[1]) if save else None
    plt.clf()

    # t against p1
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:,2], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_elm[:,2], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_1$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:,2], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_uswim[:,2], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_1$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:,2], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_aswim[:,2], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_1$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:,2], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_swim[:,2], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_1$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[2]) if save else None
    plt.clf()

    # t against p2
    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_elm[:,3], c=COLOR_ELM, label='ELM', zorder=zorders[1], linewidth=linewidths[1])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_2$')
    # plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,2)
    plt.plot(t_eval, traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_uswim[:,3], c=COLOR_USWIM, label='U-SWIM', zorder=zorders[2], linewidth=linewidths[2])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_2$')
    # plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,3)
    plt.plot(t_eval, traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_aswim[:,3], c=COLOR_ASWIM, label='A-SWIM', zorder=zorders[3], linewidth=linewidths[3])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_2$')
    # plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.subplot(1,4,4)
    plt.plot(t_eval, traj_true[:,3], c=COLOR_TRUTH, label='Ground truth', zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, traj_pred_swim[:,3], c=COLOR_SWIM, label='SWIM', zorder=zorders[4], linewidth=linewidths[4])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_2$')
    # plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.xlim(t_span)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save[3]) if save else None
    plt.clf()

def plot_predicted_trajectory_errors(t_span, t_eval, traj_true, traj_pred_elm, traj_pred_uswim, traj_pred_aswim, traj_pred_swim,
                                        zorders, linewidths, verbose=False, save=''):
    mse_error_elm = [ mean_squared_error(traj_true[i], traj_pred_elm[i]) for i in range(len(t_eval)) ]
    mse_error_uswim = [ mean_squared_error(traj_true[i], traj_pred_uswim[i]) for i in range(len(t_eval)) ]
    mse_error_aswim = [ mean_squared_error(traj_true[i], traj_pred_aswim[i]) for i in range(len(t_eval)) ]
    mse_error_swim = [ mean_squared_error(traj_true[i], traj_pred_swim[i]) for i in range(len(t_eval)) ]

    l2_error_elm = [ l2_error(traj_true[i], traj_pred_elm[i]) for i in range(len(t_eval)) ]
    l2_error_uswim = [ l2_error(traj_true[i], traj_pred_uswim[i]) for i in range(len(t_eval)) ]
    l2_error_aswim = [ l2_error(traj_true[i], traj_pred_aswim[i]) for i in range(len(t_eval)) ]
    l2_error_swim = [ l2_error(traj_true[i], traj_pred_swim[i]) for i in range(len(t_eval)) ]

    mae_error_elm = [ mean_absolute_error(traj_true[i], traj_pred_elm[i]) for i in range(len(t_eval)) ]
    mae_error_uswim = [ mean_absolute_error(traj_true[i], traj_pred_uswim[i]) for i in range(len(t_eval)) ]
    mae_error_aswim = [ mean_absolute_error(traj_true[i], traj_pred_aswim[i]) for i in range(len(t_eval)) ]
    mae_error_swim = [ mean_absolute_error(traj_true[i], traj_pred_swim[i]) for i in range(len(t_eval)) ]

    rl2_error_elm = [ l2_error_relative(traj_true[i], traj_pred_elm[i]) for i in range(len(t_eval)) ]
    rl2_error_uswim = [ l2_error_relative(traj_true[i], traj_pred_uswim[i]) for i in range(len(t_eval)) ]
    rl2_error_aswim = [ l2_error_relative(traj_true[i], traj_pred_aswim[i]) for i in range(len(t_eval)) ]
    rl2_error_swim = [ l2_error_relative(traj_true[i], traj_pred_swim[i]) for i in range(len(t_eval)) ]

    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, mse_error_elm, c=COLOR_ELM, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, mse_error_uswim, c=COLOR_USWIM, zorder=zorders[1], linewidth=linewidths[1])
    plt.plot(t_eval, mse_error_aswim, c=COLOR_ASWIM, zorder=zorders[2], linewidth=linewidths[2])
    plt.plot(t_eval, mse_error_swim, c=COLOR_SWIM, zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$MSE$')
    plt.legend(['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], shadow=True)
    plt.subplot(1,4,2)
    plt.plot(t_eval, rl2_error_elm, c=COLOR_ELM, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, rl2_error_uswim, c=COLOR_USWIM, zorder=zorders[1], linewidth=linewidths[1])
    plt.plot(t_eval, rl2_error_aswim, c=COLOR_ASWIM, zorder=zorders[2], linewidth=linewidths[2])
    plt.plot(t_eval, rl2_error_swim, c=COLOR_SWIM, zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'rel. $L^2$ error')
    plt.legend(['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], shadow=True)
    plt.subplot(1,4,3)
    plt.plot(t_eval, mae_error_elm, c=COLOR_ELM, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, mae_error_uswim, c=COLOR_USWIM, zorder=zorders[1], linewidth=linewidths[1])
    plt.plot(t_eval, mae_error_aswim, c=COLOR_ASWIM, zorder=zorders[2], linewidth=linewidths[2])
    plt.plot(t_eval, mae_error_swim, c=COLOR_SWIM, zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$MAE$')
    plt.legend(['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], shadow=True)
    plt.subplot(1,4,4)
    plt.plot(t_eval, l2_error_elm, c=COLOR_ELM, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, l2_error_uswim, c=COLOR_USWIM, zorder=zorders[1], linewidth=linewidths[1])
    plt.plot(t_eval, l2_error_aswim, c=COLOR_ASWIM, zorder=zorders[2], linewidth=linewidths[2])
    plt.plot(t_eval, l2_error_swim, c=COLOR_SWIM, zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$L^2$ error')
    plt.legend(['ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], shadow=True)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save) if save else None
    plt.clf()

def plot_predicted_trajectory_energy(t_span, t_eval, energy_true, energy_elm, energy_uswim, energy_aswim, energy_swim,
                                     zorders, linewidths, verbose=False, save=''):
    # first plot a total energy plot all together
    fig = plt.figure(figsize=[6,6], dpi=100)
    plt.plot(t_eval, energy_true, c=COLOR_TRUTH, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, energy_elm, c=COLOR_ELM, zorder=zorders[1], linewidth=linewidths[1])
    plt.plot(t_eval, energy_uswim, c=COLOR_USWIM, zorder=zorders[2], linewidth=linewidths[2])
    plt.plot(t_eval, energy_aswim, c=COLOR_ASWIM, zorder=zorders[3], linewidth=linewidths[3])
    plt.plot(t_eval, energy_swim, c=COLOR_SWIM, zorder=zorders[4], linewidth=linewidths[4])
    #plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{H}(q,p)$')
    plt.legend(['Ground truth', 'ELM', 'U-SWIM', 'A-SWIM', 'SWIM'], shadow=True)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig('allinone_' + save) if save else None
    plt.clf()

    fig = plt.figure(figsize=[12,3], dpi=100)
    plt.subplot(1,4,1)
    plt.plot(t_eval, energy_true, c=COLOR_TRUTH, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, energy_elm, c=COLOR_ELM, zorder=zorders[1], linewidth=linewidths[1])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{H}(q,p)$')
    plt.legend(['Ground truth', 'ELM'], shadow=True)
    plt.subplot(1,4,2)
    plt.plot(t_eval, energy_true, c=COLOR_TRUTH, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, energy_uswim, c=COLOR_USWIM, zorder=zorders[2], linewidth=linewidths[2])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{H}(q,p)$')
    plt.legend(['Ground truth', 'U-SWIM'], shadow=True)
    plt.subplot(1,4,3)
    plt.plot(t_eval, energy_true, c=COLOR_TRUTH, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, energy_aswim, c=COLOR_ASWIM, zorder=zorders[3], linewidth=linewidths[3])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{H}(q,p)$')
    plt.legend(['Ground truth', 'A-SWIM'], shadow=True)
    plt.subplot(1,4,4)
    plt.plot(t_eval, energy_true, c=COLOR_TRUTH, zorder=zorders[0], linewidth=linewidths[0])
    plt.plot(t_eval, energy_swim, c=COLOR_SWIM, zorder=zorders[4], linewidth=linewidths[4])
    plt.xlim(t_span)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{H}(q,p)$')
    plt.legend(['Ground truth', 'SWIM'], shadow=True)
    plt.tight_layout()
    plt.show() if verbose else None
    fig.savefig(save) if save else None
