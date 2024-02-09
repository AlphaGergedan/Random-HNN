import matplotlib.pyplot as plt


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
            xlabel='x', ylabel='y', aspect=2.5, title=None, verbose=True, save=''):
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
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(grid, extent=extent, vmin=vmin, vmax=vmax)
    if contourlines > 0:
        ax.contour(xrange, yrange, grid, contourlines, colors='white', linewidths=0.5)
    fig.colorbar(im, ax=ax, fraction=0.049*(yrange.shape[0] / xrange.shape[0]), pad=0.04)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_aspect(aspect)
    if title is not None:
        ax.set_title(title)
    if save != '':
        fig.savefig(save)
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
