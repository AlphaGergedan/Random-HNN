import matplotlib.pyplot as plt

def draw_1d(x, y, label='f(x)',
            xpredict=None, ypredict=None, labelpredict='Approximation',
            xlim=None, ylim=None,
            xlabel='x', ylabel='y', title=None, verbose=True, save=''):
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
        print()
    else:
        plt.close()

def draw_2d(grid,
            xrange, yrange,
            vmin=None, vmax=None,
            extent=None, contourlines=10,
            xlabel='x', ylabel='y',
            aspect=2.5, title=None, verbose=True, save=''):
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
        print()
    else:
        plt.close()

def draw_4d(grid, fixed_dimensions={ "q1": 0, "q2": 0 }):
    """
    # xrange, yrange,
            # vmin=None, vmax=None,
            # extent=None, showcontour=False,
            # xlabel='x', ylabel='y',
            # aspect=2.5, title=None, save='',

    This plot function is for dynamical systems with 2 degrees of freedom
    resulting a 4 dimensional system with q1,q2,p1,p2.

    fixed_dimensions: specifies what to fix, must include two entries only for a 2d plot

    Uses draw_2d but with selected features specified in the grid range
    """
    assert len(grid.shape) == 4
    assert len(fixed_dimensions.keys()) == 2
    for (k,v) in fixed_dimensions.items():
        print(k)
    pass

    # # poincare grid
    # plot_grid = grid[]
    # plot_data_1 = y_grid[:, N_q1//2, N_p1//2, :].reshape((N_p2, N_q2))
    # draw_2d(plot_data_1, q2_range, p2_range, extent=[q2_lim[0],q2_lim[1],p2_lim[0],p2_lim[1]], showcontour=True, xlabel='q2', ylabel='p2', aspect=1.5, title="H(x)" + "\n" + "with q1,p1 near 0")


def draw_6d():
    pass
