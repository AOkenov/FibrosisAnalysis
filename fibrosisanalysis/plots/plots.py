"""Summary
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def point_density(x, y):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    y : TYPE
        Description

    Returns
    -------
    tuple
        Description
    """
    xy = np.vstack([x, y])
    density = stats.gaussian_kernel(xy)(xy)
    inds = density.argsort()
    return inds, density


def polar_scattered_plot(ax, r, theta, **kwargs):
    """Summary

    Parameters
    ----------
    ax : TYPE
        Description
    r : TYPE
        Description
    theta : TYPE
        Description
    **kwargs
        Description

    Returns
    -------
    TYPE
        Description
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    inds, density = point_density(x, y)
    r, theta, density = r[inds], theta[inds], density[inds]
    ax.scatter(theta, r, c=density, **kwargs)
    return ax


def linear_scattered_plot(ax, x, y, **kwargs):
    """Summary

    Parameters
    ----------
    ax : TYPE
        Description
    x : TYPE
        Description
    y : TYPE
        Description
    **kwargs
        Description

    Returns
    -------
    TYPE
        Description
    """
    inds, density = point_density(x, y)
    x, y, density = x[inds], y[inds], density[inds]
    ax.scatter(x, y, c=density, **kwargs)
    return ax


def eigsorted(cov):
    '''
    Eigenvalues and eigenvectors of the covariance matrix.
    '''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(points, cov, nstd):
    """
    Source: http://stackoverflow.com/a/12321306/1391441
    """

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    return width, height, theta


def confidence_ellipse(ax, r, theta, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if r.size != theta.size:
        raise ValueError("x and y must be the same size")

    y = r * np.sin(theta)
    x = r * np.cos(theta)

    linear_scattered_plot(ax, x, y)

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    print(pearson)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    print(mean_x, mean_y, scale_x, scale_y)

    transf = transforms.Affine2D()
    transf = transf.rotate_deg(45)
    transf = transf.scale(scale_x, scale_y)
    transf = transf.translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
