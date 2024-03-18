"""Utilities for drawing ellipses

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import warnings
import numpy as np
import pocket.advis
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse

def visualise_one_distribution(x, y, cov, ax=None, n_std=3.0, facecolor='none', **kwargs):
    """Draw an ellipse, such that its centre and axes
    representing the mean and covariance of a distribution.

    Parameters:
    -----------
    x: float
        x coordinate of the ellipse centre
    y: float
        y coordinate of the ellipse centre
    cov: np.ndarray
        A symmetric covariance matrix of size (2, 2)
    ax: AxesSubplot, default: None
        An axis to plot the distribution with. If left as None,
        a new figure and axis will be created.
    n_std: float, default: 3.0
        The number of standard deviations to determine the size of the ellipse.
        The default is 3.0.
    facecolor: str, default: 'none'
        Colour to fill the ellipse. The default is changed to 'none' for a hollow
        ellipse.
    kwargs: dict
        Keyworded arguments for matplotlib.patches.Ellipse module.

    Returns:
    --------
    ax: AxesSubplot
        The axis that the distribution was plotted with.
    """
    if not np.isclose(cov[0, 1], cov[1, 0]):
        warnings.warn(f"WARNING: The given covariance matrix is not symmetric: {cov}.")

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def visualise_distributions(mu, sigma, n_std=3.0, labels=None, palette=None, **kwargs):
    """Visualise multiple distributions in the same figure,
    with each drawn as an ellipse. The mean is illustrated
    as the centre of the ellipse while the covariance is
    depicted as the radiuses.

    Parameters:
    -----------
    mu: np.ndarray or Iterable
        The means of given distributions, of size (N, 2)
    sigma: np.ndarray or Iterable
        The covariance matrices of given distributions, of size (N, 2, 2)
    n_std: float, default: 3.0
        The number of standard deviations to determine the size of the ellipse.
        The default is 3.0.
    labels: List[str], default: None
        Text label for each distrubtion. If left as None, natural numbers will be used.
    palette: List[List], default: None
        A list of colours to plot the ellipses with. Its length should be N.
        If left as None, a default colour palette will be used.
    kwargs: dict
        Keyworded arguments for matplotlib.patches.Ellipse module.

    Returns:
    --------
    ax: AxesSubplot
        The axis that the distribution was plotted with.
    """

    _, ax = plt.subplots()
    if palette is None:
        palette = pocket.advis.palette(n=len(mu))
    if labels is None:
        labels = [str(i + 1) for i in range(len(mu))]

    for i, (m, s, c, l) in enumerate(zip(mu, sigma, palette, labels)):
        visualise_one_distribution(
            *m, s, 
            ax=ax, n_std=n_std,
            label=f"{i + 1}. {l}", edgecolor=c, **kwargs
        )
        ax.text(*m, str(i + 1), color=c, ha='center', va='center', weight='bold')

    ax.relim()
    ax.autoscale()
    ax.legend()
    return ax
