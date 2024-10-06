import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def __eigsorted(cov):
    """Utility function for calculating the eigen-spectrum
    of a covariance matrix, and sorting the eigenvectors
    and eigenvalues by the eigenvalues."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def visualize_gaussian(mean, cov, nstd=2, ax=None, **kwargs):
    """Utility function to plot the covariance of a Gaussian
    measure. Adds ellipses corresponding to nstd around the mean."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    vals, vecs = __eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mean,
                    width=width,
                    height=height,
                    angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ax


def visualize_gmm(means, covs, ax=None,
                  kwargs_scatter={}, kwargs_ellipse={}):
    """Utility function to plot a GMM. Scatters the means,
    and adds an ellipse corresponding to the covariances of
    each component."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(means[:, 0], means[:, 1], **kwargs_scatter)
    for mean_c, cov_c in zip(means, covs):
        visualize_gaussian(mean_c, cov_c, ax=ax, **kwargs_ellipse)
    return ax
