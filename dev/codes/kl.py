# https://stackoverflow.com/questions/61709954/how-do-i-find-the-kl-divergence-of-samples-from-two-2d-distributions

# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

# A Nearest-Neighbor Approach to Estimating Divergence between
# Continuous Random Vectors
# https://ieeexplore.ieee.org/document/4035959

import numpy as np


def KLdivergence(x: np.ndarray, y: np.ndarray) -> np.float64:
    """Compute the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.

    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.

    Returns
    -------
    out : np.float64
      The estimated Kullback-Leibler divergence D(P||Q).

    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent - isue is here!!
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert d == dy

    # Build a KD tree representation of the samples and find the nearest
    # neighbour of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=0.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=0.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a
    # negative sign on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))
