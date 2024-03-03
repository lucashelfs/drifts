# https://stackoverflow.com/questions/61709954/how-do-i-find-the-kl-divergence-of-samples-from-two-2d-distributions

# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

# A Nearest-Neighbor Approach to Estimating Divergence between
# Continuous Random Vectors
# https://ieeexplore.ieee.org/document/4035959

import numpy as np
from scipy.stats import gaussian_kde


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

    # Check the dimensions are consistent - issue is here!!
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
    print("N of KLDivergence:", n)
    print("r of KLDivergence:", r)
    print("s of KLDivergence:", s)
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))


def kl_divergence(p, q) -> np.float64:
    # Ensure p and q are non-negative and not zero
    p = np.maximum(p, np.finfo(float).eps)
    q = np.maximum(q, np.finfo(float).eps)

    return np.sum(p * np.log(p / q))


def calculate_kl_divergence_with_kde(samples_p, samples_q, bigger=False) -> np.float64:
    # Compute KDE (Kernel Density Estimation) for both samples
    kde_p = gaussian_kde(samples_p)
    kde_q = gaussian_kde(samples_q)

    # Create a range for evaluation (considering the union of both sets of samples)
    min_val = min(np.min(samples_p), np.min(samples_q))
    max_val = max(np.max(samples_p), np.max(samples_q))

    # Maybe this could be bigger...
    x_eval = np.linspace(min_val, max_val, 1000)
    if bigger:
        x_eval = np.linspace(min_val, max_val, max(len(samples_p), len(samples_q)))

    # Evaluate the KDEs on the range
    p = kde_p.evaluate(x_eval)
    q = kde_q.evaluate(x_eval)

    # Normalize the densities to ensure they sum up to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute KL divergence
    kl_div = kl_divergence(p, q)

    return kl_div
