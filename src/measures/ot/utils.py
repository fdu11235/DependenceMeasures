import numpy as np
from scipy.stats import rankdata


def empirical_copula_transform(x: np.ndarray) -> np.ndarray:
    """
    Probability integral transform via normalized ranks.

    Parameters
    ----------
    x : array-like, shape (T,)
        Samples of one variable.

    Returns
    -------
    u : ndarray, shape (T,)
        Values in (0, 1), approximately uniform.
    """
    x = np.asarray(x)
    ranks = rankdata(x, method="average")
    # divide by T+1 instead of T to avoid exact 0/1
    return ranks / (len(x) + 1.0)


def empirical_copula_hist(
    u: np.ndarray,
    v: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """
    Build a 2D histogram on [0,1]^2 approximating the copula density.

    Parameters
    ----------
    u, v : array-like, shape (T,)
        Values in [0,1].
    n_bins : int
        Number of bins along each axis.

    Returns
    -------
    H : ndarray, shape (n_bins, n_bins)
        Normalized histogram that sums to 1.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    assert u.shape == v.shape

    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    H, _, _ = np.histogram2d(
        u,
        v,
        bins=n_bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    H = H.astype(np.float64)
    eps = 1e-8
    H += eps
    total = H.sum()
    if total <= 0:
        # fallback: uniform copula
        H[:] = 1.0 / H.size
    else:
        H /= total
    return H


def grid_cost_matrix(n_bins: int, p: int = 2) -> np.ndarray:
    """
    Ground cost matrix on a regular n_bins x n_bins grid on [0,1]^2.

    Parameters
    ----------
    n_bins : int
        Number of bins along each axis.
    p : int
        Order of the underlying l_p ground metric.

    Returns
    -------
    M : ndarray, shape (n_bins**2, n_bins**2)
        Pairwise distances between bin centers.
    """
    xs = (np.arange(n_bins) + 0.5) / n_bins
    xv, yv = np.meshgrid(xs, xs)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (n_bins**2, 2)
    diff = coords[:, None, :] - coords[None, :, :]

    if p == 1:
        M = np.abs(diff).sum(axis=2)
    elif p == 2:
        M = np.sqrt((diff**2).sum(axis=2))
    else:
        M = (np.abs(diff) ** p).sum(axis=2) ** (1.0 / p)
    return M
