import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy


def digitize_returns(
    df_ret: pd.DataFrame,
    min_ret: float = -0.5,
    max_ret: float = 0.5,
    n_bins: int = 101
):
    """
    Discretize continuous returns into bins (as in the paper: -50% to +50%, 101 bins).

    Returns
    -------
    digitized : np.ndarray (T x N)
        Each entry is a bin index.
    bins : np.ndarray
        Bin edges.
    """
    bins = np.linspace(min_ret, max_ret, n_bins)
    data = df_ret.to_numpy()

    digitized = np.digitize(data, bins) - 1
    digitized = np.clip(digitized, 0, n_bins - 2)

    return digitized, bins


def entropy(col: np.ndarray, n_states: int) -> float:
    """
    Shannon entropy (base 2) of a discrete variable given as integer states.
    """
    counts = np.bincount(col, minlength=n_states)
    p = counts / counts.sum()
    return shannon_entropy(p, base=2)


def mi(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> float:
    """
    Mutual Information I(X;Y) using a 2D histogram.

    Parameters
    ----------
    x, y : integer-discretized variables
    bins : histogram bin edges

    Returns
    -------
    float : mutual information in bits
    """
    joint, _, _ = np.histogram2d(x, y, bins=[bins, bins])

    total = joint.sum()
    if total == 0:
        return 0.0

    joint_prob = joint / total

    px = joint_prob.sum(axis=1, keepdims=True)
    py = joint_prob.sum(axis=0, keepdims=True)

    mask = joint_prob > 0
    px_py = px @ py  # outer product

    return np.sum(joint_prob[mask] * np.log2(joint_prob[mask] / px_py[mask]))