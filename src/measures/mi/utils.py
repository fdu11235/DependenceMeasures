import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy


def digitize_returns(
    df_ret: pd.DataFrame, min_ret: float = -0.5, max_ret: float = 0.5, n_bins: int = 101
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


def mi(X: np.ndarray, Y: np.ndarray, n_states: int) -> float:
    """
    Mutual information of two *discrete* variables X, Y
    whose values are in {0, ..., n_states-1}.
    """
    X = X.astype(int)
    Y = Y.astype(int)

    mask = (~np.isnan(X)) & (~np.isnan(Y))
    X = X[mask]
    Y = Y[mask]

    # joint counts
    joint = np.zeros((n_states, n_states), dtype=float)
    for x, y in zip(X, Y):
        if 0 <= x < n_states and 0 <= y < n_states:
            joint[x, y] += 1

    joint_sum = joint.sum()
    if joint_sum == 0:
        return 0.0
    joint /= joint_sum

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    # avoid log(0)
    mask = (joint > 0) & (px > 0) & (py > 0)
    mi_val = np.sum(joint[mask] * np.log2(joint[mask] / (px * py)[mask]))
    return float(mi_val)
