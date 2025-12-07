import numpy as np
import pandas as pd
from .utils import digitize_returns, entropy, mi


def entropy_mi_matrix(
    df_ret: pd.DataFrame, min_ret: float = -0.5, max_ret: float = 0.5, n_bins: int = 101
) -> pd.DataFrame:
    """
    Build the entropy + mutual information risk matrix as in Novais et al. (2022):

        Σ[i,i] = H(X_i)
        Σ[i,j] = MI(X_i, X_j)  for i != j

    where H is Shannon entropy (base 2)
    and MI is mutual information using histogram-based discretization.

    Parameters
    ----------
    df_ret : DataFrame (T x N)
        Return matrix.
    min_ret : float
        Minimum bin range for discretization.
    max_ret : float
        Maximum bin range.
    n_bins : int
        Number of bins (paper uses 101).

    Returns
    -------
    Sigma_df : DataFrame (N x N)
        Symmetric entropy + MI matrix.
    """
    cols = df_ret.columns
    n_assets = len(cols)

    # Discretize continuous returns
    digitized, bins = digitize_returns(
        df_ret, min_ret=min_ret, max_ret=max_ret, n_bins=n_bins
    )

    n_states = n_bins - 1
    Sigma = np.zeros((n_assets, n_assets), dtype=float)

    # Diagonal = Entropy
    for i in range(n_assets):
        Xi = digitized[:, i]
        Sigma[i, i] = entropy(Xi, n_states)

    # Off-diagonals = Mutual Information
    for i in range(n_assets):
        Xi = digitized[:, i]
        for j in range(i + 1, n_assets):
            Xj = digitized[:, j]
            m = mi(Xi, Xj, n_states)
            Sigma[i, j] = m
            Sigma[j, i] = m

    return pd.DataFrame(Sigma, index=cols, columns=cols)
