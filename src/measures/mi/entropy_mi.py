import numpy as np
import pandas as pd
from .utils import digitize_returns, entropy, mi, normalize_mi


def entropy_mi_matrix(
    df_ret: pd.DataFrame,
    min_ret: float = -0.5,
    max_ret: float = 0.5,
    n_bins: int = 101,
    mi_normalization: str = "sqrt",  # "raw", "min", "sum", "sqrt"
) -> pd.DataFrame:
    """
    Build the entropy + mutual information risk matrix:

        Σ[i,i] = H(X_i)
        Σ[i,j] = MI(X_i, X_j)  for i != j

    where H is entropy
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
    mi_normalization : str
        {"raw", "min", "sum", "sqrt"} normalization method applied to off-diagonals.

    Returns
    -------
    Sigma_df : DataFrame (N x N)
        Symmetric entropy + MI matrix.
    """
    cols = df_ret.columns
    n_assets = len(cols)

    # Discretize continuous returns
    digitized, _ = digitize_returns(
        df_ret, min_ret=min_ret, max_ret=max_ret, n_bins=n_bins
    )

    n_states = n_bins - 1
    Sigma = np.zeros((n_assets, n_assets), dtype=float)

    # Pre-compute entropies once (diagonal + needed for normalization)
    H = np.zeros(n_assets, dtype=float)
    for i in range(n_assets):
        Xi = digitized[:, i]
        H[i] = entropy(Xi, n_states)
        Sigma[i, i] = H[i]

    # Off-diagonals = MI (raw or normalized)
    for i in range(n_assets):
        Xi = digitized[:, i]
        for j in range(i + 1, n_assets):
            Xj = digitized[:, j]
            mi_ij = mi(Xi, Xj, n_states)

            # Apply normalization
            val = normalize_mi(mi_ij, H[i], H[j], method=mi_normalization)

            Sigma[i, j] = val
            Sigma[j, i] = val

    return pd.DataFrame(Sigma, index=cols, columns=cols)
