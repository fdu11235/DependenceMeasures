import pandas as pd
import numpy as np

def corr_to_cov(corr: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """
    Build a covariance matrix from a correlation matrix:

        Σ_ij = ρ_ij * σ_i * σ_j

    Parameters
    ----------
    corr : DataFrame (N x N)
        Correlation matrix.
    vol : Series (N,)
        Standard deviations (σ_i).

    Returns
    -------
    cov_df : DataFrame (N x N)
    """
    vol = vol.reindex(corr.index)
    D = np.diag(vol.values)
    Sigma = D @ corr.to_numpy() @ D
    return pd.DataFrame(Sigma, index=corr.index, columns=corr.columns)
