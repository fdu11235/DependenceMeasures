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


def spearman_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman-based covariance estimator.
    """
    corr = df.corr(method="spearman")
    vol = df.std(ddof=1)
    return corr_to_cov(corr, vol)


def pearson_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation-based covariance estimator:
        Σ_ij = ρ_ij * σ_i * σ_j
    """
    corr = df.corr(method="pearson")
    vol = df.std(ddof=1)
    return corr_to_cov(corr, vol)
