import pandas as pd
from .utils import corr_to_cov

def pearson_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix.
    """
    return df.corr(method="pearson")


def pearson_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation-based covariance estimator:
        Σ_ij = ρ_ij * σ_i * σ_j
    """
    corr = pearson_corr(df)
    vol = df.std()  # vector σ_i
    return corr_to_cov(corr, vol)
