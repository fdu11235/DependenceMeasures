import pandas as pd
from .utils import corr_to_cov

def spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman rank correlation matrix.
    """
    return df.corr(method="spearman")


def spearman_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman-based covariance estimator.
    """
    corr = spearman_corr(df)
    vol = df.std()
    return corr_to_cov(corr, vol)
