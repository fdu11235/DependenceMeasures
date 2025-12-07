from typing import Protocol, runtime_checkable
import pandas as pd
import numpy as np

from measures.mi.entropy_mi import entropy_mi_matrix
from measures.ot.copulas import build_standard_correlation_targets
from measures.ot.ot import tfdc_matrix


@runtime_checkable
class RiskEstimator(Protocol):
    """Any estimator that converts return windows into a risk matrix."""

    def estimate(self, window_ret: pd.DataFrame) -> pd.DataFrame: ...


class CovarianceEstimator:
    """Standard sample covariance matrix."""

    def estimate(self, window_ret: pd.DataFrame) -> pd.DataFrame:
        return window_ret.cov()


class EntropyMIEstimator:
    """MI - Entropy matrix."""

    def __init__(self, min_ret=-0.5, max_ret=0.5, n_bins=101):
        self.min_ret = min_ret
        self.max_ret = max_ret
        self.n_bins = n_bins

    def estimate(self, window_ret: pd.DataFrame) -> pd.DataFrame:
        return entropy_mi_matrix(
            window_ret,
            min_ret=self.min_ret,
            max_ret=self.max_ret,
            n_bins=self.n_bins,
        )


class TFDC_Estimator:
    """TFDC matrix"""

    def __init__(self, n_bins=20, reg=5e-3, p=2, method="sinkhorn"):
        self.n_bins = n_bins
        self.reg = reg
        self.p = p
        self.method = method

        # build target/forget copulas only once
        self.target_copulas, self.forget_copulas = build_standard_correlation_targets(
            n_bins=n_bins,
            n_samples=50_000,
            random_state=42,
        )

    def estimate(self, window_ret: pd.DataFrame) -> pd.DataFrame:
        return tfdc_matrix(
            window_ret,
            target_copulas=self.target_copulas,
            forget_copulas=self.forget_copulas,
            n_bins=self.n_bins,
            reg=self.reg,
            p=self.p,
            method=self.method,
        )
