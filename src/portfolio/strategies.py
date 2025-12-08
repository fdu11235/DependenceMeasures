from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

from measures.matrix_estimators import DependenceMatrixEstimator
from portfolio.mean_variance import min_risk
from portfolio.risk_parity import equal_risk_contribution


@dataclass
class RollingMVStrategy:
    """
    Generic rolling-window mean-variance strategy.
    Uses a dependence measure to compute the dependence matrix.
    """

    risk_estimator: DependenceMatrixEstimator
    start_year: int = 2023
    lookback_years: int = 1
    target_return: Optional[float] = 0.0004
    min_obs: int = 150  # minimum window size

    def compute_weights(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        print("Computing weights...")
        df_ret = df_ret.sort_index()
        df_after_start = df_ret.loc[f"{self.start_year}-01-01":]

        # monthly rebalancing dates
        rebal_dates = df_after_start.resample("MS").first().index
        tickers = df_ret.columns

        weights_list = []
        index_list = []

        for d in tqdm(rebal_dates):
            window_start = d - pd.DateOffset(years=self.lookback_years)
            window_end = d - pd.Timedelta(days=1)
            window_ret = df_ret.loc[window_start:window_end]

            if len(window_ret) < self.min_obs:
                continue

            Sigma_df = self.risk_estimator.estimate(window_ret)
            Sigma = Sigma_df.to_numpy()
            mu = window_ret.mean().to_numpy()

            w = min_risk(
                mu,
                Sigma,
                target_return=self.target_return,
                short_selling=False,
            )

            weights_list.append(w)
            index_list.append(d)

        return pd.DataFrame(
            np.vstack(weights_list),
            index=index_list,
            columns=tickers,
        )


@dataclass
class RollingERCStrategy:
    """
    Generic rolling-window mean-variance strategy.
    Uses a dependence measure to compute the dependence matrix.
    """

    risk_estimator: DependenceMatrixEstimator
    start_year: int = 2023
    lookback_years: int = 1
    target_return: Optional[float] = 0.0004
    min_obs: int = 150  # minimum window size

    def compute_weights(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        print("Computing weights...")
        df_ret = df_ret.sort_index()
        df_after_start = df_ret.loc[f"{self.start_year}-01-01":]

        # monthly rebalancing dates
        rebal_dates = df_after_start.resample("MS").first().index
        tickers = df_ret.columns

        weights_list = []
        index_list = []

        for d in tqdm(rebal_dates):
            window_start = d - pd.DateOffset(years=self.lookback_years)
            window_end = d - pd.Timedelta(days=1)
            window_ret = df_ret.loc[window_start:window_end]

            if len(window_ret) < self.min_obs:
                continue

            Sigma_df = self.risk_estimator.estimate(window_ret)
            Sigma = Sigma_df.to_numpy()
            mu = window_ret.mean().to_numpy()

            w = equal_risk_contribution(
                Sigma,
                short_selling=False,
            )

            weights_list.append(w)
            index_list.append(d)

        return pd.DataFrame(
            np.vstack(weights_list),
            index=index_list,
            columns=tickers,
        )
