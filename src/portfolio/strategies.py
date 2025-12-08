from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

from measures.risk_estimators import RiskEstimator
from portfolio.mean_variance import min_risk


@dataclass
class RollingMVStrategy:
    """
    Generic rolling-window mean-variance strategy.
    Uses a RiskEstimator to compute the risk matrix.
    """

    risk_estimator: RiskEstimator
    start_year: int = 2023
    lookback_years: int = 1
    target_return: Optional[float] = 0.0004
    min_obs: int = 150  # minimum window size

    def compute_weights(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        df_ret = df_ret.sort_index()
        df_after_start = df_ret.loc[f"{self.start_year}-01-01":]

        # monthly rebalancing dates
        rebal_dates = df_after_start.resample("MS").first().index
        tickers = df_ret.columns

        weights_list = []
        index_list = []

        for d in rebal_dates:
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
