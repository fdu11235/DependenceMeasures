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
    Implements a rolling-window mean–variance portfolio strategy with
    a modular dependence (risk) estimator.

    The strategy proceeds as follows:
      1. A user-specified dependence estimator (e.g. covariance,
         mutual information matrix, optimal transport matrix) is applied
         to each rolling lookback window of returns.
      2. At each rebalancing date (monthly by default), a one-year
         lookback window is constructed ending the day before the
         rebalance date.
      3. If the window contains at least `min_obs` observations, the
         dependence matrix Σ and expected returns μ are estimated.
      4. A mean–variance optimization problem is solved via `min_risk`
         to obtain portfolio weights, with an optional target return and
         no short selling.
      5. The resulting weights are stored at monthly frequency.

    Parameters
    ----------
    risk_estimator : DependenceMatrixEstimator
        Object providing an `.estimate(df_returns)` method that returns
        a dependence (risk) matrix as a DataFrame.
    start_year : int, default 2023
        First calendar year considered for rebalancing.
    lookback_years : int, default 1
        Length of the rolling estimation window.
    target_return : float or None, default 0.0004
        Target mean return in the mean–variance optimization. If None,
        computes a minimum-variance portfolio.
    min_obs : int, default 150
        Minimum number of observations required in a lookback window.

    Returns
    -------
    pd.DataFrame
        DataFrame of optimized portfolio weights indexed by rebalancing
        dates and with one column per asset.
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
    Implements a rolling-window Equal Risk Contribution (ERC) portfolio
    strategy using a modular dependence (risk) estimator.

    The procedure is analogous to the rolling mean–variance approach,
    but replaces the optimization step with an ERC allocation, where
    each asset is constrained to contribute equally to total portfolio risk.

    Workflow:
      1. A dependence estimator (e.g., covariance, entropy–MI matrix,
         OT-based dependence matrix) is applied to each rolling lookback
         window of returns.
      2. Rebalancing occurs at monthly frequency, using a one-year
         lookback window ending the day before the rebalancing date.
      3. If the window contains at least `min_obs` observations, the
         dependence matrix Σ is estimated.
      4. Portfolio weights are computed via `equal_risk_contribution`,
         enforcing non-negativity (no short selling) and equal marginal
         risk contributions across assets.
      5. The resulting weight vector is stored at monthly frequency.

    Parameters
    ----------
    risk_estimator : DependenceMatrixEstimator
        Object providing an `.estimate(df_returns)` method that returns a
        dependence (risk) matrix as a DataFrame.
    start_year : int, default 2023
        First calendar year considered for rebalancing.
    lookback_years : int, default 1
        Length of the rolling estimation window.
    min_obs : int, default 150
        Minimum number of observations required in the lookback window
        before computing weights.

    Returns
    -------
    pd.DataFrame
        DataFrame of ERC portfolio weights, indexed by rebalancing dates
        and with one column per asset.
    """

    risk_estimator: DependenceMatrixEstimator
    start_year: int = 2023
    lookback_years: int = 1
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
