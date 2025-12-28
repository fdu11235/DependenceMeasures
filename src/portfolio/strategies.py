from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

from measures.matrix_estimators import DependenceMatrixEstimator
from portfolio.mean_variance import min_risk, min_risk_cvxopt
from portfolio.equal_risk_contribution import (
    equal_risk_contribution,
    risk_contributions,
)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s) < 1e-12:
        return np.ones_like(w) / len(w)
    return w / s


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
            # print(mu)
            # self.target_return = mu.mean()

            w = min_risk_cvxopt(
                mu,
                Sigma,
                target_return=self.target_return,
            )
            weights_list.append(w)
            index_list.append(d)

        weights = pd.DataFrame(
            np.vstack(weights_list),
            index=index_list,
            columns=tickers,
        )
        return weights


@dataclass
class RollingEqualCorrelationStrategy:
    """
    Implements a rolling-window Equal-Correlation ("Optimally Diversified") strategy
    with a modular dependence (risk) estimator.

    The strategy proceeds as follows:
      1. A user-specified dependence estimator (e.g. covariance,
         mutual information matrix, optimal transport matrix) is applied
         to each rolling lookback window of returns.
      2. At each rebalancing date (monthly by default), a one-year
         lookback window is constructed ending the day before the
         rebalance date.
      3. If the window contains at least `min_obs` observations, the
         dependence (risk) matrix Σ is estimated.
      4. Compute σ = sqrt(diag(Σ)) and set weights as:
            w ∝ Σ^{-1} σ
         then normalize to sum to 1. If long_only=True, negative weights
         are clipped to 0 and renormalized.
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
    min_obs : int, default 150
        Minimum number of observations required in a lookback window.
    long_only : bool, default True
        If True, project weights to long-only by clipping negatives to 0
        and renormalizing. If False, allow negative weights (still normalized).
    ridge : float, default 1e-6
        Small diagonal regularization added to Σ for numerical stability.
    use_pinv : bool, default True
        If True, use pseudo-inverse instead of inverse (more stable).

    Returns
    -------
    pd.DataFrame
        DataFrame of portfolio weights indexed by rebalancing dates and
        with one column per asset.
    """

    risk_estimator: DependenceMatrixEstimator
    start_year: int = 2023
    lookback_years: int = 1
    min_obs: int = 150  # minimum window size
    long_only: bool = True
    ridge: float = 1e-6
    use_pinv: bool = True

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

            # Ensure consistent ordering
            Sigma_df = Sigma_df.reindex(index=tickers, columns=tickers)
            Sigma = Sigma_df.to_numpy(dtype=float)

            n = Sigma.shape[0]

            # Regularize (helps for near-singular / noisy matrices)
            if self.ridge and self.ridge > 0:
                Sigma = Sigma + float(self.ridge) * np.eye(n)

            # σ = sqrt(diag(Σ)) (clip diag in case estimator yields small negatives)
            diag = np.diag(Sigma)
            diag = np.maximum(diag, 0.0)
            sigma = np.sqrt(diag)

            # Fallback if sigma is degenerate
            if not np.isfinite(sigma).all() or float(np.sum(sigma)) <= 1e-12:
                w = np.ones(n) / n
            else:
                invSigma = (
                    np.linalg.pinv(Sigma) if self.use_pinv else np.linalg.inv(Sigma)
                )
                w = invSigma @ sigma
                w = np.asarray(w, dtype=float)

                if self.long_only:
                    w = np.maximum(w, 0.0)

                w = _normalize_weights(w)

            weights_list.append(w)
            index_list.append(d)

        if not weights_list:
            return pd.DataFrame(columns=tickers)

        weights = pd.DataFrame(
            np.vstack(weights_list),
            index=index_list,
            columns=tickers,
        )
        return weights


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
            print(Sigma_df)
            Sigma = Sigma_df.to_numpy()

            w = equal_risk_contribution(
                Sigma,
                short_selling=False,
            )
            rc = risk_contributions(w, Sigma)

            print("RC std / mean:", rc.std() / rc.mean())
            print("Weights std:", w.std())

            weights_list.append(w)
            index_list.append(d)

        return pd.DataFrame(
            np.vstack(weights_list),
            index=index_list,
            columns=tickers,
        )


@dataclass
class RollingMarketCapStrategy:
    """
    Rolling market-cap weighted strategy.

    At each monthly rebalancing date d, weights are set proportional to
    market caps observed at the most recent date <= d-1 (to avoid look-ahead).

    Parameters
    ----------
    start_year : int, default 2023
        First calendar year considered for rebalancing.
    min_obs : int, default 1
        Minimum number of return observations required before producing a weight
        (kept for consistency with other strategies).
    """

    start_year: int = 2023
    min_obs: int = 1

    def compute_weights(
        self, df_ret: pd.DataFrame, df_mcap: pd.DataFrame
    ) -> pd.DataFrame:
        print("Computing market-cap weights...")
        df_ret = df_ret.sort_index()
        df_mcap = df_mcap.sort_index()

        df_after_start = df_ret.loc[f"{self.start_year}-01-01":]
        rebal_dates = df_after_start.resample("MS").first().index
        tickers = df_ret.columns

        weights_list = []
        index_list = []

        for d in tqdm(rebal_dates):
            # keep the same pattern as your other strategies
            window_end = d - pd.Timedelta(days=1)
            window_ret = df_ret.loc[:window_end]

            if len(window_ret) < self.min_obs:
                continue

            # use the latest market cap available up to window_end
            mcap_hist = df_mcap.loc[:window_end]

            if mcap_hist.empty:
                continue

            mcap = mcap_hist.iloc[-1]

            # align to tickers and drop missing / non-positive values
            mcap = mcap.reindex(tickers)
            mcap = mcap.replace([np.inf, -np.inf], np.nan)
            mcap = mcap.dropna()
            mcap = mcap[mcap > 0]

            if mcap.empty or float(mcap.sum()) <= 0.0:
                continue

            w = (mcap / mcap.sum()).reindex(tickers).fillna(0.0).to_numpy()

            weights_list.append(w)
            index_list.append(d)

        return pd.DataFrame(
            np.vstack(weights_list),
            index=index_list,
            columns=tickers,
        )


@dataclass
class RollingEqualWeightStrategy:
    """
    Rolling equal-weight strategy.

    At each monthly rebalancing date d, assigns equal weights across all tickers.
    """

    start_year: int = 2023

    def compute_weights(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        print("Computing equal weights...")
        df_ret = df_ret.sort_index()
        df_after_start = df_ret.loc[f"{self.start_year}-01-01":]

        rebal_dates = df_after_start.resample("MS").first().index
        tickers = df_ret.columns

        w = np.ones(len(tickers)) / len(tickers)

        weights = pd.DataFrame(
            np.tile(w, (len(rebal_dates), 1)),
            index=rebal_dates,
            columns=tickers,
        )
        return weights
