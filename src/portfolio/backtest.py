from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio.utils import (
    load_prices,
    load_benchmark,
    build_portfolio_and_benchmark_returns_static,
)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    name: str
    port_ret: pd.Series
    bench_ret: pd.Series
    cum_port: pd.Series
    cum_bench: pd.Series
    sharpe_port: float
    sharpe_bench: float
    std_port: float
    std_bench: float
    max_dd_port: float
    max_dd_bench: float

    @property
    def total_return_port(self) -> float:
        return float(self.cum_port.iloc[-1] - 1.0)

    @property
    def total_return_bench(self) -> float:
        return float(self.cum_bench.iloc[-1] - 1.0)

    @property
    def annualized_return_port(self) -> float:
        n = int(self.port_ret.dropna().shape[0])
        if n <= 0:
            return float("nan")
        return float(self.cum_port.iloc[-1] ** (TRADING_DAYS / n) - 1.0)

    @property
    def annualized_return_bench(self) -> float:
        n = int(self.bench_ret.dropna().shape[0])
        if n <= 0:
            return float("nan")
        return float(self.cum_bench.iloc[-1] ** (TRADING_DAYS / n) - 1.0)


@dataclass
class PortfolioBacktester:
    prices_path: str
    benchmark_path: str
    start_year: int = 2023

    def compute(
        self, weights_df: pd.DataFrame, name: str = "Strategy"
    ) -> BacktestResult:
        prices = load_prices(self.prices_path)
        benchmark = load_benchmark(self.benchmark_path, col_name=None)

        port_ret, bench_ret = build_portfolio_and_benchmark_returns_static(
            prices=prices,
            weights_df=weights_df,
            benchmark_series=benchmark,
            start_year=self.start_year,
        )

        sharpe_port = (
            (port_ret.mean() / port_ret.std() * np.sqrt(252))
            if port_ret.std() > 0
            else np.nan
        )
        sharpe_bench = (
            (bench_ret.mean() / bench_ret.std() * np.sqrt(252))
            if bench_ret.std() > 0
            else np.nan
        )

        cum_p = (1 + port_ret).cumprod()
        max_dd_port = (cum_p / cum_p.cummax() - 1).min()

        cum_b = (1 + bench_ret).cumprod()
        max_dd_bench = (cum_b / cum_b.cummax() - 1).min()

        ann_factor = np.sqrt(252)

        std_port = port_ret.std() * ann_factor
        std_bench = bench_ret.std() * ann_factor

        return BacktestResult(
            name=name,
            port_ret=port_ret,
            bench_ret=bench_ret,
            cum_port=cum_p,
            cum_bench=cum_b,
            sharpe_port=sharpe_port,
            sharpe_bench=sharpe_bench,
            std_port=std_port,
            std_bench=std_bench,
            max_dd_port=max_dd_port,
            max_dd_bench=max_dd_bench,
        )
