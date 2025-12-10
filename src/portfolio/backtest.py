from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio.utils import (
    load_prices,
    load_smi_benchmark,
    build_portfolio_and_benchmark_returns,
    build_portfolio_and_benchmark_returns_static,
)


@dataclass
class PortfolioBacktester:
    prices_path: str
    smi_path: str
    start_year: int = 2023

    def run(self, weights_df: pd.DataFrame):
        prices = load_prices(self.prices_path)
        smi = load_smi_benchmark(self.smi_path, col_name=None)

        # compute returns
        port_ret, bench_ret = build_portfolio_and_benchmark_returns_static(
            prices=prices,
            weights_df=weights_df,
            smi_series=smi,
            start_year=self.start_year,
        )

        # stats
        sharpe_port = (
            port_ret.mean() / port_ret.std() * np.sqrt(252)
            if port_ret.std() > 0
            else np.nan
        )
        sharpe_bench = (
            bench_ret.mean() / bench_ret.std() * np.sqrt(252)
            if bench_ret.std() > 0
            else np.nan
        )

        cum_p = (1 + port_ret).cumprod()
        running_max_p = cum_p.cummax()
        max_dd_port = (cum_p / running_max_p - 1).min()

        cum_b = (1 + bench_ret).cumprod()
        running_max_b = cum_b.cummax()
        max_dd_bench = (cum_b / running_max_b - 1).min()

        # plot
        plt.figure(figsize=(12, 6))
        cum_p.plot(label="Strategy")
        cum_b.plot(label="SMI Benchmark")
        plt.title("Cumulative Returns")
        plt.grid(True)
        plt.legend()
        plt.show()

        return {
            "sharpe_port": sharpe_port,
            "max_dd_port": max_dd_port,
            "sharpe_bench": sharpe_bench,
            "max_dd_bench": max_dd_bench,
            "port_ret": port_ret,
            "bench_ret": bench_ret,
        }
