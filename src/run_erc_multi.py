import pandas as pd
import numpy as np
from portfolio.utils import load_prices, compute_returns
from measures.matrix_estimators import (
    EntropyMIEstimator,
    TFDC_Estimator,
    CovarianceEstimator,
)
from portfolio.strategies import RollingERCStrategy
from portfolio.backtest import PortfolioBacktester
import matplotlib.pyplot as plt


def plot_cumulative(results):
    with plt.style.context("seaborn-v0_8-darkgrid"):
        plt.figure(figsize=(12, 6))

        for r in results:
            r.cum_port.plot(label=r.name)

        if results:
            results[0].cum_bench.plot(
                label="SMI Benchmark",
                linewidth=2,
            )

        plt.title("Cumulative Returns")
        plt.ylabel("Growth")
        plt.legend()
        plt.tight_layout()
        plt.show()


def build_summary_table(results: list) -> pd.DataFrame:
    """
    Build a comparison table for strategies + one benchmark row.

    Expects each result to expose:
      - name
      - total_return_port, annualized_return_port, sharpe_port, max_dd_port
      - total_return_bench, annualized_return_bench, sharpe_bench, max_dd_bench
    """
    if not results:
        return pd.DataFrame(
            columns=["Total Return", "Ann. Return", "Sharpe", "Max Drawdown"]
        )

    rows = []

    # strategies
    for r in results:
        rows.append(
            {
                "Name": r.name,
                "Total Return": r.total_return_port,
                "Ann. Return": r.annualized_return_port,
                "Sharpe": r.sharpe_port,
                "Max Drawdown": r.max_dd_port,
            }
        )

    # benchmark once
    b = results[0]
    rows.append(
        {
            "Name": "SMI Benchmark",
            "Total Return": b.total_return_bench,
            "Ann. Return": b.annualized_return_bench,
            "Sharpe": b.sharpe_bench,
            "Max Drawdown": b.max_dd_bench,
        }
    )

    return pd.DataFrame(rows).set_index("Name")


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    out = summary.copy()
    out["Total Return"] = out["Total Return"].map(
        lambda x: f"{x:.2%}" if np.isfinite(x) else "nan"
    )
    out["Ann. Return"] = out["Ann. Return"].map(
        lambda x: f"{x:.2%}" if np.isfinite(x) else "nan"
    )
    out["Max Drawdown"] = out["Max Drawdown"].map(
        lambda x: f"{x:.2%}" if np.isfinite(x) else "nan"
    )
    out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "nan")
    return out


def main():
    prices = load_prices("data/smi+smim_prices_cleaned.xlsx")
    df_ret = compute_returns(prices)

    bt = PortfolioBacktester(
        prices_path="data/smi+smim_prices_cleaned.xlsx",
        smi_path="data/smi.xlsx",
        start_year=2023,
    )

    strategies = [
        (
            "MV + Entropy MI",
            RollingERCStrategy(
                EntropyMIEstimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
        (
            "MV + TFDC (OT)",
            RollingERCStrategy(
                TFDC_Estimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
        (
            "MV + Pearson",
            RollingERCStrategy(
                CovarianceEstimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
    ]

    results = []
    for name, strat in strategies:
        w = strat.compute_weights(df_ret)
        results.append(bt.compute(w, name=name))

    plot_cumulative(results)

    summary = build_summary_table(results)
    print(format_summary_table(summary))


if __name__ == "__main__":
    main()
