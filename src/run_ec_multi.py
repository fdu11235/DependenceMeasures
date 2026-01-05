import pandas as pd
import numpy as np
from portfolio.utils import (
    load_prices,
    compute_returns,
    plot_cumulative,
    build_summary_table,
    format_summary_table,
)
from measures.matrix_estimators import (
    EntropyMIEstimator,
    TFDCEstimator,
    CovarianceEstimator,
    SpearmanEstimator,
)
from portfolio.strategies import (
    RollingEqualCorrelationStrategy,
    RollingEqualWeightStrategy,
)
from portfolio.backtest import PortfolioBacktester


def main():
    prices = load_prices("data/smi+smim_prices_cleaned.xlsx")
    df_ret = compute_returns(prices)

    bt = PortfolioBacktester(
        prices_path="data/smi+smim_prices_cleaned.xlsx",
        benchmark_path="data/smi_expanded.xlsx",
        start_year=2023,
    )

    strategies = [
        (
            "Mutual Information",
            RollingEqualCorrelationStrategy(
                EntropyMIEstimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
        (
            "TFDC",
            RollingEqualCorrelationStrategy(
                TFDCEstimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
        (
            "Pearson",
            RollingEqualCorrelationStrategy(
                CovarianceEstimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
        (
            "Spearman",
            RollingEqualCorrelationStrategy(
                SpearmanEstimator(),
                start_year=2023,
                lookback_years=1,
            ),
        ),
        (
            "Equal Weighted",
            RollingEqualWeightStrategy(start_year=2023),
        ),
    ]

    results = []
    for name, strat in strategies:
        w = strat.compute_weights(df_ret)
        fname = name.lower().replace(" ", "_")
        w.to_csv(f"output/ec/weights_{fname}.csv", index_label="date")
        results.append(bt.compute(w, name=name))

    plot_cumulative(
        results,
        benchmark_label="SMI Expanded",
        save_path="output/EC_smi_expanded_results.pdf",
    )

    summary = build_summary_table(results, "SMI Expanded")
    summary_fmt = format_summary_table(summary)
    print(summary_fmt)
    summary_fmt.to_excel("output/EC_smi_expanded_summary_table.xlsx")


if __name__ == "__main__":
    main()
