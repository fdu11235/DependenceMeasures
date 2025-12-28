from portfolio.utils import (
    load_prices,
    compute_returns,
    plot_cumulative,
    build_summary_table,
    format_summary_table,
)
from measures.matrix_estimators import (
    EntropyMIEstimator,
    TFDC_Estimator,
    CovarianceEstimator,
    SpearmanEstimator,
)
from portfolio.strategies import (
    RollingMVStrategy,
    RollingEqualWeightStrategy,
    RollingEqualCorrelationStrategy,
)
from portfolio.backtest import PortfolioBacktester


def main():
    prices = load_prices("data/smi_prices_cleaned.xlsx")
    df_ret = compute_returns(prices)

    bt = PortfolioBacktester(
        prices_path="data/smi_prices_cleaned.xlsx",
        benchmark_path="data/smi.xlsx",
        start_year=2023,
    )

    strategies = [
        (
            "Mutual Information",
            RollingMVStrategy(
                EntropyMIEstimator(),
                start_year=2023,
                lookback_years=1,
                target_return=None,
            ),
        ),
        (
            "MTFDC",
            RollingMVStrategy(
                TFDC_Estimator(),
                start_year=2023,
                lookback_years=1,
                target_return=None,
            ),
        ),
        (
            "Pearson",
            RollingMVStrategy(
                CovarianceEstimator(),
                start_year=2023,
                lookback_years=1,
                target_return=None,
            ),
        ),
        (
            "Spearman",
            RollingMVStrategy(
                SpearmanEstimator(),
                start_year=2023,
                lookback_years=1,
                target_return=None,
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
        results.append(bt.compute(w, name=name))
    plot_cumulative(
        results,
        benchmark_label="SMI",
        save_path="output/MV_smi_results.pdf",
    )

    summary = build_summary_table(results, "SMI")
    summary_fmt = format_summary_table(summary)
    print(summary_fmt)
    summary_fmt.to_excel("output/MV_smi_summary_table.xlsx")


if __name__ == "__main__":
    main()
