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
                # use_auto_target=True,
            ),
        ),
        (
            "TFDC",
            RollingMVStrategy(
                TFDCEstimator(),
                start_year=2023,
                lookback_years=1,  # use_auto_target=True
            ),
        ),
        (
            "Pearson",
            RollingMVStrategy(
                CovarianceEstimator(),
                start_year=2023,
                lookback_years=1,
                # use_auto_target=True,
            ),
        ),
        (
            "Spearman",
            RollingMVStrategy(
                SpearmanEstimator(),
                start_year=2023,
                lookback_years=1,
                # use_auto_target=True,
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

        # clean filename
        fname = name.lower().replace(" ", "_")
        w.to_csv(f"output/smi_target/weights_{fname}.csv", index_label="date")
        results.append(bt.compute(w, name=name))
    plot_cumulative(
        results,
        benchmark_label="SMI",
        save_path="output/smi/smi_expanded_results.pdf",
    )

    summary = build_summary_table(results, "SMI")
    summary_fmt = format_summary_table(summary)
    print(summary_fmt)
    summary_fmt.to_excel("output/smi/MV_smi_summary_table.xlsx")


if __name__ == "__main__":
    main()
