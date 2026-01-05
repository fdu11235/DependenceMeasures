import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_prices(path: str) -> pd.DataFrame:
    """
    Load price data
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.drop(
        columns=["AMRZ.S", "ACLN.S", "SUNN.S", "SDZ.S", "GALD.S"], errors="ignore"
    )
    return df


def compute_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage returns
    """
    df_ret = df_prices.pct_change().dropna()
    return df_ret


def expand_weights_to_daily(
    weights: pd.DataFrame, price_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Maps monthly (or other low-frequency) rebalancing weights to a daily frequency.

    Logic:
      - Each weight row (e.g., 2023-01-01) is aligned to the next available
        trading day in the price_index.
      - Between rebalancing dates, the most recent weights are carried forward (ffill).
      - Before the first rebalancing date, the portfolio remains in cash
        (NaN weights indicate no positions).
    """
    # initialize empty df
    w_daily = pd.DataFrame(index=price_index, columns=weights.columns, dtype=float)

    # for every rebalance date find next trading date
    for date, row in weights.iterrows():
        mask = price_index >= date
        if not mask.any():
            continue
        trade_day = price_index[mask][0]
        w_daily.loc[trade_day] = row.values

    # fill on first valid weight
    first_valid = w_daily.first_valid_index()
    if first_valid is None:
        return w_daily

    w_daily.loc[first_valid:] = w_daily.loc[first_valid:].ffill()

    # normalise
    mask = w_daily.notna().any(axis=1)
    row_sums = w_daily.loc[mask].sum(axis=1)
    w_daily.loc[mask] = w_daily.loc[mask].div(row_sums, axis=0)

    return w_daily


def load_benchmark(path: str, col_name: str | None = None) -> pd.Series:
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if col_name is None:
        col_name = df.columns[0]

    benchmark = df[col_name].astype(float)
    benchmark.name = col_name

    benchmark = benchmark[~benchmark.index.duplicated(keep="last")]

    return benchmark


def build_portfolio_and_benchmark_returns_static(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    benchmark_series: pd.Series,
    start_year: int = 2023,
) -> tuple[pd.Series, pd.Series]:
    """
    Build daily portfolio and benchmark return series under constant-shares
    between rebalancing dates.

    Procedure:
      - Align tickers between prices and weight data.
      - Determine a common start date (not earlier than `start_year` and not
        earlier than the first available price, weight or benchmark observation).
      - Map each low-frequency weight vector (e.g. monthly) to a specific
        trading day, which becomes a rebalancing day.
      - Simulate the portfolio with:
          * an initial wealth of 1.0,
          * a constant number of shares in each asset between rebalancing days,
          * new holdings computed only on rebalancing days from the chosen
            weights and current portfolio value.
      - Compute daily portfolio returns from changes in total portfolio value.
      - Compute daily benchmark returns from the benchmark index.

    This corresponds to a realistic monthly rebalancing scheme with
    buy-and-hold behaviour between rebalancing dates.
    """
    # Restrict to tickers that are available both in prices and weights
    common_cols = prices.columns.intersection(weights_df.columns)
    prices = prices[common_cols]
    weights_df = weights_df[common_cols]

    # --- determine common start date ---
    start_date_candidate = pd.Timestamp(f"{start_year}-01-01")
    first_price_date = prices.index.min()
    first_weight_date = weights_df.index.min()
    first_benchmark_date = benchmark_series.index.min()

    start_date = max(
        start_date_candidate, first_price_date, first_weight_date, first_benchmark_date
    )

    # Restrict all series to the common time horizon
    prices = prices.loc[start_date:]
    weights_df = weights_df.loc[start_date:]
    benchmark_series = benchmark_series.loc[start_date:]

    # Ensure everything is sorted by date
    prices = prices.sort_index()
    weights_df = weights_df.sort_index()
    benchmark_series = benchmark_series.sort_index()

    price_index = prices.index
    assets = prices.columns
    n_assets = len(assets)

    # --- map each weight date to the next available trading day ---
    trade_days = []
    trade_weights = []

    for date, row in weights_df.iterrows():
        mask = price_index >= date
        if not mask.any():
            continue
        trade_day = price_index[mask][0]
        trade_days.append(trade_day)
        trade_weights.append(row)

    if len(trade_days) == 0:
        # No valid rebalancing dates in the horizon
        # Return zero returns for both series
        idx = price_index
        port_ret = pd.Series(0.0, index=idx, name="MV_Portfolio")
        benchmark_aligned = benchmark_series.reindex(idx).ffill()
        bench_ret = benchmark_aligned.pct_change().fillna(0.0)
        bench_ret.name = "Benchmark"
        return port_ret, bench_ret

    # Build DataFrame of mapped weights, group by trade day in case of duplicates
    weights_mapped = pd.DataFrame(trade_weights, index=pd.DatetimeIndex(trade_days))
    # if multiple rows map to same trade_day, keep the last one
    weights_mapped = weights_mapped.groupby(level=0).last()
    # normalise each weight vector to sum to 1
    row_sums = weights_mapped.sum(axis=1)
    weights_mapped = weights_mapped.div(row_sums, axis=0)

    # Quick lookup if a day is a rebalance day, and its weights
    rebalance_days = set(weights_mapped.index)

    # --- simulate portfolio with constant shares between rebalancing dates ---
    holdings = np.zeros(n_assets, dtype=float)  # number of shares per asset
    V_prev = 1.0  # initial portfolio wealth
    started = False  # becomes True once we have taken the first position

    port_values = []
    port_index = []

    for t in price_index:
        prices_t = prices.loc[t].values.astype(float)
        if not started:
            # Before first rebalancing: portfolio stays in cash
            if t in rebalance_days:
                # open initial position at this date using V_prev and current prices
                w = weights_mapped.loc[t].reindex(assets).values.astype(float)
                # avoid division by zero if any price is zero (very unlikely)
                holdings = np.where(prices_t != 0, w * V_prev / prices_t, 0.0)
                V_t = np.dot(holdings, prices_t)
                # start the portfolio here, first return is defined as zero
                started = True
            else:
                # still in cash
                V_t = V_prev
            r_t = 0.0
        else:
            # Portfolio already invested: value evolves with current holdings
            V_t = np.dot(holdings, prices_t)
            r_t = V_t / V_prev - 1.0 if V_prev != 0 else 0.0
            # At rebalancing days, adjust holdings for next day using end-of-day wealth
            if t in rebalance_days:
                w = weights_mapped.loc[t].reindex(assets).values.astype(float)
                holdings = np.where(prices_t != 0, w * V_t / prices_t, 0.0)

        V_prev = V_t
        port_values.append(r_t)
        port_index.append(t)

    port_ret = pd.Series(port_values, index=pd.DatetimeIndex(port_index))
    port_ret.name = "MV_Portfolio"

    # --- benchmark returns from index ---
    benchmark_aligned = benchmark_series.reindex(price_index).ffill()
    bench_ret = benchmark_aligned.pct_change().fillna(0.0)
    bench_ret.name = "Index Benchmark"

    # Align both series
    idx = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[idx]
    bench_ret = bench_ret.loc[idx]

    return port_ret, bench_ret


def plot_cumulative(
    results,
    benchmark_label: str,
    save_path: str | None = None,
):
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=(12, 6))

        for r in results:
            r.cum_port.plot(ax=ax, label=r.name)

        if results:
            results[0].cum_bench.plot(
                ax=ax,
                label=benchmark_label,
                linewidth=2,
            )

        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Portfolio Value (Normalized)")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def build_summary_table(results: list, benchmark) -> pd.DataFrame:
    """
    Build a comparison table for strategies + one benchmark row.

    Expects each result to expose:
      - name
      - total_return_port, annualized_return_port, sharpe_port, max_dd_port
      - total_return_bench, annualized_return_bench, sharpe_bench, max_dd_bench
    """
    if not results:
        return pd.DataFrame(
            columns=[
                "Total Return",
                "Ann. Return",
                "Std. (ann.)",
                "Sharpe",
                "Max Drawdown",
            ]
        )

    rows = []

    # strategies
    for r in results:
        rows.append(
            {
                "Name": r.name,
                "Total Return": r.total_return_port,
                "Ann. Return": r.annualized_return_port,
                "Std. (ann.)": r.std_port,
                "Sharpe": r.sharpe_port,
                "Max Drawdown": r.max_dd_port,
            }
        )

    # benchmark once
    b = results[0]
    rows.append(
        {
            "Name": benchmark,
            "Total Return": b.total_return_bench,
            "Ann. Return": b.annualized_return_bench,
            "Std. (ann.)": b.std_bench,
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
    out["Std. (ann.)"] = out["Std. (ann.)"].map(
        lambda x: f"{x:.2%}" if np.isfinite(x) else "nan"
    )
    out["Max Drawdown"] = out["Max Drawdown"].map(
        lambda x: f"{x:.2%}" if np.isfinite(x) else "nan"
    )
    out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "nan")
    return out
