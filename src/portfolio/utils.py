import pandas as pd
import numpy as np


def load_prices(path: str) -> pd.DataFrame:
    """
    Load price data
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Problemspalte droppen (falls vorhanden)
    df = df.drop(columns=["AMRZ.S", "ACLN.S", "SUNN.S", "GALD.S"], errors="ignore")

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
    # Leeres Daily-DF mit allen Handelstagen
    w_daily = pd.DataFrame(index=price_index, columns=weights.columns, dtype=float)

    # Für jedes Rebalancing-Datum den nächsten Handelstag suchen und dort die Weights setzen
    for date, row in weights.iterrows():
        # alle Handelstage >= diesem Datum
        mask = price_index >= date
        if not mask.any():
            continue  # falls Weight nach letztem Preisdatum liegt
        trade_day = price_index[mask][0]
        w_daily.loc[trade_day] = row.values

    # Ab erstem gesetzten Weight forward-fillen
    first_valid = w_daily.first_valid_index()
    if first_valid is None:
        # keine Weights gesetzt -> alles Cash
        return w_daily

    w_daily.loc[first_valid:] = w_daily.loc[first_valid:].ffill()

    # Nur Zeilen normalisieren, wo es überhaupt Gewichte gibt
    mask = w_daily.notna().any(axis=1)
    row_sums = w_daily.loc[mask].sum(axis=1)
    w_daily.loc[mask] = w_daily.loc[mask].div(row_sums, axis=0)

    return w_daily


def load_smi_benchmark(path: str, col_name: str | None = None) -> pd.Series:
    """
    Load SMI dataset
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if col_name is None:
        col_name = df.columns[0]

    smi = df[col_name].astype(float)
    smi.name = "SMI"

    # zur Sicherheit auch hier nochmal deduplizieren
    smi = smi[~smi.index.duplicated(keep="last")]

    return smi


def build_portfolio_and_benchmark_returns(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    smi_series: pd.Series,
    start_year: int = 2023,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns tuple:
      - port_ret: weighted portfolio returns
      - bench_ret: daily benchmark returns
    """
    # common tickers
    common_cols = prices.columns.intersection(weights_df.columns)
    prices = prices[common_cols]
    weights_df = weights_df[common_cols]

    # --- start date ---
    start_date_candidate = pd.Timestamp(f"{start_year}-01-01")
    first_price_date = prices.index.min()
    first_weight_date = weights_df.index.min()
    first_smi_date = smi_series.index.min()

    start_date = max(
        start_date_candidate, first_price_date, first_weight_date, first_smi_date
    )

    # alles auf Zeitraum ab start_date beschränken
    prices = prices.loc[start_date:]
    weights_df = weights_df.loc[start_date:]
    smi_series = smi_series.loc[start_date:]

    # Asset Returns (ab Start)
    asset_ret = prices.pct_change().fillna(0.0)

    # Weights auf tägliche Frequenz bringen (nur ab Startdate)
    weights_daily = expand_weights_to_daily(weights_df, prices.index)

    # Safety: Shapes prüfen
    assert weights_daily.index.equals(asset_ret.index)
    assert (weights_daily.columns == asset_ret.columns).all()

    # Portfolio-Return
    port_ret = (weights_daily * asset_ret).sum(axis=1)
    port_ret.name = "MV_Portfolio"

    # Benchmark-Returns
    smi_series = smi_series.reindex(prices.index).ffill()
    bench_ret = smi_series.pct_change().fillna(0.0)
    bench_ret.name = "SMI_Benchmark"

    # gemeinsamer Zeitraum (zur Sicherheit)
    idx = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[idx]
    bench_ret = bench_ret.loc[idx]

    return port_ret, bench_ret
