import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1) Daten laden
# ------------------------------------------------------------


def load_prices(path: str) -> pd.DataFrame:
    """
    Lädt Preis-Daten der SMI-Aktien.
    Erwartet eine Spalte 'Date' und danach die Tickers als Spalten.
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # falls es mehrere Zeilen mit dem gleichen Datum gibt → letzte nehmen
    df = df[~df.index.duplicated(keep="last")]

    df = df.drop(columns=["AMRZ.S"], errors="ignore")
    return df


def load_weights(path: str) -> pd.DataFrame:
    """
    Lädt die MV-Gewichte aus CSV.
    Index = Rebalancing-Daten (Monatsanfang),
    Spalten = Tickers.
    """
    w = pd.read_csv(path, index_col=0)
    w.index = pd.to_datetime(w.index)
    w = w.sort_index()
    return w


def load_smi_benchmark(path: str, col_name: str | None = None) -> pd.Series:
    """
    Lädt die SMI-Zeitreihe aus Excel.
    Wenn col_name None ist, wird die erste Datenspalte genommen.
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


# ------------------------------------------------------------
# 2) Weights auf tägliche Frequenz bringen
# ------------------------------------------------------------


def expand_weights_to_daily(
    weights: pd.DataFrame, price_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Bringt Rebalancing-Gewichte (Monatsanfang-Kalenderdatum) auf tägliche Frequenz.

    Logik:
      - Jede Weight-Zeile (z.B. 2023-01-01) wird auf den
        nächsten verfügbaren Handelstag im price_index gemappt.
      - Zwischen den Rebalancing-Daten werden die letzten Gewichte gehalten (ffill).
      - Vor dem ersten Rebalancing bleibt das Portfolio in Cash (NaN -> keine Orders).
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


# ------------------------------------------------------------
# 3) Portfolio- und Benchmark-Returns bauen
# ------------------------------------------------------------


def build_portfolio_and_benchmark_returns(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    smi_series: pd.Series,
    start_year: int = 2023,
) -> tuple[pd.Series, pd.Series]:
    """
    Erzeugt:
      - port_ret: tägliche Portfolio-Returns basierend auf Gewichten
      - bench_ret: tägliche Benchmark-Returns (SMI)
    Startet erst ab start_year (z.B. 2023) bzw. ab erstem verfügbaren Weight.
    """
    # gemeinsame Tickers
    common_cols = prices.columns.intersection(weights_df.columns)
    prices = prices[common_cols]
    weights_df = weights_df[common_cols]

    # --- sinnvolles Startdatum bestimmen ---
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


# ------------------------------------------------------------
# 4) Main: alles zusammenstecken + Stats + Quantstats-Report
# ------------------------------------------------------------


def main():
    prices_path = "data/smi_prices_cleaned.xlsx"
    weights_path = "mv_mi_weights_1y_2023_onwards.csv"
    smi_path = "data/smi.xlsx"

    prices = load_prices(prices_path)
    weights = load_weights(weights_path)
    smi = load_smi_benchmark(smi_path, col_name=None)  # ggf. anpassen
    print(smi)

    port_ret, bench_ret = build_portfolio_and_benchmark_returns(prices, weights, smi)

    print("Erste Zeilen Portfolio-Returns:")
    print(port_ret.head())
    print("\nErste Zeilen Benchmark-Returns:")
    print(bench_ret.head())

    # ---------- Basic Stats MV-Portfolio ----------
    excess_p = port_ret  # rf = 0
    sharpe_p_daily = excess_p.mean() / excess_p.std()
    sharpe_p_annual = sharpe_p_daily * np.sqrt(252)

    cum_p = (1 + port_ret).cumprod()
    running_max_p = cum_p.cummax()
    drawdown_p = cum_p / running_max_p - 1.0
    max_dd_p = drawdown_p.min()

    # ---------- Basic Stats Benchmark ----------
    excess_b = bench_ret
    sharpe_b_daily = excess_b.mean() / excess_b.std()
    sharpe_b_annual = sharpe_b_daily * np.sqrt(252)

    cum_b = (1 + bench_ret).cumprod()
    running_max_b = cum_b.cummax()
    drawdown_b = cum_b / running_max_b - 1.0
    max_dd_b = drawdown_b.min()

    print("\n=== Basic Stats MV-Portfolio ===")
    print(f"Sharpe (annualisiert): {sharpe_p_annual:.3f}")
    print(f"Max Drawdown: {max_dd_p:.2%}")

    print("\n=== Basic Stats Benchmark (SMI) ===")
    print(f"Sharpe (annualisiert): {sharpe_b_annual:.3f}")
    print(f"Max Drawdown: {max_dd_b:.2%}")

    # optional: quantstats direkt
    qs.extend_pandas()
    print("\nQuantstats Sharpe MV:", port_ret.sharpe())
    print("Quantstats Max DD MV:", port_ret.max_drawdown())
    print("Quantstats Sharpe SMI:", bench_ret.sharpe())
    print("Quantstats Max DD SMI:", bench_ret.max_drawdown())

    # --- Cumulative Return Plot MV vs Benchmark ---
    mv_curve = (1 + port_ret).cumprod()
    bench_curve = (1 + bench_ret).cumprod()

    plt.figure(figsize=(12, 6))
    mv_curve.plot(label="MV Portfolio")
    bench_curve.plot(label="SMI Benchmark")
    plt.title("Cumulative Returns: MV vs SMI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
