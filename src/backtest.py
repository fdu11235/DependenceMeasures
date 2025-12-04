import pandas as pd
import numpy as np
import vectorbt as vbt


# -------------------------------------------------------------------
# 1) Daten laden
# -------------------------------------------------------------------


def load_asset_prices(path: str) -> pd.DataFrame:
    """
    Lädt die Asset-Preise (z.B. SMI-Konstituenten).

    Erwartet:
        - erste Spalte: Datum (oder bereits Index)
        - restliche Spalten: Tickers (z.B. 'SIKA.S', 'NESN.S', ...)

    Passen falls nötig: index_col / parse_dates.
    """
    df = pd.read_excel(path)
    # falls deine Datei schon den Index enthält, dann:
    # df = pd.read_excel(path, index_col=0, parse_dates=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Falls AMRZ.S drin ist und gedroppt werden soll:
    df = df.drop(columns=["AMRZ.S"], errors="ignore")

    return df


def load_weights(path: str) -> pd.DataFrame:
    """
    Lädt die MV-Gewichte (Output von rolling_mean_variance_1y).

    Erwartet:
        - Index: Rebalancing-Daten (z.B. Monatsende)
        - Spalten: Tickers (gleich wie in Prices)
    """
    w = pd.read_csv(path, index_col=0)
    w.index = pd.to_datetime(w.index)
    w = w.sort_index()
    return w


def load_smi_benchmark(path: str, col_name: str = None) -> pd.Series:
    """
    Lädt die SMI-Zeitreihe als Benchmark.

    Wenn col_name None ist, wird die erste Datenspalte verwendet.
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if col_name is None:
        # nimm die erste Nicht-Datum-Spalte
        col_name = df.columns[0]

    smi = df[col_name]
    smi.name = "SMI"
    return smi


# -------------------------------------------------------------------
# 2) Weights auf tägliche Frequenz bringen
# -------------------------------------------------------------------


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


# -------------------------------------------------------------------
# 3) Backtest mit vectorbt
# -------------------------------------------------------------------


def run_backtest(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    smi_series: pd.Series,
    init_cash: float = 100_000.0,
):
    """
    Erzeugt zwei Portfolios mit vectorbt:
        - MV-Portfolio mit dynamischen Gewichten
        - SMI-Benchmark (Buy&Hold)

    Gibt Portfolio-Objekte zurück.
    """
    # Preise und Gewichte alignen
    # Nur die Tickers verwenden, die in beiden vorkommen
    common_cols = prices.columns.intersection(weights_df.columns)
    prices = prices[common_cols]
    weights_df = weights_df[common_cols]

    # auf Zeitraum ab erstem Rebalancing beschränken
    start_date = weights_df.index.min()
    prices = prices.loc[start_date:]
    smi_series = smi_series.loc[start_date:]

    # Gewichte auf tägliche Frequenz expandieren
    weights_daily = expand_weights_to_daily(weights_df, prices.index)
    print(weights_daily)
    # Sicherstellen, dass Shapes passen
    assert weights_daily.index.equals(prices.index)
    assert (weights_daily.columns == prices.columns).all()

    # MV-Portfolio
    pf_mv = vbt.Portfolio.from_orders(
        close=prices,
        size=weights_daily,
        size_type="targetpercent",
        init_cash=init_cash,
    )

    return pf_mv


# -------------------------------------------------------------------
# 4) Main
# -------------------------------------------------------------------


def main():
    # Pfade ggf. anpassen
    prices_path = "data/smi_prices_cleaned.xlsx"  # oder "data/smi_prices_cleaned.xlsx"
    weights_path = "mv_weights_1y_2023_onwards.csv"
    smi_path = "data/smi.xlsx"

    # Daten laden
    prices = load_asset_prices(prices_path)
    weights = load_weights(weights_path)
    print(weights)
    smi = load_smi_benchmark(smi_path, col_name=None)  # evtl. Spaltennamen setzen

    # Backtest laufen lassen
    pf_mv = run_backtest(prices, weights, smi, init_cash=100_000)

    # Stats
    print("=== Mean-Variance Portfolio (1Y Lookback) ===")
    print(pf_mv.stats())

    # Optional: Plot
    # pf_mv.value().vbt.plot().show()
    (pf_mv.returns() + 1).cumprod().vbt.plot(
        title="Cumulative Return – MV Portfolio"
    ).show()
    # pf_bench.value().vbt.plot().show()
    # oder gemeinsam:
    # vbt.Chart(pf_mv.value(), pf_bench.value()).show()


if __name__ == "__main__":
    main()
