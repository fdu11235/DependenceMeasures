import pandas as pd
import numpy as np
from portfolio.mean_variance import min_risk
from portfolio.utils import load_prices, compute_returns


def rolling_mean_variance(
    df_ret: pd.DataFrame,
    start_year: int = 2023,
    lookback_years: int = 1,
    target_return: float | None = 0.00038,
) -> pd.DataFrame:
    """
    Für jeden Rebalancing-Zeitpunkt (z.B. Monatsanfang) wird ein
    Minimum-Risk-Portfolio per Mean-Variance (min_risk) berechnet,
    basierend auf einem 1-Jahres-Lookback.

    Nutzt deine vorhandene Funktion:
        portfolio.mean_variance.min_risk(mu, Sigma, target_return=None, short_selling=False)

    Parameters
    ----------
    df_ret : DataFrame (T x N)
        tägliche Returns, Index = Datum, Spalten = Assets.
    start_year : int
        Ab welchem Jahr Rebalancing beginnen soll (z.B. 2023).
    rebalance_freq : str
        Resampling-Frequenz ('M' = Monatsende).
    lookback_years : int
        Anzahl Jahre für Lookback (hier 1).

    Returns
    -------
    weights_df : DataFrame
        Index = Rebalancing-Daten, Spalten = Asset-Namen, Werte = Gewichte.
    """
    df_ret = df_ret.sort_index()

    # nur Daten ab start_year
    df_after_start = df_ret.loc[f"{start_year}-01-01":]

    # 'MS' = Month Start, .first() = erster Handelstag im Monat
    rebal_dates = df_after_start.resample("MS").first().index

    tickers = df_ret.columns
    weights_list = []
    index_list = []

    for d in rebal_dates:
        # Lookback: 1 Jahr bis Tag vor Rebalancing
        window_start = d - pd.DateOffset(years=lookback_years)
        window_end = d - pd.Timedelta(days=1)

        window_ret = df_ret.loc[window_start:window_end]

        # Wenig Daten? -> überspringen
        if len(window_ret) < 150:
            continue

        Sigma_df = window_ret.cov()
        Sigma = Sigma_df.to_numpy()
        mu = window_ret.mean().to_numpy()

        # ggf. target_return anpassen / None setzen
        w = min_risk(mu, Sigma, target_return=target_return, short_selling=False)

        weights_list.append(w)
        index_list.append(d)

    weights_df = pd.DataFrame(
        data=np.vstack(weights_list),
        index=index_list,
        columns=tickers,
    )

    return weights_df


def main():
    # Pfad zu deiner Excel-Datei
    path = "data/smi_prices_cleaned.xlsx"

    # 1) Preise laden
    df_prices = load_prices(path)

    # 2) Returns berechnen
    df_ret = compute_returns(df_prices)

    # 3) Rolling Mean-Variance mit 1 Jahr Lookback ab 2023
    weights_df = rolling_mean_variance(df_ret, start_year=2023, target_return=0.0004)

    # Optional: als CSV speichern
    weights_df.to_csv("mv_weights.csv")


if __name__ == "__main__":
    main()
