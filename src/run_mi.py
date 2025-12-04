import pandas as pd
import numpy as np
from portfolio.mean_variance import min_risk
from measures.mi.entropy_mi import entropy_mi_matrix
from portfolio.utils import load_prices, compute_returns


def rolling_mean_entropy(
    df_ret: pd.DataFrame,
    start_year: int = 2023,
    lookback_years: int = 1,
    min_ret: float = -0.5,
    max_ret: float = 0.5,
    n_bins: int = 101,
    target_return: float | None = 0.00038,
) -> pd.DataFrame:
    """
    Für jeden Rebalancing-Zeitpunkt (z.B. Monatsanfang) wird ein
    Minimum-Risk-Portfolio berechnet, aber mit der Entropy+MI-Risiko-
    matrix aus Novais et al. (2022) anstelle der Kovarianzmatrix.

    Nutzt:
        - entropy_mi_matrix(window_ret, min_ret, max_ret, n_bins)
        - portfolio.mean_variance.min_risk(mu, Sigma, target_return, short_selling=False)

    Parameters
    ----------
    df_ret : DataFrame (T x N)
        tägliche Returns, Index = Datum, Spalten = Assets.
    start_year : int
        Ab welchem Jahr Rebalancing beginnen soll (z.B. 2023).
    lookback_years : int
        Anzahl Jahre für Lookback (hier 1).
    min_ret, max_ret, n_bins :
        Parameter für die Diskretisierung in entropy_mi_matrix.
    target_return : float oder None
        Zielrendite für min_risk. Wenn None, wird reines Min-Risk ohne
        Renditeconstraint gemacht (abhängig von deiner min_risk-Implementierung).

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

        # ---------- NEU: Entropy + Mutual Information Matrix ----------
        Sigma_df = entropy_mi_matrix(
            window_ret,
            min_ret=min_ret,
            max_ret=max_ret,
            n_bins=n_bins,
        )
        Sigma = Sigma_df.to_numpy()

        # Erwartungswerte
        mu = window_ret.mean().to_numpy()

        # ggf. target_return anpassen / None setzen
        w = min_risk(mu, Sigma, target_return=target_return, short_selling=False)

        weights_list.append(w)
        index_list.append(d)

    weights_df = pd.DataFrame(
        data=np.vstack(weights_list),
        index=index_list,  # -> MonatsANFANG
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

    # 3) Rolling Min-Risk mit 1 Jahr Lookback ab 2023 (Entropy+MI-Matrix)
    weights_df = rolling_mean_entropy(
        df_ret,
        start_year=2023,
        lookback_years=1,
        min_ret=-0.5,
        max_ret=0.5,
        n_bins=101,
        target_return=0.0004,  # oder None, wenn du nur Risiko minimieren willst
    )

    print("Entropy+MI Min-Risk-Gewichte (1Y Lookback, Monatsanfang):")
    print(weights_df)

    # Optional: als CSV speichern
    weights_df.to_csv("mi_weights.csv")


if __name__ == "__main__":
    main()
