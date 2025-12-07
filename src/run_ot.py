import numpy as np
import pandas as pd
from typing import Literal

from portfolio.utils import load_prices, compute_returns
from portfolio.mean_variance import min_risk

from measures.ot.copulas import build_standard_correlation_targets
from measures.ot.ot import tfdc_matrix


def rolling_mean_tfdc(
    df_ret: pd.DataFrame,
    start_year: int = 2024,
    lookback_years: int = 2,
    n_bins: int = 10,
    reg: float = 0.1,
    target_return: float | None = 0.0004,
    method: Literal["sinkhorn", "wasserstein"] = "sinkhorn",
) -> pd.DataFrame:
    """
    Für jeden Rebalancing-Zeitpunkt (z.B. Monatsanfang) wird ein
    Minimum-Risk-Portfolio berechnet, aber mit der TFDC-Abhängigkeits-
    matrix (Optimal Transport auf Copulas) anstelle der Kovarianzmatrix.

    Nutzt:
        - tfdc_matrix(window_ret, target_copulas, forget_copulas, ...)
        - portfolio.mean_variance.min_risk(mu, Sigma, target_return, short_selling=False)

    Parameters
    ----------
    df_ret : DataFrame (T x N)
        tägliche Returns, Index = Datum, Spalten = Assets.
    start_year : int
        Ab welchem Jahr Rebalancing beginnen soll (z.B. 2023).
    lookback_years : int
        Anzahl Jahre für Lookback (hier 1).
    n_bins : int
        Anzahl Bins für das Copula-Gitter (z.B. 20 -> 20x20).
    reg : float
        Sinkhorn-Regularisierungsparameter für OT.
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
    weights_list: list[np.ndarray] = []
    index_list: list[pd.Timestamp] = []

    # Target- und Forget-Copulas einmalig bauen (für alle Fenster gleich)
    target_copulas, forget_copulas = build_standard_correlation_targets(
        n_bins=n_bins,
        n_samples=50_000,
        random_state=42,
    )

    for d in rebal_dates:
        # Lookback: 1 Jahr bis Tag vor Rebalancing
        window_start = d - pd.DateOffset(years=lookback_years)
        window_end = d - pd.Timedelta(days=1)

        window_ret = df_ret.loc[window_start:window_end]

        # Wenig Daten? -> überspringen
        if len(window_ret) < 150:
            continue

        # ---------- NEU: TFDC-Abhängigkeitsmatrix ----------
        Sigma_df = tfdc_matrix(
            window_ret,
            target_copulas=target_copulas,
            forget_copulas=forget_copulas,
            n_bins=n_bins,
            reg=reg,
            p=2,
            method=method,
        )
        # Optional zum Debuggen:
        print(Sigma_df)

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

    # 3) Rolling Min-Risk mit 1 Jahr Lookback ab 2023 (TFDC-Matrix)
    weights_df = rolling_mean_tfdc(
        df_ret,
        start_year=2023,
        lookback_years=1,
        n_bins=20,
        reg=5e-3,
        target_return=0.0004,
        method="wasserstein",
    )

    print("TFDC Min-Risk-Gewichte (1Y Lookback, Monatsanfang):")
    print(weights_df)

    # Optional: als CSV speichern
    weights_df.to_csv("ot_weights.csv")


if __name__ == "__main__":
    main()
