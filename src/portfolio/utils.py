import pandas as pd
import numpy as np


def load_prices(path: str) -> pd.DataFrame:
    """
    Lädt Preis-Daten aus Excel.
    Erwartet eine Spalte 'Date' und danach die Tickers als Spalten.
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Problemspalte droppen (falls vorhanden)
    df = df.drop(columns=["AMRZ.S"], errors="ignore")

    return df


def compute_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet tägliche prozentuale Returns aus Preisen.
    """
    df_ret = df_prices.pct_change().dropna()
    return df_ret
