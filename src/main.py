from portfolio.utils import load_prices, compute_returns
from measures.risk_estimators import EntropyMIEstimator
from portfolio.strategies import RollingMVStrategy
from portfolio.backtest import PortfolioBacktester


def main():
    prices = load_prices("data/smi_prices_cleaned.xlsx")
    df_ret = compute_returns(prices)

    strategy = RollingMVStrategy(
        risk_estimator=EntropyMIEstimator(),
        start_year=2023,
        lookback_years=1,
    )

    weights = strategy.compute_weights(df_ret)
    weights.to_csv("mi_weights.csv")

    bt = PortfolioBacktester(
        prices_path="data/smi_prices_cleaned.xlsx",
        smi_path="data/smi.xlsx",
        start_year=2023,
    )

    print(bt.run(weights))


if __name__ == "__main__":
    main()
