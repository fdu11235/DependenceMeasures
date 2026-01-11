# DependenceMeasures
Research on alternative dependence measures for portfolio construction AREP HS25

## Running the experiments

To reproduce the portfolio construction and backtesting results, run one of the following scripts:

- `run_mv_multi.py` — runs the mean–variance portfolio experiments across multiple dependence measures.
- `run_erc_multi.py` — runs the equal risk contribution (ERC) portfolio experiments across multiple dependence measures.
- `run_ec_multi.py` — runs the equal correlation (EC) portfolio experiments across multiple dependence measures.

Each script computes portfolio weights, performs backtesting, and outputs performance metrics for the specified configuration.

## Note

Use  `pip install pot` for Optimal Transport library
