from typing import Optional
import numpy as np
from scipy.optimize import minimize


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize weights so that sum(w) = 1.
    """
    s = w.sum()
    if s == 0:
        return w
    return w / s


def min_risk(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: Optional[float] = None,
    short_selling: bool = False,
) -> np.ndarray:
    """
    Markowitz-style minimum-risk portfolio:

        min_w   w^T Σ w
        s.t.    sum(w) = 1
                mu^T w >= target_return    (optional)
                w_i >= 0 if short_selling=False

    Parameters
    ----------
    mu : array (N,)
        Expected returns (same order as Sigma columns).
    Sigma : array (N x N)
        Risk matrix (covariance, entropy+MI, copula+OT, ...).
    target_return : float or None
        If provided, enforces mu^T w >= target_return.
        If None, pure minimum-risk portfolio.
    short_selling : bool
        If False, enforce w_i >= 0.

    Returns
    -------
    w_opt : np.ndarray (N,)
        Optimal weights (sum to 1).
    """
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    n = len(mu)

    # Objective: quadratic risk
    def objective(w: np.ndarray) -> float:
        return float(w.T @ Sigma @ w)

    # Constraints
    constraints = []

    # Sum of weights = 1
    constraints.append(
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }
    )

    # Optional target return constraint
    if target_return is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: mu @ w - target_return,
            }
        )

    # Bounds for weights
    if short_selling:
        bounds = None  # no bounds, can be negative
    else:
        bounds = [(0.0, 1.0) for _ in range(n)]

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return _normalize_weights(res.x)


def max_sharpe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float = 0.0,
    short_selling: bool = False,
) -> np.ndarray:
    """
    Maximize Sharpe ratio:

        max_w   ( (mu - rf)^T w / sqrt(w^T Σ w) )
        s.t.    sum(w) = 1
                w_i >= 0 if short_selling=False

    We solve this by minimizing the negative Sharpe.

    Parameters
    ----------
    mu : array (N,)
        Expected returns.
    Sigma : array (N x N)
        Risk matrix.
    risk_free_rate : float
        Risk-free rate per period (same frequency as returns).
    short_selling : bool
        If False, enforce w_i >= 0.

    Returns
    -------
    w_opt : np.ndarray (N,)
        Optimal weights (sum to 1).
    """
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        var = float(w.T @ Sigma @ w)
        if var <= 1e-16:
            return 1e6  # super high penalty if variance ~ 0
        vol = np.sqrt(var)
        excess_ret = (mu - risk_free_rate) @ w
        return -excess_ret / vol

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]

    if short_selling:
        bounds = None
    else:
        bounds = [(0.0, 1.0) for _ in range(n)]

    w0 = np.ones(n) / n

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not res.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {res.message}")

    return _normalize_weights(res.x)
