import numpy as np
from scipy.optimize import minimize


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    if s == 0:
        return w
    return w / s


def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute risk contributions of each asset:

        RC_i = w_i * (Σ w)_i

    Parameters
    ----------
    w : array (N,)
        Portfolio weights.
    Sigma : array (N x N)
        Risk matrix.

    Returns
    -------
    rc : array (N,)
        Risk contribution per asset.
    """
    Sigma_w = Sigma @ w
    return w * Sigma_w


def equal_risk_contribution(
    Sigma: np.ndarray,
    short_selling: bool = False,
) -> np.ndarray:
    """
    Compute Equal Risk Contribution (ERC) portfolio:

        Find w s.t.:
            - sum(w) = 1
            - w_i >= 0 (if short_selling=False)
            - all risk contributions RC_i are as equal as possible

        We solve:
            min_w  sum_i (RC_i - avg_RC)^2
            s.t.   sum(w) = 1
                   w_i >= 0 (if no short-selling)

    Parameters
    ----------
    Sigma : array (N x N)
        Risk matrix (covariance, entropy+MI, copula+OT, ...).
    short_selling : bool
        If False, enforce w_i >= 0.

    Returns
    -------
    w_opt : np.ndarray (N,)
        ERC weights (sum to 1).
    """
    Sigma = np.asarray(Sigma)
    n = Sigma.shape[0]

    # Objective: squared distance of RCs to their mean
    def objective(w: np.ndarray) -> float:
        w = np.asarray(w)
        rc = risk_contributions(w, Sigma)
        avg_rc = rc.mean()
        return float(((rc - avg_rc) ** 2).sum())

    # Constraint: sum(w) = 1
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]

    # Bounds
    if short_selling:
        bounds = None  # weights können negativ sein
    else:
        bounds = [(0.0, 1.0) for _ in range(n)]

    # Start: Gleichgewichtete Gewichte
    w0 = np.ones(n) / n

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not res.success:
        raise RuntimeError(f"ERC optimization failed: {res.message}")

    return _normalize_weights(res.x)