import numpy as np
from scipy.optimize import minimize


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    if s == 0:
        return w
    return w / s


# Objective: squared distance of RCs to their mean
def objective(w: np.ndarray, Sigma) -> float:
    w = np.asarray(w)
    rc = risk_contributions(w, Sigma)
    avg_rc = rc.mean()
    return float(((rc - avg_rc) ** 2).sum())


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
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square (N x N). Got shape {Sigma.shape}.")

    # enforce symmetry (helps numerics)
    Sigma = 0.5 * (Sigma + Sigma.T)

    n = Sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, Sigma)

        rc_sum = float(np.sum(rc))
        if rc_sum <= 0 or not np.isfinite(rc_sum):
            return 1e6  # penalty to keep optimizer away from bad points

        rc_share = rc / rc_sum  # normalize contributions
        target = 1.0 / n
        return float(np.sum((rc_share - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if short_selling else [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-15, "disp": False},
    )

    if not res.success:
        raise RuntimeError(f"ERC optimization failed: {res.message}")

    w_raw = res.x
    w = w_raw / w_raw.sum()

    rc = risk_contributions(w, Sigma)
    rc_share = rc / rc.sum()

    print("ERC check:")
    print("RC share std:", rc_share.std())
    print("RC share min/max:", rc_share.min(), rc_share.max())

    # -----------------------
    return w
