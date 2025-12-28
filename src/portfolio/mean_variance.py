from typing import Optional
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from cvxopt import matrix, solvers


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize weights so that sum(w) = 1.
    """
    s = w.sum()
    if s == 0:
        return w
    return w / s


def mv_objective(w: np.ndarray, Sigma: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    return float(w @ Sigma @ w)


def mv_gradient(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    # if Sigma is symmetric, grad = 2 Sigma w
    return 2.0 * (Sigma @ w)


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
        bounds = [(0.0, None) for _ in range(n)]

    # Initial guess: equal weights
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    eps = 1e-3

    w0 = np.ones(n) / n
    w0 = w0 + eps * (rng.random(n) - 0.5)
    w0 = np.maximum(w0, 0.0)
    w0 = w0 / w0.sum()

    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma_sym)
    cond = eigvals.max() / max(eigvals.min(), 1e-16)

    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + 1e-6 * np.mean(np.diag(Sigma)) * np.eye(n)

    best = mv_objective(w0, Sigma)
    best_w = w0.copy()

    for _ in range(500):
        w = np.random.dirichlet(np.ones(n))  # long-only, sum=1
        val = mv_objective(w, Sigma)
        if val < best:
            best = val
            best_w = w

    print("objective(w0):", mv_objective(w0, Sigma))
    print("best random :", best)
    print("improvement :", mv_objective(w0, Sigma) - best)
    print("best_w std  :", best_w.std())

    def objective(w):
        return w @ Sigma @ w

    def grad(w):
        return 2.0 * Sigma @ w

    def hess(w):
        return 2.0 * Sigma

    lc = LinearConstraint(np.ones((1, n)), lb=1.0, ub=1.0)

    res = minimize(
        objective,
        w0,
        method="trust-constr",
        jac=grad,
        constraints=[lc],
        options={"verbose": 0},
    )

    print("nit:", res.nit, "nfev:", res.nfev)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return _normalize_weights(res.x)


def min_risk_cvxopt(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: Optional[float] = None,
    long_only: bool = True,
) -> np.ndarray:
    """
    Mean–variance minimum-risk portfolio solved as a convex QP via CVXOPT.

        min_w   w^T Σ w
        s.t.    1^T w = 1
                w >= 0           (if long_only)
                mu^T w >= r*     (optional)

    Parameters
    ----------
    mu : array (N,)
        Expected returns.
    Sigma : array (N x N)
        Covariance / risk matrix (must be symmetric PSD).
    target_return : float or None
        Optional minimum expected return constraint.
    long_only : bool
        Enforce w >= 0 if True.

    Returns
    -------
    w_opt : np.ndarray (N,)
        Optimal portfolio weights.
    """

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    # CVXOPT matrices
    P = matrix(Sigma)
    q = matrix(np.zeros(n))

    # Inequality constraints G w <= h
    G_list = []
    h_list = []

    if long_only:
        G_list.append(-np.eye(n))  # -w <= 0  → w >= 0
        h_list.append(np.zeros(n))

    if target_return is not None:
        G_list.append(-mu.reshape(1, -1))  # -mu^T w <= -r*
        h_list.append(np.array([-target_return]))

    if G_list:
        G = matrix(np.vstack(G_list))
        h = matrix(np.concatenate(h_list))
    else:
        G, h = None, None

    # Equality constraint: sum(w) = 1
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    # Solve QP
    solvers.options["show_progress"] = False
    sol = solvers.qp(P, q, G, h, A, b)

    if sol["status"] != "optimal":
        raise RuntimeError(f"CVXOPT failed: {sol['status']}")

    w = np.array(sol["x"]).flatten()

    # Numerical cleanup
    w[w < 0] = 0.0
    w = w / w.sum()

    return w
