import numpy as np
import pandas as pd
from typing import Literal
from .utils import empirical_copula_transform, empirical_copula_hist, grid_cost_matrix
from joblib import Parallel, delayed

try:
    import ot as ot_backend  # POT: Python Optimal Transport
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "measures.ot requires the 'POT' package for optimal transport. "
        "Install it with `pip install pot`."
    ) from e


def sinkhorn_distance(
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    M: np.ndarray | None = None,
    reg: float = 1e-2,
    p: int = 2,
    use_sqrt: bool = True,
) -> float:
    """
    Sinkhorn-regularized OT distance between two histograms.

    a_hist, b_hist : 2D or flattened histograms that sum to 1 (we renormalize).
    M : ground cost matrix; if None it's built assuming a regular n_bins x n_bins grid.
    reg : entropic regularization parameter.
    p   : order of ground metric if M is built inside.
    """
    eps = 1e-10

    a = np.asarray(a_hist, dtype=np.float64).ravel()
    b = np.asarray(b_hist, dtype=np.float64).ravel()
    a = np.maximum(a, eps)
    b = np.maximum(b, eps)
    a /= a.sum()
    b /= b.sum()

    n_bins2 = a.size
    if M is None:
        n_bins = int(np.sqrt(n_bins2))
        if n_bins * n_bins != n_bins2:
            raise ValueError("Cannot infer n_bins from histogram size.")
        M = grid_cost_matrix(n_bins, p=p)

    cost = ot_backend.sinkhorn2(a, b, M, reg)
    # POT can return (cost, log); keep the scalar part
    if isinstance(cost, (tuple, list)):
        cost = cost[0]
    cost = float(cost)
    return np.sqrt(cost) if use_sqrt else cost


def wasserstein_distance(
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    M: np.ndarray | None = None,
    p: int = 2,
    use_sqrt: bool = True,
) -> float:
    """
    Exact Wasserstein (EMD) distance between two histograms on the copula grid.
    """
    a = np.asarray(a_hist, dtype=np.float64).ravel()
    b = np.asarray(b_hist, dtype=np.float64).ravel()

    # no eps needed in theory, but you can keep a tiny one if you like
    a /= a.sum()
    b /= b.sum()

    n_bins2 = a.size
    if M is None:
        n_bins = int(np.sqrt(n_bins2))
        if n_bins * n_bins != n_bins2:
            raise ValueError("Cannot infer n_bins from histogram size.")
        M = grid_cost_matrix(n_bins, p=p)

    cost = ot_backend.emd2(a, b, M)  # exact OT cost
    cost = float(cost)
    return np.sqrt(cost) if use_sqrt else cost


def tfdc(
    copula_hist: np.ndarray,
    target_copulas: list[np.ndarray],
    forget_copulas: list[np.ndarray],
    M: np.ndarray | None = None,
    reg: float = 1e-2,
    p: int = 2,
    method: Literal["sinkhorn", "wasserstein"] = "sinkhorn",
) -> float:
    """
    Target/Forget Dependence Coefficient as in Marti et al. (2016), Eq. (5).

    Parameters
    ----------
    copula_hist : ndarray
        Empirical copula histogram of the pair (same shape as reference copulas).
    target_copulas, forget_copulas : list of ndarray
        Reference copulas (histograms).
    M : ndarray or None
        Ground cost matrix. If None, built from n_bins and p.
    reg : float
        Entropic regularization parameter (only used if method="sinkhorn").
    p : int
        Order of ground metric (for building M if None).
    method : {"sinkhorn", "wasserstein"}
        Which OT distance to use.
    """
    if not target_copulas:
        raise ValueError("target_copulas must be a non-empty list.")
    if not forget_copulas:
        raise ValueError("forget_copulas must be a non-empty list.")

    # Build M once if needed
    if M is None:
        n_bins2 = copula_hist.size
        n_bins = int(np.sqrt(n_bins2))
        if n_bins * n_bins != n_bins2:
            raise ValueError("Cannot infer n_bins from copula histogram size.")
        M = grid_cost_matrix(n_bins, p=p)

    # choose distance function
    if method == "sinkhorn":

        def dist(a, b):
            return sinkhorn_distance(a, b, M=M, reg=reg, p=p)

    elif method == "wasserstein":

        def dist(a, b):
            return wasserstein_distance(a, b, M=M, p=p)

    else:
        raise ValueError(f"Unknown OT method '{method}'")

    d_forget = min(dist(copula_hist, C_minus) for C_minus in forget_copulas)
    d_target = min(dist(copula_hist, C_plus) for C_plus in target_copulas)

    denom = d_forget + d_target
    if denom == 0.0:
        # Numerically identical to both a target and a forget copula
        return 0.5
    return float(d_forget / denom)


def tfdc_matrix(
    df,
    target_copulas: list[np.ndarray],
    forget_copulas: list[np.ndarray],
    n_bins: int = 20,
    reg: float = 1e-2,
    p: int = 2,
    method: Literal["sinkhorn", "wasserstein"] = "sinkhorn",
):
    """
    Compute a TFDC-based dependence matrix for all columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame, shape (T, N)
    target_copulas, forget_copulas : parameter sets for TFDC.
    n_bins, reg, p : histogram and OT parameters.

    Returns
    -------
    pandas.DataFrame, shape (N, N)
    """

    n_assets = df.shape[1]
    cols = list(df.columns)

    M = grid_cost_matrix(n_bins, p=p)

    # Precompute 1D copula transforms U_i for each variable
    U = {col: empirical_copula_transform(df[col].values) for col in cols}

    mat = np.zeros((n_assets, n_assets), dtype=float)

    for i, col_i in enumerate(cols):
        u_i = U[col_i]
        for j, col_j in enumerate(cols[i:], start=i):
            # if i == j:
            # mat[i, j] = 1.0
            # continue

            u_j = U[col_j]
            H_ij = empirical_copula_hist(u_i, u_j, n_bins=n_bins)

            val = tfdc(
                H_ij,
                target_copulas=target_copulas,
                forget_copulas=forget_copulas,
                M=M,
                reg=reg,
                p=p,
                method=method,
            )
            mat[i, j] = mat[j, i] = val

    return pd.DataFrame(mat, index=cols, columns=cols)


def _tfdc_pair(
    i: int,
    j: int,
    u_i: np.ndarray,
    u_j: np.ndarray,
    n_bins: int,
    M: np.ndarray,
    target_copulas: list[np.ndarray],
    forget_copulas: list[np.ndarray],
    reg: float,
    p: int,
    method: str,
) -> tuple[int, int, float]:
    """
    Helper to compute TFDC for one pair of assets (i, j).

    Returns
    -------
    (i, j, value) : tuple
        Indices and TFDC value.
    """
    H_ij = empirical_copula_hist(u_i, u_j, n_bins=n_bins)

    val = tfdc(
        H_ij,
        target_copulas=target_copulas,
        forget_copulas=forget_copulas,
        M=M,
        reg=reg,
        p=p,
        method=method,
    )
    return i, j, float(val)


def tfdc_matrix_parallel(
    df,
    target_copulas: list[np.ndarray],
    forget_copulas: list[np.ndarray],
    n_bins: int = 20,
    reg: float = 1e-2,
    p: int = 2,
    method: Literal["sinkhorn", "wasserstein"] = "sinkhorn",
    n_jobs: int = -1,
    verbose: int = 0,
):
    """
    Compute a TFDC-based dependence matrix for all columns of a DataFrame,
    using joblib to parallelise over asset pairs.

    Parameters
    ----------
    df : pandas.DataFrame, shape (T, N)
        Return matrix.
    target_copulas, forget_copulas : list of ndarray
        Reference copulas for TFDC.
    n_bins, reg, p : int or float
        Histogram and OT parameters.
    method : {"sinkhorn", "wasserstein"}
        Which OT distance to use.
    n_jobs : int
        Number of parallel workers (joblib). -1 = use all cores.
    verbose : int
        Verbosity level for joblib (0 = silent).

    Returns
    -------
    pandas.DataFrame, shape (N, N)
        Symmetric TFDC dependence matrix.
    """
    n_assets = df.shape[1]
    cols = list(df.columns)

    # Ground cost matrix
    M = grid_cost_matrix(n_bins, p=p)

    # Precompute 1D copula transforms
    U = {col: empirical_copula_transform(df[col].values) for col in cols}

    # Build list of (i, j) pairs, including diagonal (i == j) to match tfdc_matrix behaviour
    tasks: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for i, col_i in enumerate(cols):
        u_i = U[col_i]
        for j, col_j in enumerate(cols[i:], start=i):
            u_j = U[col_j]
            tasks.append((i, j, u_i, u_j))

    # Parallel computation over all pairs
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_tfdc_pair)(
            i,
            j,
            u_i,
            u_j,
            n_bins,
            M,
            target_copulas,
            forget_copulas,
            reg,
            p,
            method,
        )
        for (i, j, u_i, u_j) in tasks
    )

    # Fill matrix
    mat = np.zeros((n_assets, n_assets), dtype=float)
    for i, j, val in results:
        mat[i, j] = val
        mat[j, i] = val  # symmetry

    return pd.DataFrame(mat, index=cols, columns=cols)
