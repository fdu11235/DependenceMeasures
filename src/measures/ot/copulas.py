import numpy as np
from .utils import empirical_copula_transform, empirical_copula_hist


def empirical_copula_from_data(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Empirical copula histogram from two 1D sample arrays."""
    u = empirical_copula_transform(x)
    v = empirical_copula_transform(y)
    return empirical_copula_hist(u, v, n_bins=n_bins)


def simulate_reference_copula(
    kind: str = "independence",
    n_samples: int = 50_000,
    n_bins: int = 20,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate a reference copula via simulation and return it as a histogram.

    Parameters
    ----------
    kind : {"independence", "comonotonic", "countermonotonic"}
    n_samples : int
    n_bins : int
    random_state : int or None

    Returns
    -------
    H : ndarray, shape (n_bins, n_bins)
        Empirical copula histogram.
    """
    rng = np.random.default_rng(random_state)
    u = rng.uniform(0.0, 1.0, size=n_samples)

    if kind == "independence":
        v = rng.uniform(0.0, 1.0, size=n_samples)
    elif kind == "comonotonic":
        v = u.copy()
    elif kind == "countermonotonic":
        v = 1.0 - u
    else:
        raise ValueError(f"Unknown copula kind '{kind}'")

    return empirical_copula_hist(u, v, n_bins=n_bins)


def build_standard_correlation_targets(
    n_bins: int = 20,
    n_samples: int = 50_000,
    random_state: int | None = 42,
):
    """
    Reference sets reproducing the paper's 'standard correlation' example.

    Forget set: independence copula.
    Target set: Frechet–Hoeffding bounds (approx positive & negative).:contentReference[oaicite:1]{index=1}

    Returns
    -------
    target_copulas : list of ndarray
    forget_copulas : list of ndarray
    """
    C_indep = simulate_reference_copula(
        "independence",
        n_samples=n_samples,
        n_bins=n_bins,
        random_state=random_state,
    )
    C_pos = simulate_reference_copula(
        "comonotonic",
        n_samples=n_samples,
        n_bins=n_bins,
        random_state=None if random_state is None else random_state + 1,
    )
    C_neg = simulate_reference_copula(
        "countermonotonic",
        n_samples=n_samples,
        n_bins=n_bins,
        random_state=None if random_state is None else random_state + 2,
    )

    target_copulas = [C_pos, C_neg]
    forget_copulas = [C_indep]
    return target_copulas, forget_copulas
