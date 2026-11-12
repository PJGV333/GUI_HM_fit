"""Linear observation model using least squares projection."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import nnls


# Relative singular-value cutoff of the concentration matrix C in the projection D ~= C @ A.
#
# Conservation laws make C structurally rank deficient whenever a mechanism has more dynamic
# species than independent reactions + 1 (e.g. A + B -> C -> D: [B] - [A] and [A] + [C] + [D]
# are constant, so 4 columns span a 3-dimensional space). The integrator preserves those
# invariants only to round-off, which leaves a singular value of ~1e-13 relative to the
# largest one. NumPy's default cutoff (machine epsilon x max(shape) ~ 1e-14) sometimes keeps
# that direction and sometimes drops it, depending on the parameters: the projected
# residual then jumps erratically, its finite-difference Jacobian is noise and the optimiser
# stops at the initial guess. A direction below 1e-10 of the largest singular value is far
# beneath the accuracy of the ODE solution (rtol = 1e-8) and carries no chemical information.
RANK_RCOND = 1.0e-10


def effective_rank(C: np.ndarray, rcond: float = RANK_RCOND) -> int:
    """Numerical rank of C with the same cutoff the projection uses."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.size == 0:
        return 0
    sv = np.linalg.svd(C, compute_uv=False)
    if sv.size == 0 or not np.isfinite(sv[0]) or sv[0] <= 0.0:
        return 0
    return int(np.count_nonzero(sv > float(rcond) * sv[0]))


def prepare_weights(
    weights: np.ndarray | None, n_time: int, n_channels: int
) -> np.ndarray | None:
    """Validate and normalize weights to (n_time, 1) or (n_time, n_channels)."""
    if weights is None:
        return None

    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        if w.shape[0] != n_time:
            raise ValueError("Weights length does not match number of time points.")
        return w.reshape(n_time, 1)
    if w.ndim == 2:
        if w.shape[0] != n_time:
            raise ValueError("Weights rows do not match number of time points.")
        if w.shape[1] not in (1, n_channels):
            raise ValueError("Weights columns must be 1 or match number of channels.")
        return w
    raise ValueError("Weights must be 1D or 2D.")


def solve_linear_matrix(
    C: np.ndarray,
    D: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    nnls: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve D ~= C @ A by least squares (optionally weighted or NNLS)."""
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    if C.ndim != 2 or D.ndim != 2:
        raise ValueError("C and D must be 2D arrays.")
    if C.shape[0] != D.shape[0]:
        raise ValueError("C and D must have the same number of rows.")

    n_time, n_channels = D.shape
    weights = prepare_weights(weights, n_time, n_channels)

    if nnls:
        A = _solve_nnls(C, D, weights)
    else:
        A = _solve_ls(C, D, weights)

    D_hat = C @ A
    return A, D_hat


def _solve_ls(C: np.ndarray, D: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.linalg.lstsq(C, D, rcond=RANK_RCOND)[0]

    if weights.shape[1] == 1:
        w = weights[:, 0]
        Cw = C * w[:, None]
        Dw = D * w[:, None]
        return np.linalg.lstsq(Cw, Dw, rcond=RANK_RCOND)[0]

    A = np.zeros((C.shape[1], D.shape[1]), dtype=float)
    for j in range(D.shape[1]):
        w = weights[:, j]
        Cw = C * w[:, None]
        Dw = D[:, j] * w
        A[:, j] = np.linalg.lstsq(Cw, Dw, rcond=RANK_RCOND)[0]
    return A


def _solve_nnls(C: np.ndarray, D: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    A = np.zeros((C.shape[1], D.shape[1]), dtype=float)
    for j in range(D.shape[1]):
        if weights is None:
            Cw = C
            Dw = D[:, j]
        elif weights.shape[1] == 1:
            w = weights[:, 0]
            Cw = C * w[:, None]
            Dw = D[:, j] * w
        else:
            w = weights[:, j]
            Cw = C * w[:, None]
            Dw = D[:, j] * w
        A[:, j] = nnls(Cw, Dw)[0]
    return A
