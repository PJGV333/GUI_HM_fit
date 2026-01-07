"""Linear observation model using least squares projection."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import nnls


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
        return np.linalg.lstsq(C, D, rcond=None)[0]

    if weights.shape[1] == 1:
        w = weights[:, 0]
        Cw = C * w[:, None]
        Dw = D * w[:, None]
        return np.linalg.lstsq(Cw, Dw, rcond=None)[0]

    A = np.zeros((C.shape[1], D.shape[1]), dtype=float)
    for j in range(D.shape[1]):
        w = weights[:, j]
        Cw = C * w[:, None]
        Dw = D[:, j] * w
        A[:, j] = np.linalg.lstsq(Cw, Dw, rcond=None)[0]
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
