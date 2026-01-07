"""Observation model for spectroscopy data."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .linear_matrix import solve_linear_matrix


def fit_spectroscopy(
    C: np.ndarray,
    D: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    nnls: bool = False,
    channels: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit linear coefficients for spectroscopy channels."""
    D_sel = D
    weights_sel = weights
    if channels is not None:
        D_sel = D[:, channels]
        weights_sel = _select_weights(weights, channels)

    return solve_linear_matrix(C, D_sel, weights=weights_sel, nnls=nnls)


def _select_weights(
    weights: np.ndarray | None, channels: Sequence[int]
) -> np.ndarray | None:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        return w
    if w.ndim == 2:
        if w.shape[1] == 1:
            return w
        return w[:, channels]
    raise ValueError("Weights must be 1D or 2D.")
