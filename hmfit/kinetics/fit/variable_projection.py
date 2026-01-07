"""Variable projection utilities for linear observation models."""

from __future__ import annotations

import numpy as np

from ..observation.linear_matrix import solve_linear_matrix


def solve_A_ls(
    C: np.ndarray,
    D: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for A in D ~= C @ A using least squares. Returns (A, D_hat)."""
    return solve_linear_matrix(C, D, weights=weights, nnls=False)


def solve_A_nnls(
    C: np.ndarray,
    D: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for A in D ~= C @ A using NNLS per channel. Returns (A, D_hat)."""
    return solve_linear_matrix(C, D, weights=weights, nnls=True)
