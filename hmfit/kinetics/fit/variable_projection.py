"""Variable projection utilities for linear observation models."""

from __future__ import annotations

import numpy as np

from ..observation.linear_matrix import solve_linear_matrix


def solve_A_ls(
    C: np.ndarray,
    D: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    nnls: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for A in D ~= C @ A. Returns (A, D_hat)."""
    return solve_linear_matrix(C, D, weights=weights, nnls=nnls)
