"""Tests for variable projection."""

from __future__ import annotations

import numpy as np

from hmfit.kinetics.fit.variable_projection import solve_A_ls


def test_solve_A_ls_recovers_coefficients() -> None:
    rng = np.random.default_rng(0)
    C = rng.random((80, 3))
    A_true = rng.random((3, 4))
    noise = rng.normal(0.0, 1e-6, size=(80, 4))
    D = C @ A_true + noise

    A_est, _ = solve_A_ls(C, D)

    assert np.max(np.abs(A_est - A_true)) < 1e-3


def test_solve_A_ls_nnls_non_negative() -> None:
    rng = np.random.default_rng(1)
    C = rng.random((60, 3))
    A_true = rng.random((3, 2))
    D = C @ A_true

    A_est, _ = solve_A_ls(C, D, nnls=True)

    assert np.all(A_est >= -1e-10)
    assert np.max(np.abs(A_est - A_true)) < 1e-3
