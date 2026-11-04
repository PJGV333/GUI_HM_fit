"""Pinned equivalence between the dense and the sparse (guard-dispatched)
weighted penalized Newton solvers (Stage H).

The sparse path is a structural twin of the dense one: same variable layout,
same semismooth-Newton active set, same projected line search. These tests
pin coefficient equivalence on small systems (where both paths run) and pin
that systems above the dense guard (4000 coefficients) now solve with a
first-order-optimal answer instead of raising.
"""
import time

import numpy as np
import scipy.sparse as sp

from hmfit_core.processors.spectroscopy_processor import _build_smoothness_laplacian
from hmfit_core.utils.nnls_utils import (
    _solve_weighted_penalty_newton,
    _solve_weighted_penalty_newton_sparse,
)


def _problem(seed, s, nw, per_point=True):
    rng = np.random.default_rng(seed)
    C = rng.uniform(1e-4, 2e-3, size=(14, s))
    Y = rng.uniform(-1e-4, 3e-3, size=(14, nw))
    W = (rng.uniform(0.5, 1.5, size=(14, nw)) if per_point
         else np.ones((14, nw)))
    return C, Y, W


def _objective(C, Y, W, A, mu, alpha, L):
    R = W * (C @ A - Y)
    return (0.5 * np.sum(R * R)
            + 0.5 * mu * np.sum(np.minimum(A, 0) ** 2)
            + (0.5 * alpha * np.sum((A @ L) * A) if L is not None else 0.0))


def _kkt_residual(C, Y, W, A, mu, alpha, L):
    s, nw = A.shape
    W2 = W * W
    H = np.zeros((s * nw, s * nw))
    for j in range(nw):
        idx = np.arange(s) * nw + j
        H[np.ix_(idx, idx)] = C.T @ (W2[:, j, None] * C)
    if alpha > 0:
        H += alpha * sp.kron(sp.identity(s, format="csr"),
                             sp.csr_matrix(L), format="csr").toarray()
    b = (C.T @ (W2 * Y)).ravel()
    g = H @ A.ravel() - b + mu * np.minimum(A.ravel(), 0)
    return float(np.max(np.abs(g)))


def test_sparse_matches_dense_small_no_bounds():
    C, Y, W = _problem(11, 3, 30)
    L = _build_smoothness_laplacian(30)
    mu, alpha = 1e-2, 1e-6
    Ad = _solve_weighted_penalty_newton(C, Y, W, 0.0, mu, None, alpha, L, 300)
    As = _solve_weighted_penalty_newton_sparse(C, Y, W, 0.0, mu, None, alpha, L, 300)
    assert np.max(np.abs(Ad - As)) < 1e-7
    fd, fs = _objective(C, Y, W, Ad, mu, alpha, L), _objective(C, Y, W, As, mu, alpha, L)
    assert abs(fd - fs) <= 1e-12 * max(1.0, abs(fd))


def test_sparse_matches_dense_with_lower_bounds_and_active_mu():
    C, Y, W = _problem(12, 3, 24)
    L = _build_smoothness_laplacian(24)
    mu, alpha = 5e-2, 1e-6
    lower = np.full((3, 1), -2e-4)
    Ad = _solve_weighted_penalty_newton(C, Y, W, 0.0, mu, lower, alpha, L, 300)
    As = _solve_weighted_penalty_newton_sparse(C, Y, W, 0.0, mu, lower, alpha, L, 300)
    assert np.max(np.abs(Ad - As)) < 1e-7
    assert np.all(Ad >= lower - 1e-12) and np.all(As >= lower - 1e-12)


def test_guard_exceeded_now_dispatches_to_sparse_and_is_optimal():
    C, Y, W = _problem(13, 3, 1400)          # 4200 coefficients: over the guard
    L = _build_smoothness_laplacian(1400)
    mu, alpha = 1e-2, 1e-6
    t0 = time.perf_counter()
    A = _solve_weighted_penalty_newton(C, Y, W, 0.0, mu, None, alpha, L, 300)
    elapsed = time.perf_counter() - t0
    assert A.shape == (3, 1400)
    assert np.all(np.isfinite(A))
    # projected first-order optimality at the returned iterate
    assert _kkt_residual(C, Y, W, A, mu, alpha, L) < 1e-8
    # and it actually beat a plain least-squares-with-clipping start
    naive = np.maximum(np.linalg.lstsq(C, Y, rcond=None)[0], 0.0)
    assert _objective(C, Y, W, A, mu, alpha, L) < _objective(C, Y, W, naive, mu, alpha, L)
    assert elapsed < 30.0
