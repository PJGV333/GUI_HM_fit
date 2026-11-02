"""Phase 2 — Stage 1: per-point weights (m x n_lambda) in the spectroscopy
linear estimator must minimize the same weighted residual as the fit.

Covers (docs/STABILITY_PHASE1_REPORT.md section 8, limitation 5):
- uniform weights reproduce the previous (unweighted) results exactly;
- per-wavelength weights reproduce them as well (positive column scaling cancels);
- per-point weights match an independent column-by-column weighted reference
  (least squares, NNLS and soft-penalty active set) and the finite-difference
  Jacobian of the weighted profiled residual;
- degrees of freedom and observation counts are consistent with zero weights.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize, minimize_scalar, nnls as scipy_nnls
from hmfit_core.utils.nnls_utils import solve_A_nnls_pgd

from hmfit_core.solvers import NewtonRaphson
from hmfit_core.utils.errors import (
    _jac_varpro_weighted,
    _weighted_column_lstsq,
    compute_errors_spectro_varpro,
)
from hmfit_core.processors.spectroscopy_processor import (
    _exact_nnls_columns,
    _exact_soft_penalty_columns,
    _solve_spectral_model,
)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --------------------------------------------------------------------------- fixtures
def _small_system(seed=7, n_points=8, n_lam=5):
    """1:1 binding toy system: C(k) (m x 2 absorbing species), Y (n_lam x m)."""
    rng = np.random.default_rng(seed)
    H = np.full(n_points, 1e-3)
    G = np.linspace(2e-4, 2e-3, n_points)
    M = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float).T  # NewtonRaphson expects (n_comp x n_species)
    solver = NewtonRaphson(np.column_stack([H, G]), M, nas=[], model_sett="Free")
    k = np.array([3.0])
    C, Co = solver.concentraciones(k)  # C: absorbing species (m x n_abs)
    eps = rng.uniform(200.0, 1500.0, size=(C.shape[1], n_lam))
    Y_lam_m = C @ eps  # (m x n_lam)
    return solver, C, k, Y_lam_m.T.copy()  # Y as errors expects: (n_lam x m)


def _weighted_profile_residual(solver, k, Y_lam_m, W):
    """Independent weighted profiled residual: per-column weighted lstsq."""
    C, _ = solver.concentraciones(np.asarray(k, float))
    YT = Y_lam_m.T  # (m x n_lam)
    m, nw = YT.shape
    R = np.zeros_like(YT)
    for j in range(nw):
        f = W[:, j]
        valid = f > 0
        Cs = C[valid] * f[valid, None]
        ys = YT[valid, j] * f[valid]
        a = np.linalg.lstsq(Cs, ys, rcond=None)[0]
        R[:, j] = C @ a - YT[:, j]
    return (R * W).reshape(-1)


# --------------------------------------------------------------------------- weighted helpers
def test_weighted_column_lstsq_matches_independent_reference():
    rng = np.random.default_rng(11)
    C = rng.normal(0.0, 1.0, size=(9, 3))
    Y = rng.normal(0.0, 1.0, size=(9, 4))
    W = rng.uniform(0.1, 3.0, size=(9, 4))
    A, n_lin = _weighted_column_lstsq(C, Y, W, rcond=1e-12)
    assert A.shape == (3, 4)
    for j in range(4):
        f = W[:, j]
        ref = np.linalg.lstsq(C * f[:, None], Y[:, j] * f, rcond=None)[0]
        np.testing.assert_allclose(A[:, j], ref, rtol=1e-12, atol=1e-14)
        assert n_lin[j] == np.linalg.matrix_rank(C * f[:, None], tol=None)
    # Optimality: the weighted solution beats alternative columns.
    for j in range(4):
        obj = np.sum((W[:, j] * (C @ A[:, j] - Y[:, j])) ** 2)
        for _ in range(3):
            alt = A[:, j] + rng.normal(0.0, 0.2, size=3)
            assert np.sum((W[:, j] * (C @ alt - Y[:, j])) ** 2) >= obj - 1e-12


def test_weighted_helpers_work_when_backend_arrays_are_immutable(monkeypatch):
    import hmfit_core.utils.errors as errors

    class ImmutableBackend:
        def __getattr__(self, name):
            return getattr(np, name)

        def zeros(self, *args, **kwargs):
            out = np.zeros(*args, **kwargs)
            out.flags.writeable = False
            return out

    monkeypatch.setattr(errors, "np", ImmutableBackend())
    C = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    W = np.array([[1.0], [0.5], [2.0]])
    Y = np.array([[1.0], [0.2], [2.0]])
    A, _ = errors._weighted_column_lstsq(C, Y, W)
    J = errors._jac_varpro_weighted(C, A, np.ones((3, 2, 1)), W)
    assert np.isfinite(A).all() and np.isfinite(J).all()


def test_uniform_weights_reproduce_current_results():
    solver, C, k, Y = _small_system()
    m = int(Y.shape[1])
    nw = int(Y.shape[0])
    base = compute_errors_spectro_varpro(k, solver, Y, [[1, 0], [0, 1], [1, 1]], [])
    ones = compute_errors_spectro_varpro(k, solver, Y, [[1, 0], [0, 1], [1, 1]], [],
                                         weights=np.ones((m, nw)))
    for key in ("SE_log10K", "s2", "dof", "nobs", "n_linear"):
        np.testing.assert_allclose(np.asarray(ones[key], float),
                                   np.asarray(base[key], float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ones["A"], base["A"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ones["J"], base["J"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ones["yfit"], base["yfit"], rtol=1e-12, atol=1e-15)


def test_per_lambda_weights_match_broadcast_matrix_and_vector():
    rng = np.random.default_rng(3)
    solver, C, k, Y = _small_system()
    m, nw = int(Y.shape[1]), int(Y.shape[0])
    w_lam = rng.uniform(0.3, 2.0, size=nw)
    vec = compute_errors_spectro_varpro(k, solver, Y, [[1, 0], [0, 1], [1, 1]], [], weights=w_lam)
    mat = compute_errors_spectro_varpro(k, solver, Y, [[1, 0], [0, 1], [1, 1]], [],
                                        weights=np.tile(w_lam, (m, 1)))
    for key in ("SE_log10K", "s2", "dof", "nobs", "n_linear"):
        np.testing.assert_allclose(np.asarray(mat[key], float),
                                   np.asarray(vec[key], float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(mat["A"], vec["A"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(mat["J"], vec["J"], rtol=1e-12, atol=1e-15)
    # A per-lambda weight profile does not change the linear coefficients of the
    # (unconstrained) projection: each column is rescaled by a positive constant.
    base = compute_errors_spectro_varpro(k, solver, Y, [[1, 0], [0, 1], [1, 1]], [])
    np.testing.assert_allclose(vec["A"], base["A"], rtol=1e-10, atol=1e-14)


def test_per_point_weights_jacobian_matches_finite_differences():
    rng = np.random.default_rng(17)
    m, nw, s = 8, 5, 3
    C = rng.normal(size=(m, s))
    dC = rng.normal(size=(m, s))
    Y = rng.normal(size=(m, nw))
    W = rng.uniform(0.2, 2.5, size=(m, nw))
    W[2, 0] = 0.0
    A, _ = _weighted_column_lstsq(C, Y, W)
    J = _jac_varpro_weighted(C, A, dC[:, :, None], W)

    def residual(theta):
        design = C + theta * dC
        result = np.zeros_like(Y)
        for j in range(nw):
            design_w = W[:, j, None] * design
            response_w = W[:, j] * Y[:, j]
            coeff = np.linalg.lstsq(design_w, response_w, rcond=None)[0]
            result[:, j] = design_w @ coeff - response_w
        return result

    h = 1e-6
    fd = (residual(h) - residual(-h)) / (2 * h)
    # The profiled derivative includes a residual-dependent term in col(Cw).
    # Its orthogonal component equals the projected VarPro Jacobian.
    for j in range(nw):
        design_w = W[:, j, None] * C
        projector = design_w @ np.linalg.pinv(design_w)
        projected_fd = fd[:, j] - projector @ fd[:, j]
        np.testing.assert_allclose(J[0, j::nw], projected_fd, rtol=1e-7, atol=1e-9)


def test_exact_nnls_columns_weighted_matches_reference():
    rng = np.random.default_rng(23)
    C = rng.uniform(0.0, 1.0, size=(10, 3))
    Y = rng.normal(0.0, 1.0, size=(10, 4))
    Y = Y - Y.min(axis=0) - 0.2  # force several negative unconstrained coefficients
    W = rng.uniform(0.1, 3.0, size=(10, 4))
    A = _exact_nnls_columns(C, Y, weights_row=W)
    for j in range(4):
        f = W[:, j]
        ref, _ = scipy_nnls(C * f[:, None], Y[:, j] * f, maxiter=500)
        np.testing.assert_allclose(A[:, j], ref, rtol=1e-9, atol=1e-12)
        # weighted objective at the solution is optimal over random perturbations
        obj = np.sum((W[:, j] * (C @ A[:, j] - Y[:, j])) ** 2)
        for _ in range(3):
            alt = np.abs(A[:, j] + rng.normal(0.0, 0.1, size=3))
            assert np.sum((W[:, j] * (C @ alt - Y[:, j])) ** 2) >= obj - 1e-10
    # constant per-column weights leave the NNLS solution unchanged
    A_const = _exact_nnls_columns(C, Y, weights_row=np.tile(np.array([0.5, 2.0, 1.5, 0.9]), (10, 1)))
    A_none = _exact_nnls_columns(C, Y)
    np.testing.assert_allclose(A_const, A_none, rtol=1e-9, atol=1e-12)


def test_soft_penalty_weighted_matches_brute_force_active_set():
    rng = np.random.default_rng(29)
    C = rng.uniform(0.2, 1.2, size=(10, 3))
    Y = rng.normal(0.0, 1.0, size=(10, 2))
    W = rng.uniform(0.2, 2.0, size=(10, 2))
    mu = 1e-2
    A = _exact_soft_penalty_columns(C, Y, mu=mu, weights_row=W)

    def weighted_obj(a, j):
        data = np.sum((W[:, j] * (C @ a - Y[:, j])) ** 2)
        pen = mu * np.mean(W[:, j] ** 2) * np.sum(np.minimum(a, 0.0) ** 2)
        return 0.5 * (data + pen)

    s = C.shape[1]
    for j in range(2):
        obj_here = weighted_obj(A[:, j], j)
        best = float("inf")
        for mask in range(1 << s):  # brute force over active sets
            free = [i for i in range(s) if not (mask >> i) & 1]
            act = [i for i in range(s) if (mask >> i) & 1]
            G = C.T @ (C * (W[:, j] ** 2)[:, None])
            B = C.T @ (Y[:, j] * W[:, j] ** 2)
            M = G.copy()
            for i in act:
                M[i, i] += mu * np.mean(W[:, j] ** 2)
            try:
                a = np.linalg.solve(M, B)
            except np.linalg.LinAlgError:
                continue
            consistent = all(a[i] < 0 for i in act) and all(a[i] > 0 for i in free)
            if consistent:
                best = min(best, weighted_obj(a, j))
        assert np.isfinite(best), "independent enumeration must find a valid active set"
        np.testing.assert_allclose(obj_here, best, rtol=1e-10, atol=1e-10)
    # constant per-column weights leave the soft-penalty solution unchanged
    A_const = _exact_soft_penalty_columns(C, Y, mu=mu, weights_row=np.full((10, 2), 0.7))
    A_none = _exact_soft_penalty_columns(C, Y, mu=mu)
    np.testing.assert_allclose(A_const, A_none, rtol=1e-9, atol=1e-12)


def test_weighted_objective_does_not_swallow_solver_failure(monkeypatch):
    from hmfit_core.processors import spectroscopy_processor as processor
    from hmfit_core.utils.nnls_utils import WeightedLinearSolveError

    class Solver:
        n_constants = 1
        def concentraciones(self, _k):
            return (np.ones((4, 1)),)

    def fail(*args, **kwargs):
        raise WeightedLinearSolveError("not converged")

    monkeypatch.setattr(processor, "_solve_spectral_model", fail)
    state = {"cache": {}, "cache_max": 10, "eval_count": 0, "error_count": 0,
             "best_rms": np.inf, "best_k": None}
    with pytest.raises(WeightedLinearSolveError, match="not converged"):
        processor._evaluate_spectro_objective(
            [3.0], solver=Solver(), y_transposed=np.ones((4, 2)),
            weights_row=np.ones((4, 2)), eps_solver_mode="soft_bound", eps_mu=0.01,
            delta_mode="relative", delta_rel=0.01, alpha_smooth=0.0,
            smooth_matrix=None, objective_state=state,
        )


@pytest.mark.parametrize("bounded,smooth", [(False, False), (True, False), (True, True)])
def test_weighted_penalized_solver_reaches_independent_optimum(bounded, smooth):
    rng = np.random.default_rng(42)
    C = rng.uniform(0, 0.001, (12, 3))
    Y = rng.normal(0, 1, (12, 4))
    W = rng.uniform(0.2, 2.0, (12, 4))
    mu = 0.01
    alpha = 1e-6 if smooth else 0.0
    D = np.diff(np.eye(4), axis=0)
    L = D.T @ D
    lower = np.full((3, 1), -10.0) if bounded else None

    def objective_and_grad(flat):
        A = flat.reshape(3, 4)
        residual = C @ A - Y
        neg = np.minimum(A, 0)
        value = 0.5 * np.sum((W * residual) ** 2) + 0.5 * mu * np.sum(neg ** 2)
        grad = C.T @ (W ** 2 * residual) + mu * neg
        if smooth:
            value += 0.5 * alpha * np.sum((A @ L) * A)
            grad += alpha * A @ L
        return value, grad.ravel()

    reference = minimize(objective_and_grad, np.zeros(12), jac=True, method="L-BFGS-B",
                         bounds=[(-10, None)] * 12 if bounded else None,
                         options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
    assert reference.success, reference.message
    actual = solve_A_nnls_pgd(C, Y, mu=mu, lower_bound=lower, alpha_smooth=alpha,
                              smooth_matrix=L, weights_row=W)
    assert objective_and_grad(np.asarray(actual).ravel())[0] <= reference.fun + 1e-7


# --------------------------------------------------------------------------- processor level
def _spectro_fixture(tmp_path, n_points=10, n_lam=15, noise=2e-4, seed=101):
    wl = np.linspace(250.0, 550.0, n_lam)
    c = np.column_stack([np.full(n_points, 1e-3), np.linspace(0.0, 2e-3, n_points)])
    M = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float).T
    solver = NewtonRaphson(c, M, nas=[], model_sett="Free")
    C, _ = solver.concentraciones(np.array([3.0]))
    centers = np.linspace(280.0, 480.0, C.shape[1])
    eps = np.vstack([1.2e3 * np.exp(-((wl - c0) ** 2) / (2 * 20.0 ** 2)) for c0 in centers])
    y = C @ eps
    if noise > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, y.shape)
    path = os.path.join(str(tmp_path), "s.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(y.T, index=wl, columns=[f"p{i+1}" for i in range(n_points)]).to_excel(
            w, sheet_name="Spectra"
        )
        pd.DataFrame({"H": c[:, 0], "G": c[:, 1]}).to_excel(w, sheet_name="Conc", index=False)
    return path, y  # y: (n_points, n_lambda)


def _cfg(path, extra=None):
    cfg = {
        "file_path": path, "spectra_sheet": "Spectra", "conc_sheet": "Conc",
        "column_names": ["H", "G"], "receptor_label": "H", "guest_label": "G",
        "efa_enabled": False, "efa_eigenvalues": 0,
        "modelo": [[1, 0], [0, 1], [1, 1]], "non_abs_species": [],
        "algorithm": "Newton-Raphson", "model_settings": "Free", "optimizer": "powell",
        "initial_k": [2.5], "bounds": [(0.0, 6.0)], "channels_raw": "All",
        "channels_mode": "all", "channels_resolved": [], "show_stability_diagnostics": False,
        "multi_start_runs": 1, "render_graphs": False,
    }
    if extra:
        cfg.update(extra)
    return cfg


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
def test_processor_rejects_invalid_statistical_weights(tmp_path, invalid):
    from hmfit_core import run_spectroscopy

    path, Y = _spectro_fixture(tmp_path, n_points=8, n_lam=9)
    weights = np.ones_like(Y)
    weights[0, 0] = invalid
    with pytest.raises(ValueError, match="finite nonnegative"):
        _quiet(run_spectroscopy, _cfg(path, {"weights_matrix": weights}))


@pytest.mark.parametrize("options", [
    {"eps_solver_mode": "soft_bound"},
    {"delta_mode": "relative"},
    {"alpha_smooth": 0.1},
])
def test_varying_point_weights_with_regularization_runs(tmp_path, options):
    from hmfit_core import run_spectroscopy

    path, Y = _spectro_fixture(tmp_path, n_points=8, n_lam=9)
    weights = np.ones_like(Y)
    weights[0, :] = 2.0
    result = _quiet(run_spectroscopy, _cfg(path, {"weights_matrix": weights, **options}))
    assert np.all(np.isfinite(result["export_data"]["k"]))
    np.testing.assert_allclose(result["export_data"]["weights"], np.sqrt(weights))


def test_processor_uniform_weights_matrix_matches_default(tmp_path):
    from hmfit_core import run_spectroscopy

    path, Ym = _spectro_fixture(tmp_path)
    base = _quiet(run_spectroscopy, _cfg(path))["export_data"]
    uni = _quiet(run_spectroscopy, _cfg(path, {"weights_matrix": np.ones_like(Ym)}))["export_data"]
    np.testing.assert_allclose(uni["k"], base["k"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(uni["SE_log10K"], float),
                               np.asarray(base["SE_log10K"], float), rtol=1e-12, atol=1e-15)


def test_processor_per_point_weights_match_reference_profile(tmp_path):
    from hmfit_core import run_spectroscopy

    path, Ym = _spectro_fixture(tmp_path, n_points=8, n_lam=9, noise=3e-4, seed=5)
    rng = np.random.default_rng(9)
    W = rng.uniform(0.2, 2.0, size=Ym.shape)
    # Public weights_matrix contains statistical weights, not residual multipliers.
    F = np.sqrt(W)

    M = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float).T
    H = np.full(8, 1e-3)
    Gt = np.linspace(0.0, 2e-3, 8)
    solver = NewtonRaphson(np.column_stack([H, Gt]), M, nas=[], model_sett="Free")

    # same estimator the processor uses by default (eps_solver_mode='soft_penalty', mu=1e-2)
    def profile_rms(logK):
        C, _ = solver.concentraciones(np.array([logK]))
        A = _exact_soft_penalty_columns(C, Ym, mu=1e-2, weights_row=F)
        return float(np.sqrt(np.mean((F * (C @ A - Ym)) ** 2)))

    ref = minimize_scalar(profile_rms, bracket=(2.0, 2.6), method="brent",
                          options={"xtol": 1e-8})
    result = _quiet(run_spectroscopy, _cfg(path, {"weights_matrix": W}))
    out = result["export_data"]
    np.testing.assert_allclose(np.asarray(out["weights"], float), F)
    est = float(np.asarray(out["k"], float)[0])
    # the processor must find the same weighted optimum (within optimizer tolerance)
    assert profile_rms(est) <= profile_rms(float(ref.x)) + 1e-14
    assert abs(est - float(ref.x)) < 1e-5
    # and the weighted optimum differs from the unweighted fit (weights are not ignored)
    base = _quiet(run_spectroscopy, _cfg(path))["export_data"]
    assert abs(est - float(np.asarray(base["k"], float)[0])) > 1e-4

    # SE consistency with an independent finite-difference weighted Jacobian.
    se = float(np.asarray(out["SE_log10K"], float)[0])
    Ck, _ = solver.concentraciones(np.array([est]))
    YT = Ym
    h = 1e-6

    def resid(logK):
        # SE reference for the ANALYTIC method: the unconstrained weighted
        # projection (the analytic covariance is derived for it; the soft-penalty
        # fit optimum is used only as the evaluation point k-hat).
        Ck2, _ = solver.concentraciones(np.array([logK]))
        A2, _ = _weighted_column_lstsq(Ck2, YT, F, rcond=1e-10)
        return (F * (Ck2 @ A2 - YT)).reshape(-1)

    r0 = resid(est)
    d = resid(est + h) - resid(est - h)
    d /= 2 * h
    sse = float(r0 @ r0)
    dof = int(np.count_nonzero(W > 0)) - 1 - int(W.shape[1] * Ck.shape[1])
    s2 = sse / dof
    se_ref = float(np.sqrt(s2 / float(d @ d)))
    assert abs(se - se_ref) / se_ref < 1e-3


def test_gui_full_refit_retains_per_point_weights(tmp_path):
    from types import SimpleNamespace
    from hmfit_core import run_spectroscopy
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    path, Y = _spectro_fixture(tmp_path, n_points=8, n_lam=9, seed=5)
    W = np.random.default_rng(9).uniform(0.2, 2.0, size=Y.shape)
    out = _quiet(run_spectroscopy, _cfg(path, {"weights_matrix": W}))["export_data"]
    context = {
        "C_T": out["C_T"], "modelo_solver": out["modelo"], "weights": out["weights"],
        "non_abs_species": [], "algorithm": "Newton-Raphson", "model_settings": "Free",
        "optimizer": "powell", "bounds": [(0.0, 6.0)], "eps_solver_mode": "soft_penalty",
    }
    fake_tab = SimpleNamespace(_last_fit_context=context, _parse_float=lambda x, default: float(x))
    refit_k, success, details = SpectroscopyTab._refit_from_data(
        fake_tab, np.asarray(out["Y"]), np.asarray(out["k"]), max_iter=100, tol=1e-9
    )
    assert success, details
    np.testing.assert_allclose(refit_k, out["k"], atol=2e-5, rtol=0)


def test_zero_weights_drop_observations_and_dof(tmp_path):
    from hmfit_core import run_spectroscopy

    path, Ym = _spectro_fixture(tmp_path, n_points=10, n_lam=9, noise=3e-4, seed=12)
    W = np.ones_like(Ym)
    W[3, :] = 0.0   # one titration point fully excluded
    W[7, 2:5] = 0.0  # three observations (point, wavelength) excluded
    out = _quiet(run_spectroscopy, _cfg(path, {"weights_matrix": W}))["export_data"]
    n_expected = W.shape[0] * W.shape[1] - W.shape[1] - 3

    # direct metrics check (same estimator as the processor uses at the end of the fit)
    M = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float).T
    H = np.full(10, 1e-3)
    Gt = np.linspace(0.0, 2e-3, 10)
    solver = NewtonRaphson(np.column_stack([H, Gt]), M, nas=[], model_sett="Free")
    k = np.asarray(out["k"], float)
    metrics = compute_errors_spectro_varpro(k, solver, Ym.T, M, [], weights=W)
    # 3 absorbing species (model H,G,HG all absorbing) => rank 3 per wavelength
    assert metrics["nobs"] == int(np.count_nonzero(W > 0)) == n_expected
    assert metrics["dof"] == n_expected - 1 - 9 * 3
