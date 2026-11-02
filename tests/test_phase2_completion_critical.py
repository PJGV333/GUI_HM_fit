"""Regressions for critical uncertainty and weighted-solver defects."""
import numpy as np
import pytest
from typing import Any

from hmfit_core.acid_base_errors import compute_errors_acid_base
from hmfit_core.utils import errors
from hmfit_core.utils.nnls_utils import solve_A_nnls_pgd


def _refit_context() -> Any:
    from types import SimpleNamespace
    ctx = {"C_T": [[0.001, 0.0002], [0.001, 0.0008], [0.001, 0.002]],
           "modelo_solver": [[1, 0, 1], [0, 1, 1]], "weights": [1.0, 1.0],
           "non_abs_species": [], "algorithm": "Newton-Raphson", "model_settings": "Free",
           "optimizer": "powell", "bounds": [(0.0, 6.0)]}
    return SimpleNamespace(_last_fit_context=ctx, _parse_float=lambda x, default: float(x))


def test_refit_rejects_optimizer_nonconvergence(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **kw:
                        SimpleNamespace(x=np.array([3.0]), success=False, nit=1, message="budget exceeded"))
    _, ok, info = SpectroscopyTab._refit_from_data(_refit_context(), np.ones((2, 3)), [3.0])
    assert not ok
    assert info.get("failure_reason") == "optimization_failed"


def test_refit_rejects_inner_linear_solver_failure(monkeypatch):
    from hmfit_core.processors import spectroscopy_processor as processor
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    def fail(*args, **kwargs):
        raise RuntimeError("injected linear failure")
    monkeypatch.setattr(processor, "_solve_spectral_model", fail)
    _, ok, info = SpectroscopyTab._refit_from_data(_refit_context(), np.ones((2, 3)), [3.0])
    assert not ok
    assert info.get("failure_reason") == "linear_solver_failed"


def test_refit_classifies_out_of_bounds_result(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **kw:
                        SimpleNamespace(x=np.array([7.0]), success=True, nit=1))
    _, ok, info = SpectroscopyTab._refit_from_data(_refit_context(), np.ones((2, 3)), [3.0])
    assert not ok
    assert info.get("failure_reason") == "bounds_violated"


def test_refit_classifies_chemical_solver_failure(monkeypatch):
    from hmfit_core.solvers import NewtonRaphson
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    def fail(*args, **kwargs):
        raise RuntimeError("chemical model did not converge")
    monkeypatch.setattr(NewtonRaphson, "concentraciones", fail)
    _, ok, info = SpectroscopyTab._refit_from_data(_refit_context(), np.ones((2, 3)), [3.0])
    assert not ok
    assert info.get("failure_reason") == "chemical_solver_failed"


def test_refit_propagates_bootstrap_cancellation(monkeypatch):
    from hmfit_core.processors import spectroscopy_processor as processor
    from hmfit_core.utils.errors import BootstrapCancelled
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    def cancel(*args, **kwargs):
        raise BootstrapCancelled("cancelled")
    monkeypatch.setattr(processor, "_solve_spectral_model", cancel)
    with pytest.raises(BootstrapCancelled):
        SpectroscopyTab._refit_from_data(_refit_context(), np.ones((2, 3)), [3.0])


def test_full_refit_counts_failures_and_rejects_nonfinite_success():
    calls = iter([
        (np.array([3.1]), True, {}),
        (np.array([3.0]), False, {"failure_reason": "linear_solver_failed"}),
        (np.array([np.nan]), True, {}),
        (np.array([3.0]), False, {"failure_reason": "linear_solver_failed"}),
    ])
    out = errors.bootstrap_full_refit([3.0], lambda rng, wild: None,
                                     lambda *a, **k: next(calls), B=4, seed=1)
    assert out["n_attempted"] == 4
    assert out["n_success"] == 1
    assert out["n_fail"] == 3
    assert out["failure_reasons"] == {"linear_solver_failed": 2, "nonfinite_result": 1}


def test_all_fixed_acid_base_correlations_are_undefined():
    with pytest.warns(UserWarning, match="fixed"):
        result = compute_errors_acid_base(
            lambda theta: np.array([theta[0] - 1, theta[1] - 2, 0.1]),
            [1.0, 2.0], ["pKa1", "pKa2"], fixed_mask=[True, True],
        )
    np.testing.assert_array_equal(result["se"], [0.0, 0.0])
    assert np.isnan(result["correlation_matrix"]).all()


def test_nmr_one_step_identifiability_uses_original_jacobian(monkeypatch):
    calls = []

    def fake_metrics(k, res, dq, D_cols, modelo, nas, **kwargs):
        # Original fit has no sensitivity. Every bootstrap realization does.
        calls.append(np.array(dq, copy=True))
        J = np.zeros((1, 4)) if len(calls) == 1 else np.ones((1, 4))
        return {"J": J, "r": np.zeros(4), "n_linear": 0}

    monkeypatch.setattr(errors, "compute_errors_nmr_varpro", fake_metrics)
    result = errors.bootstrap_one_step_nmr(
        [3.0], object(), np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([[1.1], [2.1], [3.1], [4.1]]), None, None, [],
        B=3, seed=7, leverage_correction=False,
    )
    assert len(calls) == 4
    assert np.isnan(result["samples"]).all()


def test_absolute_unregularized_wavelengths_do_not_build_dense_global_system():
    from hmfit_core.processors.spectroscopy_processor import _solve_absorptivities
    rng = np.random.default_rng(44)
    C = rng.uniform(0.5, 1.5, size=(6, 3))
    truth = rng.uniform(1.0, 2.0, size=(3, 1400))
    Y = C @ truth
    W = np.tile(np.linspace(0.5, 1.5, Y.shape[1]), (Y.shape[0], 1))
    A, _ = _solve_absorptivities(C, Y, mu=0.01, weights_row=W,
                                  weight_regularization_mode="absolute")
    np.testing.assert_allclose(A, truth, rtol=1e-7, atol=1e-7)


def test_absolute_weighted_smoothing_matches_independent_optimum():
    from scipy.optimize import minimize
    from hmfit_core.processors.spectroscopy_processor import _solve_absorptivities

    rng = np.random.default_rng(31)
    C = rng.normal(size=(8, 2))
    Y = rng.normal(size=(8, 3))
    W = np.broadcast_to(np.array([0.4, 1.2, 2.0]), Y.shape)
    D = np.diff(np.eye(3), axis=0)
    L = D.T @ D
    mu, alpha = 0.4, 0.7

    def objective_gradient(flat):
        A = flat.reshape(2, 3)
        R = C @ A - Y
        neg = np.minimum(A, 0.0)
        f = 0.5 * np.sum((W * R) ** 2) + 0.5 * mu * np.sum(neg ** 2)
        f += 0.5 * alpha * np.sum((A @ L) * A)
        g = C.T @ (W ** 2 * R) + mu * neg + alpha * A @ L
        return f, g.ravel()

    ref = minimize(objective_gradient, np.zeros(6), jac=True, method="BFGS", options={"gtol": 1e-10})
    A, _ = _solve_absorptivities(C, Y, mu=mu, alpha_smooth=alpha, smooth_matrix=L,
                                  weights_row=W, weight_regularization_mode="absolute")
    f, g = objective_gradient(np.asarray(A).ravel())
    assert f <= ref.fun + 1e-9
    assert np.linalg.norm(g, ord=np.inf) < 1e-8
    np.testing.assert_allclose(A, ref.x.reshape(2, 3), rtol=1e-7, atol=1e-8)
    legacy, _ = _solve_absorptivities(C, Y, mu=mu, alpha_smooth=alpha, smooth_matrix=L, weights_row=W)
    assert np.linalg.norm(A - legacy) > 1e-3


@pytest.mark.parametrize("weight_kind", ["uniform", "lambda", "point", "zeros"])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_absolute_mode_kkt_and_reference(weight_kind, bounded, smooth):
    from scipy.optimize import minimize
    from hmfit_core.processors.spectroscopy_processor import _solve_absorptivities

    rng = np.random.default_rng(902)
    C = rng.normal(size=(8, 2))
    Y = rng.normal(size=(8, 4))
    W = np.ones_like(Y)
    if weight_kind == "lambda":
        W *= np.array([0.4, 0.8, 1.5, 2.0])
    elif weight_kind in {"point", "zeros"}:
        W = rng.uniform(0.2, 2.0, size=Y.shape)
        if weight_kind == "zeros":
            W[2, 1] = 0.0
            W[4, 2] = 0.0
    D = np.diff(np.eye(4), axis=0)
    L = D.T @ D
    mu = 0.4
    alpha = 0.3 if smooth else 0.0
    A, lb = _solve_absorptivities(C, Y, mu=mu, weights_row=W,
        delta_mode="relative" if bounded else "off", delta_rel=0.15,
        alpha_smooth=alpha, smooth_matrix=L, weight_regularization_mode="absolute")
    A = np.asarray(A)
    bounds = None if lb is None else [(float(lb[i]), None) for i in range(2) for _ in range(4)]

    def objective_gradient(x):
        x = x.reshape(2, 4)
        residual = C @ x - Y
        negative = np.minimum(x, 0)
        value = 0.5 * (np.sum((W * residual) ** 2) + mu * np.sum(negative ** 2)
                       + alpha * np.sum((x @ L) * x))
        gradient = C.T @ (W ** 2 * residual) + mu * negative + alpha * x @ L
        return value, gradient.ravel()

    ref = minimize(objective_gradient, np.zeros(8), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10})
    assert ref.success, ref.message
    value, gradient = objective_gradient(A.ravel())
    assert value <= ref.fun + 1e-9
    grad = gradient.reshape(A.shape)
    if lb is not None:
        lower = np.asarray(lb).reshape(-1, 1)
        assert np.all(A >= lower - 1e-10)
        grad = np.where((A <= lower + 1e-9) & (grad > 0), 0.0, grad)
    assert np.linalg.norm(grad.ravel(), ord=np.inf) < 1e-8


def test_absolute_mode_optimizer_objective_uses_penalized_projection():
    from hmfit_core.processors.spectroscopy_processor import (
        _evaluate_spectro_objective, _new_objective_state, _solve_spectral_model,
        _spectral_profile_objective,
    )
    rng = np.random.default_rng(91)
    C = rng.normal(size=(6, 2))
    Y = rng.normal(size=(6, 3))
    W = np.array([[0.5, 1.0, 2.0]])
    D = np.diff(np.eye(3), axis=0)
    L = D.T @ D

    class Solver:
        n_constants = 1
        def concentraciones(self, k):
            return (C,)

    fit = _solve_spectral_model(C, Y, mu=0.4, weights_row=W, alpha_smooth=0.2,
                                smooth_matrix=L, weight_regularization_mode="absolute")
    expected = _spectral_profile_objective(fit, Y, W, 0.4, 0.2, L, "absolute")
    actual = _evaluate_spectro_objective(
        [3.0], solver=Solver(), y_transposed=Y, weights_row=W,
        eps_solver_mode="soft_penalty", eps_mu=0.4, delta_mode="off", delta_rel=0.01,
        alpha_smooth=0.2, smooth_matrix=L, objective_state=_new_objective_state([3.0]),
        weight_regularization_mode="absolute",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_absolute_profile_objective_includes_both_penalties():
    from hmfit_core.processors.spectroscopy_processor import _spectral_profile_objective

    A = np.array([[1.0, -2.0, 3.0]])
    Y = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    yfit = np.zeros_like(Y)
    W = np.array([[1.0, 2.0, 0.5]])
    D = np.diff(np.eye(3), axis=0)
    L = D.T @ D
    mu, alpha = 0.4, 0.7
    expected = np.sqrt((np.sum((W * (yfit - Y)) ** 2)
                        + mu * np.sum(np.minimum(A, 0.0) ** 2)
                        + alpha * np.sum((A @ L) * A)) / Y.size)
    value = _spectral_profile_objective({"yfit": yfit, "E_group": A}, Y, W,
                                        mu, alpha, L, "absolute")
    np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_absolute_mode_propagates_through_spectroscopy_config(tmp_path):
    from hmfit_core import run_spectroscopy
    from hmfit_core.processors.spectroscopy_processor import _solve_absorptivities, _build_smoothness_laplacian
    from hmfit_core.solvers import NewtonRaphson
    from tests.test_phase2_weights import _cfg, _quiet, _spectro_fixture

    path, Y = _spectro_fixture(tmp_path, n_points=8, n_lam=9, seed=22)
    W = np.tile(np.linspace(0.5, 2.0, 9), (8, 1))
    result = _quiet(run_spectroscopy, _cfg(path, {
        "weights_matrix": W, "weight_regularization_mode": "absolute",
        "skip_optimization": True, "preset_k": [3.0], "alpha_smooth": 1e-6,
    }))["export_data"]
    c = np.column_stack([np.full(8, 1e-3), np.linspace(0, 2e-3, 8)])
    model = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    C, _ = NewtonRaphson(c, model, nas=[], model_sett="Free").concentraciones([3.0])
    ref, _ = _solve_absorptivities(C, Y, mu=0.01, alpha_smooth=1e-6,
                                    smooth_matrix=_build_smoothness_laplacian(9),
                                    weights_row=np.sqrt(W), weight_regularization_mode="absolute")
    np.testing.assert_allclose(np.asarray(result["A_solver"]), np.asarray(ref).T, rtol=1e-8, atol=1e-8)
    assert result["weight_regularization_mode"] == "absolute"


def test_absolute_mode_full_refit_matches_fitted_model(tmp_path):
    from types import SimpleNamespace
    from hmfit_core import run_spectroscopy
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab
    from tests.test_phase2_weights import _cfg, _quiet, _spectro_fixture

    path, Y = _spectro_fixture(tmp_path, n_points=8, n_lam=9, seed=4)
    W = np.tile(np.linspace(0.5, 2.0, 9), (8, 1))
    out = _quiet(run_spectroscopy, _cfg(path, {
        "weights_matrix": W, "weight_regularization_mode": "absolute",
        "alpha_smooth": 1e-6,
    }))["export_data"]
    ctx = {"C_T": out["C_T"], "modelo_solver": out["modelo"], "weights": out["weights"],
           "non_abs_species": [], "algorithm": "Newton-Raphson", "model_settings": "Free",
           "optimizer": "powell", "bounds": [(0.0, 6.0)], "eps_solver_mode": "soft_penalty",
           "weight_regularization_mode": "absolute", "mu": 0.01, "alpha_smooth": 1e-6}
    tab: Any = SimpleNamespace(_last_fit_context=ctx, _parse_float=lambda x, default: float(x))
    refit_k, success, details = SpectroscopyTab._refit_from_data(tab, np.asarray(out["Y"]),
                                                                 np.asarray(out["k"]), max_iter=150, tol=1e-9)
    assert success, details
    np.testing.assert_allclose(refit_k, out["k"], atol=2e-5, rtol=0)


def test_wavelength_constant_multiplier_applies_to_penalized_objective():
    C = np.array([[1.0], [1.0]])
    Y = np.array([[-2.0], [-2.0]])
    # For .5*sum((2*(A-Y))**2) + .5*A**2 the unique minimizer is -16/9.
    result = solve_A_nnls_pgd(C, Y, mu=1.0, weights_row=np.array([2.0]))
    np.testing.assert_allclose(result, [[-16.0 / 9.0]], rtol=1e-9, atol=1e-10)
