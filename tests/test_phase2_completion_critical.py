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


def test_wavelength_constant_multiplier_applies_to_penalized_objective():
    C = np.array([[1.0], [1.0]])
    Y = np.array([[-2.0], [-2.0]])
    # For .5*sum((2*(A-Y))**2) + .5*A**2 the unique minimizer is -16/9.
    result = solve_A_nnls_pgd(C, Y, mu=1.0, weights_row=np.array([2.0]))
    np.testing.assert_allclose(result, [[-16.0 / 9.0]], rtol=1e-9, atol=1e-10)
