"""Phase 1 — mathematical-integrity invariants (bounds, fixed parameters, dof, identifiability).

Every test documents the defect it guards against. Synthetic data use fixed seeds.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from hmfit.kinetics.data.dataset import KineticsDataset
from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import fit_global, resolve_fixed_parameters
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit_core.acid_base import distribution_fractions_from_pH
from hmfit_core.acid_base_errors import (
    _normalize_bounds,
    _normalize_fixed_mask,
    compute_errors_acid_base,
    compute_errors_acid_base_from_context,
)
from hmfit_core.run_acid_base import run_acid_base
from hmfit_core.solvers import NewtonRaphson
from hmfit_core.utils.errors import (
    bootstrap_linearized_wild,
    bootstrap_one_step_spectro,
    compute_errors_spectro_varpro,
    linearized_covariance,
)


# --------------------------------------------------------------------------------------
# Step 2 — bounds and fixed parameters (kinetics optimizer)
# --------------------------------------------------------------------------------------

def test_resolve_free_fixed_and_invalid_bounds():
    x0 = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 2.0, -np.inf])
    hi = np.array([5.0, 2.0, np.inf])
    x, fixed = resolve_fixed_parameters(x0, lo, hi)
    assert fixed.tolist() == [False, True, False]
    assert x.tolist() == [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="Invalid bounds"):
        resolve_fixed_parameters(x0, np.array([0.0, 3.0, 0.0]), np.array([5.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="NaN"):
        resolve_fixed_parameters(x0, np.array([0.0, np.nan, 0.0]), hi)
    with pytest.raises(ValueError, match="NaN"):
        resolve_fixed_parameters(np.array([1.0, np.nan, 3.0]), lo, hi)
    with pytest.raises(ValueError, match="fixed_mask has 2 entries"):
        resolve_fixed_parameters(x0, lo, hi, fixed_mask=[True, False])
    # empty mask = nothing fixed by mask
    _, fixed = resolve_fixed_parameters(x0, lo, hi, fixed_mask=[])
    assert fixed.tolist() == [False, True, False]
    # mask-fixed with equal bounds that disagree with the value -> explicit error
    with pytest.raises(ValueError, match="bounds force"):
        resolve_fixed_parameters(np.array([1.0, 2.5, 3.0]), lo, hi, fixed_mask=[False, True, False])
    # mask-fixed without equal bounds keeps x0 exactly
    x, fixed = resolve_fixed_parameters(x0, lo, hi, fixed_mask=[True, False, False])
    assert fixed.tolist() == [True, True, False] and x[0] == 1.0


def _first_order_problem(seed=0, noise=1e-2, n_t=30, n_wl=10):
    mech = "species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"
    model = KineticsModel(parse_mechanism(mech))
    t = np.linspace(0.0, 4.0, n_t)
    y0 = {"A": 1.0, "B": 0.0, "C": 0.0}
    A = np.random.default_rng(1).random((3, n_wl))
    D = model.solve_concentrations(t, y0, {"k1": 1.2, "k2": 0.4}) @ A
    D = D + np.random.default_rng(seed).normal(0.0, noise, D.shape)
    return GlobalKineticsObjective(model, [KineticsDataset(t=t, y0=y0, D=D)], param_names=["k1", "k2"])


def test_mask_fixed_kinetic_parameter_keeps_its_exact_value():
    """Defect: a mask-fixed parameter with finite bounds was moved to its lower bound."""
    obj = _first_order_problem()
    res = fit_global(
        obj, np.array([1.0, 0.4]),
        bounds=(np.array([0.0, 0.01]), np.array([10.0, 5.0])), fixed_mask=[False, True],
    )
    assert res.params["k2"] == 0.4
    assert res.errors[1] == 0.0 and res.fixed_mask.tolist() == [False, True]
    assert any("Fixed (not estimated)" in n for n in res.notes)


def test_inverted_bounds_raise_instead_of_fixing():
    obj = _first_order_problem()
    with pytest.raises(ValueError, match="Invalid bounds"):
        fit_global(obj, np.array([1.0, 0.4]), bounds=(np.array([0.0, 0.5]), np.array([10.0, 0.3])))


def test_all_fixed_none_fixed_and_mixed():
    obj = _first_order_problem()
    none = fit_global(obj, np.array([1.0, 0.5]), bounds=(0.0, np.inf))
    assert np.all(none.errors > 0) and not np.any(none.fixed_mask)
    mixed = fit_global(obj, np.array([1.0, 0.4]), bounds=(0.0, np.inf), fixed_mask=[False, True])
    assert mixed.errors[1] == 0.0 and mixed.errors[0] > 0
    allf = fit_global(obj, np.array([1.2, 0.4]), bounds=(0.0, np.inf), fixed_mask=[True, True])
    assert allf.params == {"k1": 1.2, "k2": 0.4}
    assert np.all(allf.errors == 0.0)


def test_initial_value_outside_bounds_is_rejected():
    obj = _first_order_problem()
    with pytest.raises(ValueError, match="outside its bounds"):
        fit_global(obj, np.array([20.0, 0.4]), bounds=(np.array([0.0, 0.0]), np.array([10.0, 5.0])))


def test_unidentifiable_kinetic_parameter_is_nan_not_zero():
    """k3 does not influence the data (C -> D never observed differently)."""
    mech = "species: A, B, C\nA -> B ; k1\nB -> C ; k2\nC -> C ; k3\n"
    model = KineticsModel(parse_mechanism(mech))
    t = np.linspace(0.0, 4.0, 30)
    y0 = {"A": 1.0, "B": 0.0, "C": 0.0}
    A = np.random.default_rng(1).random((3, 8))
    D = model.solve_concentrations(t, y0, {"k1": 1.2, "k2": 0.4, "k3": 1.0}) @ A
    D = D + np.random.default_rng(0).normal(0.0, 1e-3, D.shape)
    obj = GlobalKineticsObjective(model, [KineticsDataset(t=t, y0=y0, D=D)], param_names=["k1", "k2", "k3"])
    res = fit_global(obj, np.array([1.0, 0.5, 1.0]), bounds=(0.0, np.inf))
    assert np.isfinite(res.errors[0]) and np.isfinite(res.errors[1])
    assert np.isnan(res.errors[2])
    assert res.nonidentifiable.tolist() == [False, False, True]


# --------------------------------------------------------------------------------------
# Step 3 — fixed-parameter invariants in every error method (spectroscopy 1:2 model)
# --------------------------------------------------------------------------------------

def _spectro_12(seed=1):
    m = 12
    wl = np.linspace(250, 450, 25)
    c = np.column_stack([np.full(m, 1e-3), np.linspace(0, 4e-3, m)])
    M = np.array([[1.0, 0, 1, 1], [0, 1, 1, 2]])
    solver = NewtonRaphson(c, M, nas=[1], model_sett="Free")
    k = np.array([4.0, 7.0])
    C, _ = solver.concentraciones(k)
    E = np.abs(np.random.default_rng(0).normal(size=(C.shape[1], wl.size))) * 1e3
    Y = (C @ E + np.random.default_rng(seed).normal(0, 1e-3, (m, wl.size))).T
    return k, solver, Y, M


def test_spectro_fixed_mask_analytic_and_one_step():
    k, solver, Y, M = _spectro_12()
    free = compute_errors_spectro_varpro(k, solver, Y, M, [1])
    fx = compute_errors_spectro_varpro(k, solver, Y, M, [1], fixed_mask=[False, True])
    assert fx["SE_log10K"][1] == 0.0
    assert fx["dof"] == free["dof"] + 1  # one fewer estimated parameter
    assert fx["Cov_log10K"].shape == (2, 2)
    assert np.all(fx["Cov_log10K"][1, :] == 0.0)
    boot = bootstrap_one_step_spectro(k, solver, Y, M, [1], 30, seed=0, fixed_mask=[False, True])
    assert np.all(boot["samples"][:, 1] == 7.0)
    assert np.std(boot["samples"][:, 0]) > 0
    with pytest.raises(ValueError):
        compute_errors_spectro_varpro(k, solver, Y, M, [1], fixed_mask=[True])


def test_linearized_bootstrap_never_moves_absent_or_null_directions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    J = np.vstack([a, 2.0 * a, rng.normal(size=100)])  # p1, p2 collinear
    out = bootstrap_linearized_wild(np.zeros(3), J, rng.normal(size=100), 50, seed=1)
    assert np.all(np.isnan(out["samples"][:, :2]))
    assert np.all(np.isfinite(out["samples"][:, 2]))


# --------------------------------------------------------------------------------------
# Step 4 — NumPy arrays in contexts (no truth testing)
# --------------------------------------------------------------------------------------

def _acid_base_nmr_ctx():
    pH = np.linspace(2.0, 10.0, 25)
    y = distribution_fractions_from_pH(pH, [4.0, 7.5]) @ np.array([[7.0], [7.4], [7.9]])
    y = y + np.random.default_rng(0).normal(0.0, 3e-3, y.shape)
    path = os.path.join(tempfile.mkdtemp(), "ab.csv")
    pd.DataFrame({"pH": pH, "H1": y[:, 0]}).to_csv(path, index=False)
    res = run_acid_base(
        {"file_path": path, "data_type": "nmr", "pka_initial": "4.2,7.2",
         "analyte_concentration": 1e-3, "base_charge": -2},
        progress_cb=lambda _m: None,
    )
    return res["errors_context"]


def test_acid_base_context_accepts_numpy_arrays():
    ctx = _acid_base_nmr_ctx()
    ref = compute_errors_acid_base_from_context(ctx, {"method": "analytic"})
    ctx_np = dict(ctx)
    ctx_np["theta_hat"] = np.asarray(ctx["theta_hat"])
    ds = dict(ctx["dataset"])
    ds["pH"] = np.asarray(ds["pH"])
    ds["shifts_observed"] = np.asarray(ds["shifts_observed"])
    ctx_np["dataset"] = ds
    ctx_np["fixed_mask"] = np.asarray(ctx["fixed_mask"], dtype=bool)
    out = compute_errors_acid_base_from_context(ctx_np, {"method": "analytic"})
    assert out["dof"] == ref["dof"] == 25 - 2 - 3
    se_out = out["export_frames"]["Parameters"]["SE"].astype(float).to_numpy()
    se_ref = ref["export_frames"]["Parameters"]["SE"].astype(float).to_numpy()
    assert np.allclose(se_out, se_ref)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_acid_base_mask_and_bounds_normalisation(n):
    assert _normalize_fixed_mask(None, n).tolist() == [False] * n
    assert _normalize_fixed_mask([], n).tolist() == [False] * n
    assert _normalize_fixed_mask(np.ones(n, bool), n).all()
    with pytest.raises(ValueError):
        _normalize_fixed_mask([True] * (n + 1), n)
    lo, hi = _normalize_bounds(np.tile([0.0, 14.0], (n, 1)), n)
    assert lo.tolist() == [0.0] * n and hi.tolist() == [14.0] * n
    with pytest.raises(ValueError, match="lower > upper"):
        _normalize_bounds([(5.0, 1.0)] * n, n)
    with pytest.raises(ValueError):
        _normalize_bounds((np.zeros(n + 1), np.ones(n + 1)), n)


def test_acid_base_empty_parameter_vector_is_rejected():
    with pytest.raises(ValueError):
        compute_errors_acid_base(lambda th: np.ones(3), np.zeros(0), [])


# --------------------------------------------------------------------------------------
# Step 5 — dof and identifiability are never hidden
# --------------------------------------------------------------------------------------

def test_wide_jacobian_identifies_all_nullspace_affected_parameters():
    # n_obs < p: the reduced SVD does not contain the implicit right-null vector.
    J = np.array([[1., 0., 1.], [0., 1., 1.]])
    result = linearized_covariance(J, sse=1.0, n_obs=2)
    assert result["rank"] == 2
    assert result["dof"] == -1
    np.testing.assert_array_equal(result["nonidentifiable"], [True, True, True])
    assert np.isnan(result["cov"]).all()


def test_wide_jacobian_preserves_individually_identifiable_parameter():
    J = np.array([[1., 0., 0.], [0., 2., 2.]])
    for scale in ([1., 1., 1.], [1e-8, 1e4, 1e-3]):
        result = linearized_covariance(J * scale, sse=1.0, n_obs=2)
        np.testing.assert_array_equal(result["nonidentifiable"], [False, True, True])
        assert result["rank"] == 2
        assert result["dof"] == -1


def test_sigma2_reduced_never_clamps_nonpositive_degrees_of_freedom():
    from hmfit_core.utils.errors import sigma2_reduced

    residual = np.array([1., 2.])
    assert np.isnan(sigma2_reduced(residual, p=2))
    assert np.isnan(sigma2_reduced(residual, p=3))
    assert np.isnan(sigma2_reduced(residual, p=0, dof=0))
    assert np.isnan(sigma2_reduced(residual, p=0, dof=-2))
    assert sigma2_reduced(residual, p=1) == pytest.approx(5.0)
    assert sigma2_reduced(residual, p=0, dof=4) == pytest.approx(1.25)


def test_param_errors_cs_nonpositive_dof_reports_undefined():
    from hmfit_core.utils.errors import param_errors_cs

    def residual(theta):
        return np.array([theta[0] - 1.0, theta[1] - 3.0])

    se_log, se_k, pct, variance, cov = param_errors_cs(residual, np.array([2.0, 2.0]))
    assert np.isnan(variance)
    assert np.isnan(se_log).all() and np.isnan(se_k).all() and np.isnan(pct).all()
    assert np.isnan(cov).all()


def test_param_errors_cs_rank_deficiency_does_not_imply_precision():
    from hmfit_core.utils.errors import param_errors_cs

    def residual(theta):
        return np.array([theta[0] + theta[1] - 1.,
                         2. * (theta[0] + theta[1]) - 3.,
                         3. * (theta[0] + theta[1]) - 4.])

    se_log, se_k, pct, variance, cov = param_errors_cs(residual, np.array([1., 1.]))
    assert np.isfinite(variance)
    assert np.isnan(se_log).all() and np.isnan(se_k).all() and np.isnan(pct).all()
    assert np.isnan(cov).all()


def test_param_errors_cs_matches_full_rank_linear_reference():
    from hmfit_core.utils.errors import param_errors_cs

    X = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
    y = np.array([1., 3., 4.5, 6.])
    theta = np.array([1., 2.])
    residual = lambda k: X @ k - y
    se_log, _, _, variance, cov = param_errors_cs(residual, theta)
    expected_s2 = float(residual(theta) @ residual(theta)) / 2
    expected_cov = expected_s2 * np.linalg.inv(X.T @ X)
    assert variance == pytest.approx(expected_s2)
    np.testing.assert_allclose(cov, expected_cov, rtol=1e-12)
    np.testing.assert_allclose(se_log, np.sqrt(np.diag(expected_cov)), rtol=1e-12)


def test_linearized_covariance_reference_and_degenerate_cases():
    rng = np.random.default_rng(3)
    J = rng.normal(size=(40, 2))
    r = rng.normal(size=40)
    lin = linearized_covariance(J, float(r @ r), 40, n_linear=5)
    ref = (r @ r) / (40 - 2 - 5) * np.linalg.inv(J.T @ J)
    assert lin["dof"] == 33 and np.allclose(lin["cov"], ref)
    # dof <= 0: variance undefined, not clamped
    bad = linearized_covariance(J[:6], 1.0, 6, n_linear=4)
    assert bad["dof"] == 0 and np.isnan(bad["s2"]) and not bad["valid"]
    assert np.all(np.isnan(bad["cov"])) and bad["notes"]
    # scale invariance of the rank decision (units: M^-1 s^-1 vs log10 K)
    Js = J * np.array([1e-9, 1e6])
    lin_s = linearized_covariance(Js, float(r @ r), 40)
    assert lin_s["rank"] == 2 and not lin_s["nonidentifiable"].any()


def test_acid_base_dof_nonpositive_reports_nan():
    pH = np.array([3.0, 5.0, 7.0])
    y = distribution_fractions_from_pH(pH, [5.0]) @ np.array([7.1, 7.6]) + np.array([1e-3, -1e-3, 2e-3])
    path = os.path.join(tempfile.mkdtemp(), "tiny.csv")
    pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
    res = run_acid_base(
        {"file_path": path, "data_type": "nmr", "pka_initial": "4.8",
         "analyte_concentration": 1e-3, "base_charge": -1},
        progress_cb=lambda _m: None,
    )
    out = compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    assert out["dof"] == 0
    se = pd.to_numeric(out["export_frames"]["Parameters"]["SE"], errors="coerce").to_numpy(dtype=float)
    assert np.all(np.isnan(se))
    assert "Insufficient degrees of freedom" in out["summary"]


@pytest.mark.parametrize("kind", ["nmr", "spectroscopy"])
def test_acid_base_fixed_pka_is_respected_in_nmr_and_spectroscopy(kind):
    """Defect: pka_fit_mask was ignored outside potentiometry (fixed pKa silently refitted)."""
    from hmfit_core.acid_base import pka_to_log_beta

    pH = np.linspace(2.0, 7.0, 15)
    frac = distribution_fractions_from_pH(pH, pka_to_log_beta([4.0, 9.0]))
    y = frac @ np.array([[7.9], [7.4], [7.0]]) + np.random.default_rng(0).normal(0.0, 3e-3, (15, 1))
    path = os.path.join(tempfile.mkdtemp(), "fx.csv")
    col = "H1" if kind == "nmr" else "signal"
    pd.DataFrame({"pH": pH, col: y[:, 0]}).to_csv(path, index=False)
    res = run_acid_base(
        {"file_path": path, "data_type": kind, "pka_initial": "4.2,9.0", "analyte_concentration": 1e-3,
         "base_charge": -2, "pka_fit_mask": [True, False]},
        progress_cb=lambda _m: None,
    )
    ctx = res["errors_context"]
    assert ctx["fixed_mask"] == [False, True]
    assert ctx["theta_hat"][1] == 9.0
    for method, extra in (("analytic", {}), ("bootstrap_linear", {"B": 40, "seed": 1}),
                          ("bootstrap_full_refit_audit", {"B": 15, "seed": 1})):
        out = compute_errors_acid_base_from_context(ctx, {"method": method, **extra})
        params = out["export_frames"]["Parameters"]
        assert float(params.loc[1, "Estimate"]) == 9.0
        assert float(params.loc[1, "SE"]) == 0.0
        assert abs(float(params.loc[0, "Estimate"]) - 4.0) < 0.05
        assert float(params.loc[0, "SE"]) > 0.0
        assert out["dof"] == 15 - 1 - 3


# --------------------------------------------------------------------------------------
# Steps 6–7 — absorptivity constraints, HC1 terminology, reproducibility
# --------------------------------------------------------------------------------------

def test_absorptivity_constraint_report_counts_active_coefficients():
    from hmfit_core.processors.spectroscopy_processor import absorptivity_constraint_report

    A = np.array([[1.0, -0.2, 3.0], [-1e-3, 2.0, 0.5]])
    rep = absorptivity_constraint_report(A, eps_solver_mode="nnls_hard")
    assert rep["n_active"] == 2 and rep["n_total"] == 6 and "full refit" in rep["note"]
    clean = absorptivity_constraint_report(np.abs(A), eps_solver_mode="soft_penalty")
    assert clean["n_active"] == 0 and clean["note"] == ""
    smooth = absorptivity_constraint_report(np.abs(A), alpha_smooth=0.1)
    assert smooth["penalized"] and "smoothness" in smooth["note"]


def test_hc1_factor_and_resampling_notes():
    from hmfit_core.utils.errors import hc1_rescale_factor, leverage_rescale_factor, resampling_notes

    assert hc1_rescale_factor(100, 4) == pytest.approx(np.sqrt(100 / 96))
    assert hc1_rescale_factor(3, 5) == 1.0
    assert leverage_rescale_factor is hc1_rescale_factor  # backwards-compatible alias
    assert any("Only 12 independent blocks" in n for n in resampling_notes(12, "row", 1))
    assert resampling_notes(200, "row", 2) == []
    assert any("independent errors" in n for n in resampling_notes(500, "element"))


def test_bootstrap_reproducible_with_seed_and_counts_failures():
    from hmfit_core.utils.errors import bootstrap_full_refit

    rng0 = np.random.default_rng(5)
    J = rng0.normal(size=(2, 60))
    r = rng0.normal(size=60)
    a = bootstrap_linearized_wild(np.zeros(2), J, r, 50, seed=11)["samples"]
    b = bootstrap_linearized_wild(np.zeros(2), J, r, 50, seed=11)["samples"]
    c = bootstrap_linearized_wild(np.zeros(2), J, r, 50, seed=12)["samples"]
    assert np.array_equal(a, b) and not np.array_equal(a, c)

    calls = {"n": 0}

    def refit(d, t0, max_iter=30, tol=1e-8):
        calls["n"] += 1
        ok = calls["n"] % 4 != 0  # every 4th replicate fails
        return t0 + 0.01 * d[:1], ok, ({} if ok else {"error": "no convergence"})

    out = bootstrap_full_refit(np.array([1.0]), lambda rng, wild="rademacher": rng.normal(size=3), refit, 40, seed=3)
    assert out["n_fail"] == 10 and out["n_success"] == 30
    assert any("failed and were discarded" in n for n in out["quality_notes"])


def test_acid_base_full_refit_bootstrap_is_reproducible():
    ctx = _acid_base_nmr_ctx()
    o1 = compute_errors_acid_base_from_context(ctx, {"method": "bootstrap_full_refit_audit", "B": 12, "seed": 7})
    o2 = compute_errors_acid_base_from_context(ctx, {"method": "bootstrap_full_refit_audit", "B": 12, "seed": 7})
    p1 = o1["export_frames"]["Parameters"]["SE"].astype(float).to_numpy()
    p2 = o2["export_frames"]["Parameters"]["SE"].astype(float).to_numpy()
    assert np.array_equal(p1, p2)
    assert "Resampling unit" in o1["summary"]
