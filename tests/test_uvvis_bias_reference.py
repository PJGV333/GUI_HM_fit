"""UV-Vis 1:1 estimator/reference equivalence tests (bias investigation, Stage A-D).

These lock the conclusions of docs/UVVIS_BIAS_INVESTIGATION.md against
regression:
  * zero-noise fits recover log K exactly (no model or implementation bias);
  * the independent closed-form 1:1 equilibrium and independent VarPro
    reference agree with production to <1e-5 (no convergence gap);
  * the analytic VarPro Jacobian equals the projected identity
    (I - P_C) dC/dtheta * A built from closed-form sensitivities;
  * production default soft penalty (mu=1e-2) is numerically equivalent to
    exact NNLS at pilot concentrations (C ~ 1e-3 M, C^T C ~ 1e-5);
  * the archived pilot estimates remain reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation.bias_experiment import (MODEL, argmin_1d, dataset, profile_objective,
                                        reference_C, reference_rss_profile)

SEED0 = 3796957675          # pilot replicate 0 (seed derived from base seed 20261102)
PILOT0_ESTIMATE = 3.0014548933722764   # archived results.csv replicate 0


def _simple_dataset(seed=SEED0, noise=2e-4):
    return dataset(seed, {"points": 8, "noise": noise, "truth": 3.0, "g_max": 2e-3})


def test_closed_form_equilibrium_matches_newton_raphson():
    from hmfit_core.solvers import NewtonRaphson
    d = _simple_dataset(noise=0.0)
    solver = NewtonRaphson(d["c"], d["M"], nas=[], model_sett="Free")
    worst = 0.0
    for theta in np.linspace(1.0, 5.0, 9):
        C = solver.concentraciones(np.asarray([theta]))[0]
        worst = max(worst, float(np.max(np.abs(C - reference_C(d["c"], theta))) / np.max(C)))
    # NewtonRaphson converges to an absolute tolerance of 1e-10 on concentrations
    # of order 1e-3 M, so agreement at the 1e-7 relative level is the solver floor.
    assert worst < 1e-6, worst


def test_zero_noise_argmin_recovers_truth_for_all_estimators():
    d = _simple_dataset(noise=0.0)
    for mode in ("ols", "soft_penalty", "nnls_hard"):
        x, fx, edge = argmin_1d(profile_objective(d["c"], d["y"], mode=mode), 2.5, 3.5)
        assert not edge
        assert abs(x - 3.0) < 1e-6, (mode, x)
        assert fx < 1e-12, (mode, fx)


def test_zero_noise_production_fit_recovers_truth():
    import tempfile
    from pathlib import Path
    from tests.phase1_cases import _spectro_cfg, _spectro_dataset, _quiet
    from hmfit_core import run_spectroscopy
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.xlsx"
        _spectro_dataset(path, [3.0], MODEL, n_points=8, noise=0.0, seed=SEED0)
        out = _quiet(run_spectroscopy, _spectro_cfg(path, MODEL, [2.5], [(0.0, 6.0)]))
        assert out["success"] and out["optimizer_result"]["success"]
        k = float(np.asarray(out["export_data"]["k"])[0])
        assert abs(k - 3.0) < 1e-6, k


def test_production_powell_matches_exact_argmin_and_independent_reference():
    from scipy.optimize import OptimizeResult, minimize_scalar
    d = _simple_dataset()
    x_pen, _, _ = argmin_1d(profile_objective(d["c"], d["y"], mode="soft_penalty"), 2.0, 4.0)
    assert abs(x_pen - PILOT0_ESTIMATE) < 5e-6, x_pen  # exact argmin == archived Powell fit
    res: OptimizeResult = minimize_scalar(reference_rss_profile(d["c"], d["y"], kind="nnls"),
                          bounds=(2.0, 4.0), method="bounded", options={"xatol": 1e-9})
    assert abs(float(res["x"]) - PILOT0_ESTIMATE) < 5e-6   # independent code path
    x_ols, _, _ = argmin_1d(profile_objective(d["c"], d["y"], mode="ols"), 2.0, 4.0)
    res_ols: OptimizeResult = minimize_scalar(reference_rss_profile(d["c"], d["y"], kind="ols"),
                              bounds=(2.0, 4.0), method="bounded", options={"xatol": 1e-9})
    assert abs(x_ols - float(res_ols["x"])) < 1e-5


def test_analytic_jacobian_equals_projected_identity():
    from hmfit_core.solvers import NewtonRaphson
    from hmfit_core.utils.errors import compute_errors_spectro_varpro
    d = _simple_dataset()
    th = PILOT0_ESTIMATE
    solver = NewtonRaphson(d["c"], d["M"], nas=[], model_sett="Free")
    m = compute_errors_spectro_varpro(np.asarray([th]), solver, d["y"].T, d["M"], [],
                                      weights=np.ones_like(d["y"]))
    J_impl = np.asarray(m["J"])[0].reshape(d["c"].shape[0], d["y"].shape[1])
    h = 1e-5
    C = reference_C(d["c"], th)
    A = np.linalg.lstsq(C, d["y"], rcond=None)[0]
    dC = (reference_C(d["c"], th + h) - reference_C(d["c"], th - h)) / (2 * h)
    dCp = dC - C @ np.linalg.lstsq(C, dC, rcond=None)[0]
    J_ident = dCp @ A
    rel = np.abs(J_impl - J_ident).max() / np.abs(J_ident).max()
    assert rel < 1e-3, rel
    # s2 and dof-based SE agree between implementation and the identity Jacobian
    se_ident = float(np.sqrt(float(m["s2"]) / float(J_ident.ravel() @ J_ident.ravel())))
    se_impl = float(np.asarray(m["SE_log10K"])[0])
    assert abs(se_ident - se_impl) / se_impl < 1e-3


def test_soft_penalty_equivalence_regimes():
    d = _simple_dataset()
    from hmfit_core.processors import spectroscopy_processor as sp
    C = reference_C(d["c"], 3.0)
    a_ols = np.linalg.lstsq(C, d["y"], rcond=None)[0]
    a_mu0 = sp._exact_soft_penalty_columns(C, d["y"], mu=0.0)
    assert np.max(np.abs(a_mu0 - a_ols)) < 1e-9           # mu=0 is the OLS projection
    a_pen = sp._exact_soft_penalty_columns(C, d["y"], mu=1e-2)
    a_nnls = sp._exact_nnls_columns(C, d["y"])
    scale = np.abs(a_nnls).max()
    assert np.max(np.abs(a_pen - a_nnls)) / scale < 1e-2  # near-hard at C^T C ~ 1e-5
    # and the penalty is genuinely soft at mu << C^T C: differs from both
    a_soft = sp._exact_soft_penalty_columns(C, d["y"], mu=1e-9)
    assert np.max(np.abs(a_soft - a_nnls)) / scale > 1e-3


@pytest.mark.parametrize("rep,expected", [(0, 3.0014548933722764), (1, 3.000339637023707),
                                          (2, 3.0233454943974807)])
def test_pilot_estimates_are_reproducible(rep, expected):
    """Re-run the production fitter on archived pilot replicate seeds."""
    import tempfile
    from pathlib import Path
    from tests.phase1_cases import _spectro_cfg, _spectro_dataset, _quiet
    from hmfit_core import run_spectroscopy
    from validation.bias_experiment import seed_for
    seed = seed_for(20261102, rep)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.xlsx"
        _spectro_dataset(path, [3.0], MODEL, n_points=8, noise=2e-4, seed=seed)
        out = _quiet(run_spectroscopy, _spectro_cfg(path, MODEL, [2.5], [(0.0, 6.0)]))
        k = float(np.asarray(out["export_data"]["k"])[0])
        assert abs(k - expected) < 1e-6, (rep, k)
