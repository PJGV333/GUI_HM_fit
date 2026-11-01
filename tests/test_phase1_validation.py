"""Phase 1 numerical validation with synthetic data of known parameters.

Tolerances are statistical, not tuned: with Gaussian noise |estimate - true| / SE follows
approximately |t_dof|, so 4 is a > 99.9 % bound for a single deterministic realisation.
Small Monte Carlo checks use 40 replicates: the sampling SD of the ratio SE/SD is about
1/sqrt(2*39) ~ 0.11, hence the [0.67, 1.4] acceptance band (~3 sigma). Large coverage
campaigns are deferred to Phase 2.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, os.path.dirname(__file__))
import phase1_cases as pc  # noqa: E402


@pytest.fixture(scope="module")
def cases():
    return {c["case"]: c for c in pc.all_cases()}


def _z(c):
    est, tru, se = (np.asarray(c[k], float) for k in ("estimate", "true", "se"))
    free = se > 0
    return np.abs(est[free] - tru[free]) / se[free], est, tru, se, free


NOISY = [
    "UV-Vis 1:1, log K = 3", "UV-Vis 1:2, log b2 fixed", "NMR 1:1, log K = 3", "NMR 1:2, log b2 fixed",
    "Acid-base NMR, pKa 4.0/7.5", "Acid-base NMR, close pKa 4.0/4.4", "Acid-base NMR, pKa2 fixed",
    "Potentiometry, pKa 5.0", "Kinetics 1st order A->B->C", "Kinetics 1st order, k2 fixed",
    "Kinetics 2nd order, k = 5e3 M^-1 s^-1",
]


@pytest.mark.parametrize("name", NOISY)
def test_parameter_recovery_consistent_with_reported_se(cases, name):
    z, est, tru, se, free = _z(cases[name])
    assert np.all(np.isfinite(se[free])) and np.all(se[free] > 0)
    assert np.all(z < 4.0), (name, est, tru, se)


@pytest.mark.parametrize("name", [n for n in NOISY if "fixed" in n])
def test_fixed_parameters_exact_and_zero_se(cases, name):
    c = cases[name]
    se = np.asarray(c["se"], float)
    est = np.asarray(c["estimate"], float)
    tru = np.asarray(c["true"], float)
    assert np.any(se == 0.0)
    assert np.all(est[se == 0.0] == tru[se == 0.0])


@pytest.mark.parametrize("name", ["UV-Vis 1:1, no noise", "Kinetics 1st order, no noise"])
def test_noiseless_data_recover_exact_parameters(cases, name):
    c = cases[name]
    assert np.allclose(c["estimate"], c["true"], rtol=1e-6, atol=1e-6)


def test_acid_base_covariance_matches_independent_finite_difference():
    """SE and correlation of two close pKa vs an independent central-difference computation
    of s^2 (J^T J)^-1 with dof = n - p - n_linear (VarPro residual, 3 limiting shifts)."""
    from hmfit_core.acid_base import NMRAcidBaseDataset, distribution_fractions_from_pH, nmr_acid_base_residuals, pka_to_log_beta
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from hmfit_core.run_acid_base import _build_system, run_acid_base

    pH = np.linspace(2.0, 10.0, 25)
    y = distribution_fractions_from_pH(pH, pka_to_log_beta([4.0, 4.4])) @ np.array([7.9, 7.5, 7.0])
    y = y + np.random.default_rng(0).normal(0.0, 3e-3, pH.size)
    path = os.path.join(tempfile.mkdtemp(), "fd.csv")
    pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
    cfg = {"file_path": path, "data_type": "nmr", "pka_initial": "3.8,4.6",
           "analyte_concentration": 1e-3, "base_charge": -2}
    r = run_acid_base(cfg, progress_cb=lambda _m: None)
    out = compute_errors_acid_base_from_context(r["errors_context"], {"method": "analytic"})
    theta = np.asarray(r["errors_context"]["theta_hat"], float)

    ds = NMRAcidBaseDataset(pH=pH, shifts=y.reshape(-1, 1), nuclei_labels=["H1"])
    system = _build_system(cfg)
    f = lambda th: nmr_acid_base_residuals(np.asarray(th, float), ds, system, {})
    h = 1e-5
    J = np.column_stack([(f(theta + h * e) - f(theta - h * e)) / (2 * h) for e in np.eye(2)])
    res = f(theta)
    cov = (res @ res) / (25 - 2 - 3) * np.linalg.inv(J.T @ J)
    se_ref = np.sqrt(np.diag(cov))
    corr_ref = cov[0, 1] / (se_ref[0] * se_ref[1])
    P = out["export_frames"]["Parameters"]
    assert np.allclose(P["SE"].astype(float).to_numpy(), se_ref, rtol=1e-3)
    assert abs(np.asarray(out["corr"], float)[0, 1] - corr_ref) < 1e-3


def test_second_order_rate_constant_scale(cases):
    """k ~ 5e3 M^-1 s^-1 with concentrations ~1e-4 M: SE must be on the scale of k, not ~0."""
    c = cases["Kinetics 2nd order, k = 5e3 M^-1 s^-1"]
    assert 1.0 < c["se"][0] < 100.0


# ------------------------------------------------------------------ small Monte Carlo

def test_mc_acid_base_nmr_small_titration_se_calibrated():
    """10-point monoprotic NMR titration (dof = 10 - 1 pKa - 2 limiting shifts = 7)."""
    from hmfit_core.acid_base import distribution_fractions_from_pH
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from hmfit_core.run_acid_base import run_acid_base

    pH = np.linspace(3.0, 7.0, 10)
    sh = distribution_fractions_from_pH(pH, [5.0]) @ np.array([7.10, 7.60])
    est, se, cover = [], [], []
    path = os.path.join(tempfile.mkdtemp(), "mc.csv")
    for i in range(40):
        y = sh + np.random.default_rng(1000 + i).normal(0.0, 5e-3, pH.size)
        pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
        r = run_acid_base({"file_path": path, "data_type": "nmr", "pka_initial": "4.6",
                           "analyte_concentration": 1e-3, "base_charge": -1}, progress_cb=lambda _m: None)
        P = compute_errors_acid_base_from_context(r["errors_context"], {"method": "analytic"})["export_frames"]["Parameters"]
        est.append(float(P.loc[0, "Estimate"]))
        se.append(float(P.loc[0, "SE"]))
        cover.append(float(P.loc[0, "CI 2.5"]) <= 5.0 <= float(P.loc[0, "CI 97.5"]))
    ratio = np.mean(se) / np.std(est, ddof=1)
    assert 0.67 < ratio < 1.4, ratio
    assert np.mean(cover) >= 0.85


def test_mc_spectro_constrained_estimator_has_no_large_bias():
    """Before the exact absorptivity solver, log K was biased by ~+0.043 on this design."""
    vals = []
    for i in range(12):
        c = pc.spectro_case("mc", [3.0], [[1, 0], [0, 1], [1, 1]], [2.5], [(0.0, 6.0)], seed=500 + i)
        vals.append(c["estimate"][0])
    assert abs(np.mean(vals) - 3.0) < 0.02, np.mean(vals)
