"""Regression tests for validation/uncertainty_methods.py (Stage F plumbing).

Fast and deterministic: fixed seeds, small B. These lock interface and
statistical sanity of the uncertainty zoo; the scientific numbers live in
validation/results (curated summaries), not here.
"""
from __future__ import annotations

import numpy as np

from validation.bias_experiment import argmin_1d, dataset, profile_objective
from validation.uncertainty_methods import (
    _metrics,
    _pct_ci,
    refit_theta,
    replicate_uncertainty,
)

SCEN = {"points": 8, "noise": 2e-4, "truth": 3.0, "g_max": 2e-3}


def _fit(estimator):
    d = dataset(12345, SCEN)
    mode, mu = ("ols", None) if estimator == "ols" else ("soft_penalty", float(estimator))
    f = profile_objective(d["c"], d["y"], mode=mode, mu=mu if mu is not None else 1e-2)
    theta = argmin_1d(f, 1.6, 6.0)[0]
    return d, theta


def test_metrics_shape_and_dof():
    d, theta = _fit(1e-2)
    m = _metrics(d, theta)
    assert int(m["dof"]) == 8 * 31 - 3 * 31 - 1  # 154
    assert np.isfinite(np.asarray(m["SE_log10K"])).all()


def test_analytic_vs_hc1_factor_closed_form():
    d, theta = _fit(1e-2)
    res = replicate_uncertainty(d, theta, 1e-2, 3.0, B_wild=50, seed=7)
    n_obs, n_lin = int(_metrics(d, theta)["nobs"]), 3 * 31
    fac_expected = np.sqrt(n_obs / (n_obs - 1 - n_lin))
    assert abs(res["hc1"]["hc1_factor"] - fac_expected) / fac_expected < 1e-9
    assert abs(res["hc1"]["se"] / res["analytic"]["se"] - fac_expected) < 1e-9


def test_lin_wild_se_close_to_analytic():
    # Wild linearized bootstrap SE must track the analytic SE (same first-order
    # variance, plus the HC1 factor). Ratio analytic:lin_wild ~ 1/1.27 * ~1.1.
    d, theta = _fit(1e-2)
    res = replicate_uncertainty(d, theta, 1e-2, 3.0, B_wild=800, seed=11)
    ratio = res["lin_wild_el"]["se"] / res["analytic"]["se"]
    assert 0.9 < ratio < 1.8


def test_bootstrap_determinism_given_seed():
    d, theta = _fit(1e-2)
    a = replicate_uncertainty(d, theta, 1e-2, 3.0, B_wild=60, B_refit=4, seed=99)
    b = replicate_uncertainty(d, theta, 1e-2, 3.0, B_wild=60, B_refit=4, seed=99)
    for k in ("lin_wild_el", "lin_wild_spec", "onestep_spec", "onestep_wl",
              "refit_wild_el", "refit_wild_spec", "refit_resample"):
        assert a[k]["ci_low"] == b[k]["ci_low"], k
        assert a[k]["ci_high"] == b[k]["ci_high"], k


def test_refit_theta_determinism_and_span():
    d, theta = _fit(1e-2)
    t1 = refit_theta(d, 1e-2, lo=theta - 0.05, hi=theta + 0.05)
    t2 = refit_theta(d, 1e-2, lo=theta - 0.05, hi=theta + 0.05)
    assert abs(t1 - t2) < 1e-9
    assert abs(t1 - theta) < 0.05
    # full-interval argmin (coarse) also reproduces the estimator
    t3 = refit_theta(d, 1e-2)
    assert abs(t3 - theta) < 1e-4


def test_basic_ci_is_reflection_of_percentile():
    d, theta = _fit(1e-2)
    res = replicate_uncertainty(d, theta, 1e-2, 3.0, B_wild=100, seed=5)
    m = res["lin_wild_el"]
    blo = 2 * theta - m["ci_high"]
    bhi = 2 * theta - m["ci_low"]
    assert abs(m["ci_basic_low"] - blo) < 1e-12
    assert abs(m["ci_basic_high"] - bhi) < 1e-12


def test_empty_samples_ci_is_none():
    m = _pct_ci([np.nan, np.nan], 3.0, center=3.0)
    assert m["n_valid"] == 0 and m["ci_low"] is None


def test_bca_reduces_to_percentile_when_symmetric():
    """With no bias (z0=0) and no acceleration (symmetric jackknife), the
    BCa level adjustment must return the plain 2.5/97.5 percentiles."""
    from validation.uncertainty_methods import _add_bca
    from scipy.stats import norm
    import numpy as np

    # deterministic symmetric bootstrap distribution: exactly half below 3.0
    levels = (np.arange(400) + 0.5) / 400.0
    samples = list(np.sort(3.0 + 0.01 * norm.ppf(levels)))
    theta = 3.0
    jack = [3.0 + dx for dx in (-0.002, -0.001, 0.0, 0.001, 0.002)]
    entry = {}
    _add_bca(entry, samples, theta, jack, 400, truth=3.0)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    assert abs(entry["ci_low_bca"] - lo) < 1e-9
    assert abs(entry["ci_high_bca"] - hi) < 1e-9


def test_bca_shifts_interval_against_positive_bias():
    """A bootstrap distribution centered above theta_hat with a right-skewed
    jackknife should push the BCa interval downward relative to percentiles."""
    from validation.uncertainty_methods import _add_bca
    import numpy as np

    rng = np.random.default_rng(1)
    samples = list(rng.normal(3.01, 0.01, 400))   # boot centered above theta
    theta = 3.005
    jack = [3.012, 3.011, 3.010, 3.009, 3.020]    # right-skewed
    entry = {}
    _add_bca(entry, samples, theta, jack, 400, truth=3.0)
    lo_pct = float(np.quantile(samples, 0.025))
    assert entry["ci_low_bca"] < lo_pct


def test_hc2_hc3_sandwich_ordering_and_finiteness():
    """HC3 inflates at least as much as HC2 (same residuals, squared leverage
    correction); both must be finite and same order as the analytic SE."""
    d, theta = _fit("ols")
    res = replicate_uncertainty(d, theta, "ols", 3.0, B_wild=50, seed=3)
    a, h2, h3 = res["analytic"]["se"], res["hc2"]["se"], res["hc3"]["se"]
    assert all(np.isfinite(v) and v > 0 for v in (a, h2, h3))
    assert h3 >= h2 > 0
    assert 0.3 < h2 / a < 4.0
    # identical center and Student-t width convention
    assert res["hc2"]["ci_low"] < theta < res["hc2"]["ci_high"]


def test_profile_ci_brackets_theta_and_is_finite():
    """The profile interval must bracket the fitted theta_hat with finite,
    positive width on both the constrained and unconstrained projectors."""
    for est in (1e-2, "ols"):
        d, theta = _fit(est)
        res = replicate_uncertainty(d, theta, est, 3.0, B_wild=20, seed=4)
        p = res["profile"]
        assert p["n_valid"] == 1
        assert p["ci_low"] < theta < p["ci_high"]
        assert 0 < p["width95"] < 1.0
        assert abs(p["width95"] - (p["ci_high"] - p["ci_low"])) < 1e-12
