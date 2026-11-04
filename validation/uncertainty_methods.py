"""Uncertainty-method zoo for one validation replicate (Stage F evidence).

All methods are evaluated at the SAME fitted theta_hat and dataset, so
intervals from different methods are directly comparable. Methods:

- analytic        production VarPro covariance (unconstrained projector)
- hc1             analytic SE rescaled by the global HC1 factor
- lin_wild_el     linearized wild bootstrap, elementwise multipliers (B draws)
- lin_wild_spec   linearized wild bootstrap, one multiplier per spectrum
- onestep_el      one-step (LM damped) bootstrap, elementwise wild
- onestep_spec    one-step bootstrap, wild multipliers per spectrum
- refit_wild_el   full-refit bootstrap on elementwise-wild perturbed data
- refit_wild_spec full-refit bootstrap on per-spectrum wild perturbed data
- refit_resample  full-refit bootstrap, residuals resampled i.i.d. per element

Full refits re-run the profiled argmin with the SAME linear projector used to
fit (estimator: "ols", "nnls", or a float mu for the soft penalty). Layouts:
the harness dataset stores y as (n_points, n_wl); compute_errors_spectro_varpro
consumes Y as (n_wl, n_points) with weights (n_points, n_wl), matching the
production call already locked by tests/test_uvvis_bias_reference.py.

Pure validation-layer code: production routines are called, never modified.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import t

from validation.bias_experiment import argmin_1d, profile_objective

from hmfit_core.solvers import NewtonRaphson
from hmfit_core.utils.errors import (
    bootstrap_linearized_wild,
    bootstrap_one_step_spectro,
    compute_errors_spectro_varpro,
    hc1_rescale_factor,
    wild_multipliers,
)

MODEL = [[1, 0], [0, 1], [1, 1]]


def _solver(d):
    return NewtonRaphson(d["c"], np.asarray(MODEL, dtype=float).T, nas=[], model_sett="Free")


def _objective(d, estimator):
    """theta -> profiled RMS under the requested linear projector."""
    if estimator == "ols":
        mode, mu = "ols", None
    elif estimator == "nnls":
        mode, mu = "nnls_hard", None
    else:
        mode, mu = "soft_penalty", float(estimator)
    f = profile_objective(d["c"], d["y"], mode=mode, mu=mu if mu is not None else 1e-2)
    return f


def refit_theta(d, estimator, lo=1.6, hi=6.0):
    """Argmin of the profiled objective for the given projector (coarse grid:
    bootstrap refits need only ~1e-6 resolution)."""
    return argmin_1d(_objective(d, estimator), lo, hi, coarse=81, tol=1e-8)[0]


def _metrics(d, theta):
    return compute_errors_spectro_varpro(np.asarray([theta], dtype=float), _solver(d),
                                         d["y"].T, np.asarray(MODEL, dtype=float).T, [],
                                         weights=np.ones_like(d["y"]))


def _pct_ci(samples, truth, center=None):
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"se": None, "ci_low": None, "ci_high": None, "covered95": None,
                "covered68": None, "width95": None, "n_valid": 0}
    lo, hi = np.percentile(s, [2.5, 97.5])
    l68, h68 = np.percentile(s, [16, 84])
    out = {"se": float(np.std(s, ddof=1)),
           "ci_low": float(lo), "ci_high": float(hi),
           "covered95": bool(lo <= truth <= hi), "covered68": bool(l68 <= truth <= h68),
           "width95": float(hi - lo), "n_valid": int(s.size)}
    if center is not None:
        blo, bhi = 2.0 * center - hi, 2.0 * center - lo
        out["ci_basic_low"], out["ci_basic_high"] = float(blo), float(bhi)
        out["covered95_basic"] = bool(blo <= truth <= bhi)
        out["width95_basic"] = float(bhi - blo)
        out["boot_bias"] = float(np.mean(s) - center)
    return out


def replicate_uncertainty(d, theta, estimator, truth, *,
                          B_wild=1000, B_refit=0, seed=0, refit_span=0.05):
    """Return {method: metrics dict} for one replicate."""
    out = {}
    t0 = time.perf_counter()
    m = _metrics(d, theta)
    se = float(np.asarray(m["SE_log10K"]).ravel()[0])
    dof = int(m["dof"])
    q = float(t.ppf(0.975, dof))
    q68 = float(t.ppf(0.84, dof))
    out["analytic"] = {"se": se, "ci_low": theta - q * se, "ci_high": theta + q * se,
                       "covered95": bool(theta - q * se <= truth <= theta + q * se),
                       "covered68": bool(theta - q68 * se <= truth <= theta + q68 * se),
                       "width95": 2 * q * se, "n_valid": 1, "dof": dof}
    nlin = int(m.get("n_linear", 0) or 0)
    fac = float(hc1_rescale_factor(int(m["nobs"]), 1 + nlin))
    se1 = se * fac
    out["hc1"] = {"se": se1, "ci_low": theta - q * se1, "ci_high": theta + q * se1,
                  "covered95": bool(theta - q * se1 <= truth <= theta + q * se1),
                  "covered68": bool(theta - q68 * se1 <= truth <= theta + q68 * se1),
                  "width95": 2 * q * se1, "n_valid": 1, "hc1_factor": fac}

    J = np.asarray(m["J"], dtype=float)[np.asarray(m["free_idx"], dtype=int), :]
    r = np.asarray(m["r"], dtype=float).ravel()      # layout (m, nw) row-major
    nres = r.size
    n_pts = d["y"].shape[0]
    n_wl = d["y"].shape[1]
    groups_spec = np.repeat(np.arange(n_pts), n_wl)  # spectrum index per residual

    bl = bootstrap_linearized_wild(np.asarray([theta]), J, r, B_wild, seed=seed,
                                   wild="rademacher", n_linear=nlin,
                                   leverage_correction=True)
    out["lin_wild_el"] = _pct_ci(bl["samples"][:, 0], truth, center=theta)
    bl2 = bootstrap_linearized_wild(np.asarray([theta]), J, r, B_wild, seed=seed + 1,
                                    wild="rademacher", groups=groups_spec,
                                    n_linear=nlin, leverage_correction=True)
    out["lin_wild_spec"] = _pct_ci(bl2["samples"][:, 0], truth, center=theta)

    # one-step wild: "row" = per spectrum (titration point), "col" = per wavelength
    out["onestep_spec"] = _onestep(d, theta, "row", B_wild, seed + 3, truth)
    out["onestep_wl"] = _onestep(d, theta, "col", B_wild, seed + 4, truth)

    if B_refit > 0:
        yfit_h = np.asarray(m["yfit"], dtype=float).T          # (m, nw) harness layout
        R_h = (yfit_h - d["y"]).reshape(-1)                    # aligned with r
        for name, kind in (("refit_wild_el", "wild_el"), ("refit_wild_spec", "wild_spec"),
                           ("refit_resample", "resample_el")):
            samples = []
            fails = 0
            rb = np.random.default_rng(seed + 17)
            for _ in range(int(B_refit)):
                if kind == "wild_el":
                    pert = -R_h * wild_multipliers(rb, nres)
                elif kind == "wild_spec":
                    pert = -R_h * wild_multipliers(rb, n_pts)[groups_spec]
                else:
                    pert = r[rb.integers(0, nres, nres)]
                d_star = dict(d)
                d_star["y"] = yfit_h + pert.reshape(n_pts, n_wl)
                try:
                    th = refit_theta(d_star, estimator,
                                     lo=theta - refit_span, hi=theta + refit_span)
                    samples.append(th)
                except Exception:
                    fails += 1
            out[name] = _pct_ci(samples, truth, center=theta)
            out[name]["n_fail"] = int(fails)
            out[name]["B"] = int(B_refit)
    out["_seconds"] = {"total": time.perf_counter() - t0}
    return out


def _onestep(d, theta, block, B, seed, truth):
    res = bootstrap_one_step_spectro(np.asarray([theta], dtype=float), _solver(d),
                                     d["y"].T, np.asarray(MODEL, dtype=float).T, [], B,
                                     seed=seed, wild_block=block, leverage_correction=True)
    return _pct_ci(res["samples"][:, 0], truth, center=theta)
