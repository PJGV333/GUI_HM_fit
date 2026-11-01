"""Regression tests for the reliability of HM fit uncertainty estimates.

Each test pins a defect that silently produced wrong (mostly too small) errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hmfit_core.acid_base import distribution_fractions_from_pH
from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
from hmfit_core.run_acid_base import run_acid_base
from hmfit_core.utils.errors import (
    block_wild_multipliers,
    bootstrap_linearized_wild,
    leverage_rescale_factor,
    lm_damped_step,
    make_wild_data_star_fn,
    t_quantile,
)
from hmfit.kinetics.data.dataset import KineticsDataset
from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import fit_global
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel


def test_lm_damped_step_is_scale_invariant():
    """lam*I damping shrank steps to ~0 when J^T J was tiny (k in M^-1 s^-1)."""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(2, 200))
    r = rng.normal(size=200)
    for scale in (1e-6, 1.0, 1e6):
        step = lm_damped_step(J * scale, r, lam=1e-3)
        ref = lm_damped_step(J, r, lam=1e-3) / scale
        assert np.allclose(step, ref, rtol=1e-6)
    gn = -np.linalg.solve(J @ J.T, J @ r)
    assert np.allclose(lm_damped_step(J * 1e-4, r, lam=1e-3) * 1e-4, gn, rtol=5e-3)


def test_block_wild_multipliers_share_rows():
    rng = np.random.default_rng(1)
    v = block_wild_multipliers(rng, (7, 5), block="row")
    assert v.shape == (7, 5)
    assert np.all(v == v[:, :1])
    v_el = block_wild_multipliers(rng, (50, 50), block="element")
    assert not np.all(v_el == v_el[:, :1])


def test_t_quantile_and_leverage():
    assert abs(t_quantile(7) - 2.3646) < 1e-3
    assert abs(t_quantile(None) - 1.96) < 1e-2
    assert leverage_rescale_factor(10, 3) == np.sqrt(10 / 7)


def test_data_star_keeps_unobserved_entries():
    fitted = np.ones((4, 3))
    observed = fitted + 0.1
    observed[1, 2] = np.nan
    fn = make_wild_data_star_fn(fitted, observed, point_axis=0, block="row")
    star = fn(np.random.default_rng(0))
    assert np.isnan(star[1, 2])
    assert np.all(np.isfinite(np.delete(star.ravel(), 5)))


def test_linearized_bootstrap_groups_validate_length():
    J = np.ones((1, 6))
    r = np.arange(6.0)
    out = bootstrap_linearized_wild(np.zeros(1), J, r, 20, seed=0, groups=np.repeat([0, 1, 2], 2))
    assert out["samples"].shape == (20, 1)


def _kinetics_problem(seed=0, noise=1e-2):
    mech = "species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"
    model = KineticsModel(parse_mechanism(mech))
    t = np.linspace(0.0, 4.0, 30)
    y0 = {"A": 1.0, "B": 0.0, "C": 0.0}
    A = np.random.default_rng(1).random((3, 12))
    D = model.solve_concentrations(t, y0, {"k1": 1.2, "k2": 0.4}) @ A
    D = D + np.random.default_rng(seed).normal(0.0, noise, D.shape)
    return model, KineticsDataset(t=t, y0=y0, D=D)


def test_kinetics_fixed_parameter_does_not_crash_and_has_zero_se():
    model, ds = _kinetics_problem()
    obj = GlobalKineticsObjective(model, [ds], param_names=["k1", "k2"])
    res = fit_global(obj, np.array([1.0, 0.4]), bounds=(np.array([0.0, 0.4]), np.array([10.0, 0.4])))
    assert res.params["k2"] == 0.4
    assert abs(res.params["k1"] - 1.2) < 0.1
    assert res.errors[1] == 0.0 and res.errors[0] > 0.0


def test_kinetics_dof_counts_linear_spectra():
    model, ds = _kinetics_problem()
    obj = GlobalKineticsObjective(model, [ds], param_names=["k1", "k2"])
    res = fit_global(obj, np.array([1.0, 0.5]), bounds=(0.0, np.inf))
    n = ds.D.size
    assert res.result.n_linear == 3 * ds.D.shape[1]
    assert res.result.dof == n - 2 - 3 * ds.D.shape[1]


def test_kinetics_analytic_se_is_calibrated():
    est, se = [], []
    for i in range(40):
        model, ds = _kinetics_problem(seed=100 + i)
        obj = GlobalKineticsObjective(model, [ds], param_names=["k1", "k2"])
        res = fit_global(obj, np.array([1.0, 0.5]), bounds=(0.0, np.inf))
        est.append([res.params["k1"], res.params["k2"]])
        se.append(res.errors)
    ratio = np.mean(se, axis=0) / np.std(est, axis=0, ddof=1)
    assert np.all((ratio > 0.75) & (ratio < 1.3)), ratio


def test_acid_base_nmr_dof_subtracts_limiting_shifts(tmp_path):
    pH = np.linspace(3.0, 7.0, 10)
    y = distribution_fractions_from_pH(pH, [5.0]) @ np.array([7.10, 7.60])
    y = y + np.random.default_rng(0).normal(0.0, 5e-3, y.shape)
    path = tmp_path / "nmr.csv"
    pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
    result = run_acid_base(
        {"file_path": str(path), "data_type": "nmr", "pka_initial": "4.6",
         "analyte_concentration": 1e-3, "base_charge": -1},
        progress_cb=lambda _m: None,
    )
    err = compute_errors_acid_base_from_context(result["errors_context"], {"method": "analytic"})
    assert err["dof"] == 10 - 1 - 2
    row = err["export_frames"]["Parameters"].iloc[0]
    half_width = (float(row["CI 97.5"]) - float(row["CI 2.5"])) / 2.0
    assert abs(half_width / float(row["SE"]) - t_quantile(7)) < 1e-6


def test_soft_penalty_absorptivity_solver_is_exact_at_molar_scale():
    """The fixed-iteration gradient solver did not converge for C ~ 1e-3 M (biased log K)."""
    from hmfit_core.processors.spectroscopy_processor import _solve_absorptivities

    rng = np.random.default_rng(0)
    C = np.abs(rng.normal(size=(12, 3))) * 1e-3
    A_true = np.abs(rng.normal(size=(3, 20))) * 1e3
    A_true[:, :4] = 0.0  # transparent region -> LS estimate goes slightly negative
    Y = C @ A_true + rng.normal(0.0, 2e-4, (12, 20))
    A, _ = _solve_absorptivities(C, Y, eps_solver_mode="soft_penalty", mu=1e-2)
    A = np.asarray(A)
    # KKT of 1/2||CA-Y||^2 + 1/2 mu ||min(A,0)||^2 : gradient must vanish
    grad = C.T @ (C @ A - Y) + 1e-2 * np.minimum(A, 0.0)
    assert np.max(np.abs(grad)) < 1e-9
    # columns without negative LS entries are the plain LS solution
    A_ls = np.linalg.lstsq(C, Y, rcond=None)[0]
    clean = np.all(A_ls >= 0.0, axis=0)
    assert np.allclose(A[:, clean], A_ls[:, clean])


def test_bootstrap_correlation_for_single_parameter():
    """np.corrcoef returns a scalar for one parameter; 1:1 bootstraps used to crash."""
    from hmfit_core.utils.errors import bootstrap_full_refit

    out = bootstrap_full_refit(
        np.array([3.0]),
        lambda rng, wild="rademacher": rng.normal(size=3),
        lambda d, t0, max_iter=30, tol=1e-8: (t0 + 0.01 * d[:1], True, {}),
        10,
        seed=0,
    )
    assert out["corr"].shape == (1, 1)
