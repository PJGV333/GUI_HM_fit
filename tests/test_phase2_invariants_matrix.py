"""Cross-cutting uncertainty conventions (Phase 2, stage 2.1.3).

Every technique x error method x configuration goes through the SAME dispatch
functions the GUI calls when the user presses "Compute errors":

* spectroscopy / NMR   -> ModelOptPlotsWidget._compute_errors_from_context
* kinetics             -> KineticsModelOptPlotsWidget._compute_errors_from_context
* acid-base (NMR, spectroscopy) and potentiometry
                       -> compute_errors_acid_base_from_context (acid-base tab)

Full-refit bootstraps use the tabs' own ``_refit_from_data`` / ``refit_fn``.

Conventions (Phase 1, docs/STABILITY_PHASE1_REPORT.md):

* SE = 0 means "fixed" and nothing else; a fixed parameter keeps its value and
  its interval collapses to it.
* Undefined uncertainty (non-identifiable parameter, nu <= 0) is NaN, never 0.
* Every correlation that involves a fixed or non-identifiable parameter is NaN;
  when nu <= 0 every correlation is NaN.
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import types
import warnings

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

B_FAST = 12
CONFIGS = ["none_fixed", "mixed", "all_fixed", "nonidentifiable", "dof_le_0"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _fake_tab(tab_cls, context):
    fake = types.SimpleNamespace(_last_fit_context=dict(context))
    fake._parse_float = lambda value, default: float(value) if value not in (None, "") else float(default)
    fake._refit_from_data = types.MethodType(tab_cls._refit_from_data, fake)
    return fake


# ----------------------------------------------------------------------------- data
def _spectro_case(config):
    from hmfit_core.solvers import NewtonRaphson

    rng = np.random.default_rng(11)
    if config == "dof_le_0":
        # 1:1, H, G and HG absorbing, 3 points: n_obs = 3 n_wl, linear = 3 n_wl, p = 1 -> nu = -1
        m, model, nas, k = 3, [[1, 0], [0, 1], [1, 1]], [], np.array([3.0])
    elif config == "nonidentifiable":
        # X has zero stoichiometry and does not absorb: K2 has no effect on the data.
        m, model, nas, k = 10, [[1, 0], [0, 1], [1, 1], [0, 0]], [3], np.array([3.0, 1.0])
    else:
        m, model, nas, k = 12, [[1, 0], [0, 1], [1, 1], [1, 2]], [1], np.array([4.0, 7.0])
    c = np.column_stack([np.full(m, 1e-3), np.linspace(0.0, 4e-3, m)])
    M = np.asarray(model, dtype=float).T
    C, _ = NewtonRaphson(c, M, nas=nas, model_sett="Free").concentraciones(k)
    n_wl = 15
    E = np.abs(rng.normal(size=(C.shape[1], n_wl))) * 1e3
    y_true = (C @ E).T
    Y = y_true + rng.normal(0.0, 1e-3, y_true.shape)
    fixed = {"mixed": [False, True], "all_fixed": [True] * k.size}.get(config, [False] * k.size)
    ctx = {
        "technique": "spectro", "k_hat": k.tolist(), "C_T": c.tolist(), "Y": Y.tolist(), "weights": [],
        "y_fit_hat": y_true.tolist(), "modelo_solver": M.tolist(), "non_abs_species": nas,
        "abs_group_map": [], "algorithm": "Newton-Raphson", "model_settings": "Free",
        "solver_model_settings": "Free", "solver_param_transform": None, "optimizer": "powell",
        "bounds": [(-2.0, 12.0)] * k.size, "fixed_mask": fixed, "eps_solver_mode": "ols",
        "weight_regularization_mode": "legacy", "mu": 1e-2, "delta_mode": "off", "delta_rel": 0.01,
        "alpha_smooth": 0.0, "param_names": [f"K{i + 1}" for i in range(k.size)],
    }
    return ctx, np.asarray(fixed, dtype=bool)


def _nmr_case(config):
    from hmfit_core.processors.nmr_processor import build_D_cols
    from hmfit_core.solvers import NewtonRaphson

    rng = np.random.default_rng(5)
    if config == "dof_le_0":
        # 1:1, two signals, 2 points: n_obs = 4, linear = 4 (H, HG per signal), p = 1 -> nu = -1
        m, model, nas, k = 2, [[1, 0], [0, 1], [1, 1]], [1], np.array([3.0])
    elif config == "nonidentifiable":
        m, model, nas, k = 12, [[1, 0], [0, 1], [1, 1], [0, 0]], [1, 3], np.array([3.0, 1.0])
    else:
        m, model, nas, k = 12, [[1, 0], [0, 1], [1, 1], [1, 2]], [1], np.array([3.5, 6.5])
    H = np.full(m, 1e-3)
    G = np.linspace(2e-4, 5e-3, m)
    C_T = np.column_stack([H, G])
    M = np.asarray(model, dtype=float).T
    _, Co = NewtonRaphson(C_T, M, nas=nas, model_sett="Free").concentraciones(k)
    host = [i for i, row in enumerate(model) if row[0] > 0]
    frac = Co[:, host] * np.asarray([model[i][0] for i in host]) / H[:, None]
    dq_true = np.column_stack([frac @ np.linspace(7.0, 7.6, len(host)), frac @ np.linspace(8.0, 7.5, len(host))])
    dq = dq_true + rng.normal(0.0, 2e-3, dq_true.shape)
    D_cols, _ = build_D_cols(C_T, ["H", "G"], ["s1", "s2"], default_idx=0)
    fixed = {"mixed": [False, True], "all_fixed": [True] * k.size}.get(config, [False] * k.size)
    ctx = {
        "technique": "nmr", "k_hat": k.tolist(), "C_T": C_T.tolist(), "dq": dq.tolist(),
        "dq_fit": dq_true.tolist(), "y_fit_hat": dq_true.tolist(), "column_names": ["H", "G"],
        "signal_names": ["s1", "s2"], "non_abs_species": nas, "fixed_mask": fixed,
        "stoichiometry_map": [], "modelo_solver": M.tolist(), "algorithm": "Newton-Raphson",
        "model_settings": "Free", "optimizer": "powell", "bounds": [(-2.0, 12.0)] * k.size,
        "param_names": [f"K{i + 1}" for i in range(k.size)], "D_cols": D_cols,
    }
    return ctx, np.asarray(fixed, dtype=bool)


def _kinetics_case(config):
    from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab, _run_kinetics_fit

    rng = np.random.default_rng(2)
    if config == "dof_le_0":
        # A -> B observed at 2 times: n_obs = 2 n_wl, linear = 2 n_wl, p = 1 -> nu = -1
        mech, k_true, t = "species: A, B\nA -> B ; k1\n", {"k1": 0.8}, np.array([0.5, 2.0])
    elif config == "nonidentifiable":
        mech, k_true, t = ("species: A, B, C\nA -> B ; k1\nB -> C ; k2\nC -> C ; k3\n",
                           {"k1": 1.2, "k2": 0.4, "k3": 1.0}, np.linspace(0.0, 4.0, 30))
    else:
        mech, k_true, t = ("species: A, B, C\nA -> B ; k1\nB -> C ; k2\n", {"k1": 1.2, "k2": 0.4},
                           np.linspace(0.0, 6.0, 40))
    names = list(k_true)
    model = KineticsModel(parse_mechanism(mech))
    y0 = {sp: (1.0 if sp == "A" else 0.0) for sp in model.dynamic_species}
    C = model.solve_concentrations(t, y0, k_true)
    D = C @ (rng.random((C.shape[1], 8)) + 0.2)
    D = D + rng.normal(0.0, 1e-3, D.shape)
    fixed = {"mixed": [False, True], "all_fixed": [True] * len(names)}.get(config, [False] * len(names))
    config_fit = {
        "model": model, "datasets": [KineticsFitDataset(t=t, y0=y0, D=D)],
        "params0": [k_true[n] * 1.05 if not f else k_true[n] for n, f in zip(names, fixed)],
        "bounds": (0.0, np.inf), "param_names": names, "log_params": [], "fixed_mask": fixed,
        "method": "least_squares", "backend": "scipy", "nnls": False,
    }
    result = _quiet(_run_kinetics_fit, config_fit, progress_cb=lambda *_a, **_k: None)
    # Same bookkeeping as KineticsTab._on_fit_result before the errors context is built.
    for dataset, fit_info in zip(result.get("datasets") or [], result.get("fit_results") or []):
        dataset.fit_C = fit_info.get("C")
        dataset.fit_A = fit_info.get("A")
        dataset.fit_D_hat = fit_info.get("D_hat")
        dataset.fit_residuals = fit_info.get("residuals")
    captured = {}
    fake = types.SimpleNamespace(
        model_opt_plots=types.SimpleNamespace(set_errors_context=lambda ctx, auto_compute=False: captured.update(ctx))
    )
    KineticsTab._update_errors_context_from_result(fake, result)
    return captured, np.asarray(fixed, dtype=bool)


def _acid_base_case(kind, config, tmp):
    from hmfit_core.acid_base import distribution_fractions_from_pH, make_simple_acid_base_system, pka_to_log_beta
    from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
    from hmfit_core.run_acid_base import run_acid_base

    rng = np.random.default_rng(7)
    pka = [8.0, 4.0]  # HM fit convention: pKa1 (most basic site) > pKa2
    mask = {"mixed": [True, False], "all_fixed": [False, False]}.get(config, [True, True])
    path = os.path.join(tmp, f"{kind}_{config}.csv")
    if kind == "potentiometry":
        n = 2 if config == "dof_le_0" else 40
        vol = np.linspace(2.0, 25.0, n)
        system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=pka, base_charge=-2)
        exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                      titrant_concentration=1e-2, titrant_type="base")
        pH = simulate_pH_titration(exp, system) + rng.normal(0.0, 3e-3, n)
        pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
        cfg = {"titrant_concentration": 1e-2, "initial_volume": 10.0, "titrant_type": "base",
               "analyte_concentration": 1e-2}
    else:
        # nu <= 0: 2 pKa + 3 species signals per channel, 4 pH points -> nu = 4 - 2 - 3 = -1
        n = 4 if config == "dof_le_0" else 25
        pH = np.linspace(2.0, 10.0, n)
        frac = distribution_fractions_from_pH(pH, pka_to_log_beta(pka))
        y = frac @ np.asarray([7.9, 7.4, 7.0]) + rng.normal(0.0, 3e-3, n)
        col = "H1" if kind == "nmr" else "signal"
        pd.DataFrame({"pH": pH, col: y}).to_csv(path, index=False)
        cfg = {"analyte_concentration": 1e-3}
        if kind == "spectroscopy":
            cfg["baseline"] = False
    # No acid-base model offers an exactly non-identifiable constant without
    # misspecifying the model (a pKa far outside the pH range is only *nearly*
    # non-identifiable: see STABILITY_PHASE2_REPORT.md, open problems). The
    # "nonidentifiable" configuration therefore fits the regular 2-pKa model and
    # replaces the pKa2 column of the context Jacobian by an exact null column;
    # every method must then report pKa2 as undefined.
    pka_fit = pka
    cfg.update({"file_path": path, "data_type": kind, "base_charge": -len(pka_fit),
                "pka_initial": ",".join(str(v + (0.2 if m else 0.0)) for v, m in zip(pka_fit, mask)),
                "pka_fit_mask": mask})
    if config == "all_fixed":
        cfg["pka_initial"] = ",".join(str(v) for v in pka_fit)
    try:
        res = run_acid_base(cfg, progress_cb=lambda _m: None)
    except ValueError as exc:
        return None, str(exc)
    if not res.get("success", False):
        return None, str(res.get("error"))
    ctx = res["errors_context"]
    if config == "nonidentifiable":
        jac = np.asarray(ctx["jacobian"], dtype=float)
        jac[:, 1] = 0.0
        ctx = dict(ctx, jacobian=jac.tolist())
    return ctx, None


# ----------------------------------------------------------------------------- dispatch
def _run_gui_dispatch(technique, method, config, qapp):
    if technique == "spectro":
        from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab
        from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

        ctx, fixed = _spectro_case(config)
        ctx["refit_from_data"] = _fake_tab(SpectroscopyTab, ctx)._refit_from_data
        widget = ModelOptPlotsWidget()
    elif technique == "nmr":
        from hmfit_gui_qt.tabs.nmr_tab import NMRTab
        from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

        ctx, fixed = _nmr_case(config)
        ctx["refit_from_data"] = _fake_tab(NMRTab, ctx)._refit_from_data
        widget = ModelOptPlotsWidget()
    else:
        from hmfit_gui_qt.widgets.kinetics_model_opt_plots import KineticsModelOptPlotsWidget

        ctx, fixed = _kinetics_case(config)
        widget = KineticsModelOptPlotsWidget()
        if method == "bootstrap_full_refit_audit":
            method = "bootstrap_full_refit"
    opts = {"method": method, "B": B_FAST, "seed": 123, "wild": "rademacher", "lam": 1e-3,
            "max_iter": 60, "tol": 1e-8, "fail_policy": "skip", "wild_block": "row"}
    try:
        out = _quiet(widget._compute_errors_from_context, ctx, options=opts)
    finally:
        widget.deleteLater()
    est = np.asarray(out["k_hat"], dtype=float)
    se = np.asarray(out.get("se_log10", out.get("se")), dtype=float)
    return {
        "est": est, "se": se, "lo": np.asarray(out["ci_2p5"], dtype=float),
        "hi": np.asarray(out["ci_97p5"], dtype=float), "corr": np.asarray(out["corr"], dtype=float),
        "fixed": fixed, "k_in": np.asarray(ctx["k_hat"], dtype=float),
    }


def _run_acid_base(kind, method, config, tmp):
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context

    ctx, problem = _acid_base_case(kind, config, tmp)
    if ctx is None:
        return {"rejected": problem}
    opts = {"method": method, "B": B_FAST, "seed": 123, "wild": "rademacher", "max_iter": 60,
            "tol": 1e-8, "fail_policy": "skip", "wild_block": "row"}
    out = _quiet(compute_errors_acid_base_from_context, ctx, opts)
    rows = out["results_rows"]
    fixed = np.asarray([row[-1] == "Yes" for row in rows], dtype=bool)
    num = lambda col: np.asarray([np.nan if row[col] in ("", None) else float(row[col]) for row in rows])
    full_theta = dict(zip(ctx["parameter_names"], ctx["theta_hat"]))
    for item in ctx.get("fixed_constants") or []:
        full_theta[item["name"]] = item["value"]
    return {
        "est": num(1), "se": num(2), "lo": num(3), "hi": num(4), "corr": np.asarray(out["corr"], dtype=float),
        "fixed": fixed, "k_in": np.asarray([full_theta[row[0]] for row in rows], dtype=float),
        "dof": out["dof"],
    }


# ----------------------------------------------------------------------------- matrix
GUI_TECHNIQUES = ["spectro", "nmr", "kinetics"]
GUI_METHODS = ["analytic", "bootstrap_linear", "bootstrap_onestep", "bootstrap_full_refit_audit"]
AB_KINDS = ["nmr", "spectroscopy", "potentiometry"]
# The acid-base tab offers no one-step bootstrap (see acid_base_tab combo).
AB_METHODS = ["analytic", "bootstrap_linear", "bootstrap_full_refit_audit"]

CASES = (
    [("gui", t, m, c) for t in GUI_TECHNIQUES for m in GUI_METHODS for c in CONFIGS]
    + [("acid_base", k, m, c) for k in AB_KINDS for m in AB_METHODS for c in CONFIGS]
)


def _nonidentifiable_index(family, technique):
    # spectro/NMR: K2 (zero-stoichiometry spectator); kinetics: k3 (C -> C); acid-base: pKa2.
    if family == "acid_base":
        return 1
    return 2 if technique == "kinetics" else 1


@pytest.mark.parametrize("family, technique, method, config", CASES,
                         ids=[f"{'ab-' if f == 'acid_base' else ''}{t}-{m}-{c}" for f, t, m, c in CASES])
def test_uncertainty_conventions(family, technique, method, config, qapp, tmp_path):
    if family == "gui":
        res = _run_gui_dispatch(technique, method, config, qapp)
    else:
        res = _run_acid_base(technique, method, config, str(tmp_path))
        if "rejected" in res:
            # A configuration the fit refuses outright (e.g. nu <= 0 in potentiometry with
            # 2 points) is acceptable only with an explicit message, never a silent number.
            assert res["rejected"], "fit refused the configuration without a message"
            return
    est, se, lo, hi, corr, fixed = res["est"], res["se"], res["lo"], res["hi"], res["corr"], res["fixed"]
    p = est.size
    assert se.shape == (p,) and corr.shape == (p, p)

    # Fixed parameters: value kept, SE exactly 0, interval collapsed, correlations NaN.
    for i in np.where(fixed)[0]:
        assert est[i] == pytest.approx(res["k_in"][i], abs=1e-12)
        assert se[i] == 0.0, f"fixed parameter {i}: SE={se[i]}"
        assert lo[i] == pytest.approx(est[i], abs=1e-12) and hi[i] == pytest.approx(est[i], abs=1e-12)
        assert np.all(np.isnan(corr[i, :])) and np.all(np.isnan(corr[:, i])), f"corr with fixed {i}:\n{corr}"

    free = np.where(~fixed)[0]
    if config == "dof_le_0":
        # Undefined: NaN SE for every free parameter (never 0) and NaN correlations.
        assert free.size > 0
        assert np.all(np.isnan(se[free])), f"nu <= 0 must give NaN SE, got {se}"
        assert np.all(np.isnan(corr)), f"nu <= 0 must give NaN correlations:\n{corr}"
        return

    undefined = np.zeros(p, dtype=bool)
    if config == "nonidentifiable":
        j = _nonidentifiable_index(family, technique)
        undefined[j] = True
        assert np.isnan(se[j]), f"non-identifiable parameter {j}: SE={se[j]} (must be NaN)"
        assert np.all(np.isnan(corr[j, :])) and np.all(np.isnan(corr[:, j])), f"corr with non-identifiable:\n{corr}"

    defined = np.where(~fixed & ~undefined)[0]
    for i in defined:
        assert np.isfinite(se[i]) and se[i] > 0.0, f"identifiable free parameter {i}: SE={se[i]}"
        assert corr[i, i] == pytest.approx(1.0)
    for a in defined:
        for b in defined:
            assert np.isfinite(corr[a, b]), f"defined pair ({a},{b}) has corr {corr[a, b]}"
