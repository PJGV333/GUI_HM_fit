"""Phase 2 coverage campaign (stage 2.2) through the GUI entry points.

Every replicate is fitted and analysed exactly as the GUI does it:

* fit: ``hmfit_core.api.run_spectroscopy_fit`` / ``run_nmr_fit`` / ``run_acid_base_fit``
  and ``hmfit_gui_qt.tabs.kinetics_tab._run_kinetics_fit`` (the FitWorker targets);
* errors context: the tabs' own ``_update_errors_context_from_result`` (and the tabs'
  ``_refit_from_data`` / ``refit_fn`` for full-refit bootstraps);
* errors: ``ModelOptPlotsWidget._compute_errors_from_context`` (UV-Vis, NMR),
  ``KineticsModelOptPlotsWidget._compute_errors_from_context`` (kinetics) and
  ``compute_errors_acid_base_from_context`` (acid-base tab).

Nothing statistical is re-implemented here; this module only generates synthetic data
with known parameters, calls those entry points, and summarises the results. It is the
regression oracle for the Phase 3 unification of the uncertainty engines.

Usage (see docs/PHASE2_PREREGISTRATION.md for the frozen grid):

    python -m validation.coverage_campaign grid --write validation/configs/coverage/primary_grid.json
    python -m validation.coverage_campaign run  GRID OUT [--select REGEX] [--tier T] [--workers K]
                                                [--n-rep N] [--methods m1,m2]
    python -m validation.coverage_campaign summarize OUT [--csv FILE]

Determinism: replicate r of cell c uses SeedSequence([BASE_SEED, crc32(c), r]); each
error method gets its own child seed. Results are appended per replicate to
``OUT/<cell>.jsonl``; re-running skips replicates already present (resumable).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import types
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

BASE_SEED = 20261105
TRUTH_K11 = [3.0]
TRUTH_K12 = [3.0, 2.5]

# GUI method keys, resampling block. Full-refit for kinetics is "bootstrap_full_refit".
METHODS = {
    "analytic": ("analytic", None),
    "lin_row": ("bootstrap_linear", "row"),
    "lin_el": ("bootstrap_linear", "element"),
    "os_row": ("bootstrap_onestep", "row"),
    "os_el": ("bootstrap_onestep", "element"),
    "fr_row": ("bootstrap_full_refit_audit", "row"),
    "fr_el": ("bootstrap_full_refit_audit", "element"),
}
ALL_METHODS = list(METHODS)
NO_ONESTEP = [m for m in ALL_METHODS if not m.startswith("os_")]


# ============================================================================ grid
CHEAP = ["analytic", "lin_row", "lin_el", "os_row", "os_el"]
CHEAP_NO_OS = ["analytic", "lin_row", "lin_el"]


def _cell_id(cell):
    base = "-".join(str(cell[k]) for k in ("technique", "design", "n_points", "noise", "saturation"))
    if cell.get("estimator", "soft_penalty") != "soft_penalty":
        base += f"-{cell['estimator']}"
    return base + cell.get("pass_suffix", "")


def primary_grid() -> list[dict[str, Any]]:
    """The pre-registered primary grid (docs/PHASE2_PREREGISTRATION.md, section 3).

    Each cell is one *pass*: a physical design plus the methods, N and B run on it.
    Passes ``~fr`` re-use the same datasets (same seeds) for the full-refit bootstrap
    at reduced N/B (compute budget, pre-registration section 5).
    """
    cells: list[dict[str, Any]] = []

    def add(**cell):
        cell.setdefault("N", 500)
        cell.setdefault("B", 1000)
        cell.setdefault("B_refit", 500)
        cell.setdefault("estimator", "soft_penalty")
        cell["seed_id"] = _cell_id({**cell, "pass_suffix": ""})
        cell["id"] = _cell_id(cell)
        cells.append(cell)

    # --- UV-Vis (GUI default estimator: soft_penalty, mu = 1e-2) -------------------
    for design in ("11zero", "11pos", "12zero"):
        for n in (8, 10, 12, 15, 20, 30):
            for noise in ("iid", "hetero", "offset"):
                for sat in ("rec", "poor"):
                    tier = 1 if design.startswith("11") and n <= 15 else 4
                    add(technique="uvvis", design=design, n_points=n, noise=noise, saturation=sat,
                        methods=CHEAP, tier=tier)
                    if design == "11zero" and n in (8, 12, 15) and noise in ("iid", "offset"):
                        add(technique="uvvis", design=design, n_points=n, noise=noise, saturation=sat,
                            methods=["fr_row"] + (["fr_el"] if noise == "iid" else []), tier=2,
                            N=200, B_refit=200, pass_suffix="~fr")
    # --- NMR -------------------------------------------------------------------------
    for design in ("11", "12"):
        for n in (8, 10, 12, 15, 20):
            for noise in ("iid1", "iid5", "point"):
                for sat in ("rec", "poor"):
                    tier = 1 if design == "11" and n <= 15 else 4
                    add(technique="nmr", design=design, n_points=n, noise=noise, saturation=sat,
                        methods=CHEAP, tier=tier)
                    if design == "11" and n in (8, 12, 15) and noise in ("iid1", "point"):
                        add(technique="nmr", design=design, n_points=n, noise=noise, saturation=sat,
                            methods=["fr_row"] + (["fr_el"] if noise == "iid1" else []), tier=2,
                            N=200, B_refit=200, pass_suffix="~fr")
    # --- Stage 2.3 estimator comparison (UV-Vis, same data as the default estimator) --
    for estimator in ("unconstrained", "nnls_hard"):
        for design in ("11zero", "11pos", "12zero"):
            for n in (8, 12, 15):
                for sat in ("rec", "poor"):
                    add(technique="uvvis", design=design, n_points=n, noise="iid", saturation=sat,
                        estimator=estimator, methods=["analytic", "lin_row", "os_row"], tier=3)
                    if design != "11pos" and n in (8, 15):
                        add(technique="uvvis", design=design, n_points=n, noise="iid", saturation=sat,
                            estimator=estimator, methods=["fr_row"], tier=3, N=200, B_refit=200,
                            pass_suffix="~fr")
    for design in ("12zero",):
        for n in (8, 15):
            for sat in ("rec", "poor"):
                add(technique="uvvis", design=design, n_points=n, noise="iid", saturation=sat,
                    methods=["fr_row"], tier=3, N=200, B_refit=200, pass_suffix="~fr")
    # --- acid-base (NMR, spectroscopy), potentiometry, kinetics ------------------------
    for kind in ("ab_nmr", "ab_spec"):
        for design in ("sep", "close", "fixed"):
            for n in (10, 15, 25, 40):
                for noise in ("iid", "offset"):
                    add(technique=kind, design=design, n_points=n, noise=noise, saturation="na",
                        methods=CHEAP_NO_OS + ["fr_row", "fr_el"], tier=3)
    for design in ("1pka", "2pka"):
        for n in (30, 60):
            add(technique="pot", design=design, n_points=n, noise="iid", saturation="na",
                methods=["analytic", "lin_row"], tier=3)
            add(technique="pot", design=design, n_points=n, noise="iid", saturation="na",
                methods=["fr_row"], tier=3, N=200, B_refit=200, pass_suffix="~fr")
    for design in ("consec_sim", "consec_diff", "second"):
        for n in (50, 200):
            for noise in ("iid", "drift"):
                add(technique="kin", design=design, n_points=n, noise=noise, saturation="na",
                    methods=CHEAP, tier=3)
                if n == 50:
                    add(technique="kin", design=design, n_points=n, noise=noise, saturation="na",
                        methods=["fr_row"], tier=3, N=200, B_refit=200, pass_suffix="~fr")
    return cells


def diagnostic_grid() -> list[dict[str, Any]]:
    """Titrant-concentration error of +/-3 % (no coverage expectation)."""
    cells = []
    for tech, design in (("uvvis", "11zero"), ("nmr", "11")):
        for bias in (-0.03, 0.03):
            cell = dict(technique=tech, design=design, n_points=12, noise="iid" if tech == "uvvis" else "iid1",
                        saturation="rec", methods=["analytic", "lin_row", "fr_row"], tier=9, N=200, B=1000,
                        B_refit=200, titrant_bias=bias)
            cell["id"] = f"diag-{tech}-{design}-titrant{bias:+.2f}"
            cells.append(cell)
    return cells


def smoke_grid() -> list[dict[str, Any]]:
    """Tiny cells, one per technique, for CI (campaign_smoke)."""
    cells = [
        dict(technique="uvvis", design="11zero", n_points=10, noise="iid", saturation="rec"),
        dict(technique="nmr", design="11", n_points=10, noise="iid1", saturation="rec"),
        dict(technique="ab_nmr", design="sep", n_points=15, noise="iid", saturation="na"),
        dict(technique="ab_spec", design="fixed", n_points=15, noise="iid", saturation="na"),
        dict(technique="pot", design="2pka", n_points=30, noise="iid", saturation="na"),
        dict(technique="kin", design="consec_diff", n_points=50, noise="iid", saturation="na"),
    ]
    for cell in cells:
        cell.update(N=3, B=30, B_refit=6, tier=0,
                    methods=["analytic", "lin_row", "fr_row"] + (["os_row"] if cell["technique"] in ("uvvis", "nmr", "kin") else []))
        cell["id"] = "smoke-" + "-".join(str(cell[k]) for k in ("technique", "design", "n_points", "noise"))
    return cells


# ============================================================================ data
def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _guest_for_saturation(model, k_log, h0, targets):
    """Total guest giving each target host-bound fraction (bisection on the GUI solver)."""
    from hmfit_core.solvers import NewtonRaphson

    M = np.asarray(model, dtype=float).T
    out = []
    for s in targets:
        if s <= 0:
            out.append(0.0)
            continue
        lo, hi = 0.0, h0 * 1e4
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sol = NewtonRaphson(np.array([[h0, max(mid, 1e-30)]]), M, nas=[], model_sett="Free")
            C, _ = sol.concentraciones(np.asarray(k_log, dtype=float))
            frac = 1.0 - C[0, 0] / h0
            lo, hi = (mid, hi) if frac < s else (lo, mid)
        out.append(0.5 * (lo + hi))
    return np.asarray(out)


def _saturation_targets(n, regime):
    lo, hi = (0.2, 0.8) if regime == "rec" else (0.05, 0.45)
    return np.concatenate([[0.0], np.linspace(lo, hi, n - 1)])


def _binding_design(cell):
    model = [[1, 0], [0, 1], [1, 1]] if cell["design"].startswith("11") else [[1, 0], [0, 1], [1, 1], [1, 2]]
    k_true = TRUTH_K11 if cell["design"].startswith("11") else TRUTH_K12
    h0 = 1e-3
    g = _guest_for_saturation(model, k_true, h0, _saturation_targets(int(cell["n_points"]), cell["saturation"]))
    return model, np.asarray(k_true, dtype=float), h0, g


def _uvvis_data(cell, rng, workdir):
    import pandas as pd
    from hmfit_core.solvers import NewtonRaphson

    model, k_true, h0, g = _binding_design(cell)
    n = g.size
    c_true = np.column_stack([np.full(n, h0), g])
    M = np.asarray(model, dtype=float).T
    C, _ = NewtonRaphson(c_true, M, nas=[1], model_sett="Free").concentraciones(k_true)
    wl = np.linspace(240.0, 600.0, 31)
    centers = np.linspace(280.0, 420.0, C.shape[1])
    bands = np.vstack([1.0e3 * np.exp(-((wl - c0) ** 2) / (2 * 30.0**2)) for c0 in centers])
    if cell["design"].endswith("zero"):
        bands[np.abs(wl[None, :] - centers[:, None]) > 2.0 * 30.0] = 0.0  # exact zeros
    else:
        bands = bands + 150.0  # every absorptivity > 0
    y = C @ bands
    noise = cell["noise"]
    if noise == "iid":
        y = y + rng.normal(0.0, 1e-3, y.shape)
    elif noise == "hetero":
        y = y + rng.normal(0.0, 1.0, y.shape) * 1e-3 * (0.5 + 1.5 * np.abs(y) / np.max(np.abs(y)))
    elif noise == "offset":
        y = y + rng.normal(0.0, 1e-3, (n, 1)) + rng.normal(0.0, 5e-4, y.shape)
    else:
        raise ValueError(noise)
    c_file = c_true.copy()
    c_file[:, 1] *= 1.0 + float(cell.get("titrant_bias", 0.0))
    path = os.path.join(workdir, "uvvis.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(y.T, index=wl, columns=[f"p{i + 1}" for i in range(n)]).to_excel(w, sheet_name="Spectra")
        pd.DataFrame({"H": c_file[:, 0], "G": c_file[:, 1]}).to_excel(w, sheet_name="Conc", index=False)
    cfg = {
        "file_path": path, "spectra_sheet": "Spectra", "conc_sheet": "Conc", "column_names": ["H", "G"],
        "receptor_label": "H", "guest_label": "G", "efa_enabled": False, "efa_eigenvalues": 0,
        "modelo": model, "non_abs_species": [1], "algorithm": "Newton-Raphson", "model_settings": "Free",
        "optimizer": "powell", "initial_k": (k_true + 0.3).tolist(), "bounds": [(0.0, 8.0)] * k_true.size,
        "channels_raw": "All", "channels_mode": "all", "channels_resolved": [],
        "show_stability_diagnostics": False, "multi_start_runs": 1,
        # GUI estimators: soft_penalty (default, mu = 1e-2); "unconstrained" = soft_penalty
        # with mu = 0 (reachable from the GUI mu field); nnls_hard.
        "eps_solver_mode": "nnls_hard" if cell.get("estimator") == "nnls_hard" else "soft_penalty",
        "mu": 0.0 if cell.get("estimator") == "unconstrained" else 1e-2,
    }
    return cfg, k_true


def _nmr_data(cell, rng, workdir):
    import pandas as pd
    from hmfit_core.solvers import NewtonRaphson

    model, k_true, h0, g = _binding_design(cell)
    n = g.size
    c_true = np.column_stack([np.full(n, h0), g])
    M = np.asarray(model, dtype=float).T
    _, Co = NewtonRaphson(c_true, M, nas=[1], model_sett="Free").concentraciones(k_true)
    host = [i for i, row in enumerate(model) if row[0] > 0]
    frac = Co[:, host] * np.asarray([model[i][0] for i in host]) / h0
    s1 = frac @ np.linspace(7.00, 7.60, len(host))
    s2 = frac @ np.linspace(8.00, 7.50, len(host))
    shifts = np.column_stack([s1, s2])
    noise = cell["noise"]
    if noise in ("iid1", "iid5"):
        shifts = shifts + rng.normal(0.0, 1e-3 if noise == "iid1" else 5e-3, shifts.shape)
    elif noise == "point":
        shifts = shifts + rng.normal(0.0, 3e-3, (n, 1)) + rng.normal(0.0, 1e-3, shifts.shape)
    else:
        raise ValueError(noise)
    g_file = g * (1.0 + float(cell.get("titrant_bias", 0.0)))
    path = os.path.join(workdir, "nmr.xlsx")
    pd.DataFrame({"H": np.full(n, h0), "G": g_file, "s1": shifts[:, 0], "s2": shifts[:, 1]}).to_excel(
        path, index=False, sheet_name="S")
    cfg = {
        "file_path": path, "nmr_sheet": "S", "conc_sheet": "S", "column_names": ["H", "G"],
        "signal_names": ["s1", "s2"], "receptor_label": "H", "guest_label": "G", "modelo": model,
        "initial_k": (k_true + 0.3).tolist(), "bounds": [(0.0, 8.0)] * k_true.size,
        "algorithm": "Newton-Raphson", "optimizer": "powell", "model_settings": "Free",
        "non_abs_species": [1], "multi_start_runs": 1,
    }
    return cfg, k_true


# HM fit convention (docs/ACID_BASE_POTENTIOMETRY.md): pKa_n = log beta_n - log beta_(n-1)
# with cumulative *protonation* constants, so pKa1 belongs to HL <=> L + H+ (the most
# basic site) and a physical diprotic ladder has pKa1 > pKa2. [4, 8] would describe a
# cooperative system in which HL is never populated and only pKa1 + pKa2 is identifiable.
AB_PKA = {"sep": [8.0, 4.0], "close": [5.0, 4.4], "fixed": [8.0, 4.0]}


def _ab_data(cell, rng, workdir):
    import pandas as pd
    from hmfit_core.acid_base import distribution_fractions_from_pH, pka_to_log_beta

    pka = AB_PKA[cell["design"]]
    n = int(cell["n_points"])
    pH = np.linspace(2.0, 10.0, n)
    frac = distribution_fractions_from_pH(pH, pka_to_log_beta(pka))
    if cell["technique"] == "ab_nmr":
        signal = frac @ np.asarray([[7.90, 8.40], [7.40, 8.10], [7.00, 7.60]])
        sigma, offset, cols = 3e-3, 3e-3, ["H1", "H2"]
    else:
        wl = np.linspace(250.0, 400.0, 8)
        eps = np.vstack([0.9 * np.exp(-((wl - c) ** 2) / (2 * 35.0**2)) + 0.05 for c in (270.0, 320.0, 370.0)])
        signal = frac @ eps
        sigma, offset, cols = 1e-3, 1e-3, [f"A{int(w)}" for w in wl]
    signal = signal + rng.normal(0.0, sigma, signal.shape)
    if cell["noise"] == "offset":
        signal = signal + rng.normal(0.0, offset, (n, 1))
    path = os.path.join(workdir, "ab.csv")
    pd.DataFrame(np.column_stack([pH, signal]), columns=["pH"] + cols).to_csv(path, index=False)
    mask = [True, False] if cell["design"] == "fixed" else [True, True]
    cfg = {
        "file_path": path, "data_type": "nmr" if cell["technique"] == "ab_nmr" else "spectroscopy",
        "pka_initial": ",".join(str(v + (0.3 if m else 0.0)) for v, m in zip(pka, mask)),
        "pka_fit_mask": mask, "analyte_concentration": 1e-3, "base_charge": -len(pka),
    }
    if cell["technique"] == "ab_spec":
        cfg["baseline"] = False
    return cfg, np.asarray(pka, dtype=float)


def _pot_data(cell, rng, workdir):
    import pandas as pd
    from hmfit_core.acid_base import make_simple_acid_base_system
    from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration

    pka = [5.0] if cell["design"] == "1pka" else [7.5, 4.0]  # pKa1 > pKa2 (see AB_PKA)
    n = int(cell["n_points"])
    system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=pka, base_charge=-len(pka))
    vol = np.linspace(0.0, 10.0 * len(pka) * 1.8, n)
    exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                  titrant_concentration=1e-2, titrant_type="base")
    pH = simulate_pH_titration(exp, system) + rng.normal(0.0, 5e-3, n)
    path = os.path.join(workdir, "pot.csv")
    pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
    cfg = {"file_path": path, "data_type": "potentiometry", "pka_initial": ",".join(str(v + 0.3) for v in pka),
           "analyte_concentration": 1e-2, "titrant_concentration": 1e-2, "initial_volume": 10.0,
           "titrant_type": "base", "base_charge": -len(pka)}
    return cfg, np.asarray(pka, dtype=float)


KIN = {
    "consec_sim": ("species: A, B, C\nA -> B ; k1\nB -> C ; k2\n", {"k1": 1.0, "k2": 0.7}, 8.0),
    "consec_diff": ("species: A, B, C\nA -> B ; k1\nB -> C ; k2\n", {"k1": 1.0, "k2": 0.1}, 40.0),
    "second": ("species: A, B, C\nA + B -> C ; k1\n", {"k1": 2.0}, 4.0),
}


def _kin_data(cell, rng, workdir):
    from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel

    mech, k_true, t_max = KIN[cell["design"]]
    model = KineticsModel(parse_mechanism(mech))
    t = np.linspace(0.0, t_max, int(cell["n_points"]))
    y0 = {sp: (1.0 if sp == "A" else (1.2 if sp == "B" and cell["design"] == "second" else 0.0))
          for sp in model.dynamic_species}
    C = model.solve_concentrations(t, y0, k_true)
    A = np.random.default_rng(1).random((C.shape[1], 12)) + 0.2  # fixed spectra across replicates
    D = C @ A + rng.normal(0.0, 5e-3, (t.size, A.shape[1]))
    if cell["noise"] == "drift":
        D = D + np.outer(t / t_max, rng.normal(0.0, 5e-3, A.shape[1]))
    names = list(k_true)
    cfg = {"model": model, "datasets": [KineticsFitDataset(t=t, y0=y0, D=D)],
           "params0": [k_true[nm] * 1.3 for nm in names], "bounds": (0.0, np.inf), "param_names": names,
           "log_params": [], "fixed_mask": [False] * len(names), "method": "least_squares",
           "backend": "scipy", "nnls": False}
    return cfg, np.asarray([k_true[nm] for nm in names], dtype=float)


# ============================================================================ GUI paths
_WIDGETS: dict[str, Any] = {}


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _widget(kind):
    if kind not in _WIDGETS:
        _qapp()
        if kind == "kin":
            from hmfit_gui_qt.widgets.kinetics_model_opt_plots import KineticsModelOptPlotsWidget

            _WIDGETS[kind] = KineticsModelOptPlotsWidget()
        else:
            from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

            _WIDGETS[kind] = ModelOptPlotsWidget()
    return _WIDGETS[kind]


def _capture_context(tab_cls, result, last_config):
    """Run the tab's own _update_errors_context_from_result and capture the context."""
    captured: dict[str, Any] = {}
    fake = types.SimpleNamespace(
        _last_config=dict(last_config), _last_fit_context=None,
        model_opt_plots=types.SimpleNamespace(
            set_errors_context=lambda ctx, auto_compute=False: captured.update(ctx or {})),
    )
    fake._parse_float = lambda value, default: float(value) if value not in (None, "") else float(default)
    if hasattr(tab_cls, "_refit_from_data"):
        fake._refit_from_data = types.MethodType(tab_cls._refit_from_data, fake)
    tab_cls._update_errors_context_from_result(fake, result)
    return captured


def _fit(cell, cfg):
    """Fit with the GUI's worker target; returns (ctx, estimate, converged, info)."""
    tech = cell["technique"]
    if tech in ("uvvis", "nmr"):
        from hmfit_core.api import run_nmr_fit, run_spectroscopy_fit

        if tech == "uvvis":
            from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab as Tab

            result = _quiet(run_spectroscopy_fit, cfg)
        else:
            from hmfit_gui_qt.tabs.nmr_tab import NMRTab as Tab

            result = _quiet(run_nmr_fit, cfg)
        ok = bool(result.get("success", True))
        ctx = _capture_context(Tab, result, cfg) if ok else {}
        k = np.asarray((result.get("export_data") or {}).get("k") or [np.nan], dtype=float)
        return ctx, k, ok and bool(ctx), {}
    if tech in ("ab_nmr", "ab_spec", "pot"):
        from hmfit_core.api import run_acid_base_fit

        result = _quiet(run_acid_base_fit, cfg)
        ok = bool(result.get("success", False))
        ctx = result.get("errors_context") or {}
        return ctx, None, ok and bool(ctx), {}
    if tech == "kin":
        from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab, _run_kinetics_fit

        result = _quiet(_run_kinetics_fit, cfg, progress_cb=lambda *_a, **_k: None)
        for dataset, fit_info in zip(result.get("datasets") or [], result.get("fit_results") or []):
            dataset.fit_C, dataset.fit_A = fit_info.get("C"), fit_info.get("A")
            dataset.fit_D_hat, dataset.fit_residuals = fit_info.get("D_hat"), fit_info.get("residuals")
        captured: dict[str, Any] = {}
        fake = types.SimpleNamespace(model_opt_plots=types.SimpleNamespace(
            set_errors_context=lambda ctx, auto_compute=False: captured.update(ctx or {})))
        KineticsTab._update_errors_context_from_result(fake, result)
        ok = bool(result.get("success", True)) and bool((result.get("diagnostics") or {}).get("optimizer_success", True))
        return captured, np.asarray(captured.get("k_hat"), dtype=float), ok, {}
    raise ValueError(tech)


def _errors(cell, ctx, method_key, B, seed):
    method, block = METHODS[method_key]
    tech = cell["technique"]
    opts = {"method": method, "B": int(B), "seed": int(seed), "wild": "rademacher", "lam": 1e-3,
            "max_iter": 30, "tol": 1e-8, "fail_policy": "skip", "wild_block": block or "row",
            "include_16_84": False}
    if tech in ("ab_nmr", "ab_spec", "pot"):
        from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context

        out = _quiet(compute_errors_acid_base_from_context, ctx, opts)
        rows = out["results_rows"]
        num = lambda c: np.asarray([np.nan if r[c] in ("", None) else float(r[c]) for r in rows], dtype=float)
        fixed = np.asarray([r[-1] == "Yes" for r in rows], dtype=bool)
        return num(1), num(2), num(3), num(4), fixed, out.get("n_fail")
    if tech == "kin" and method == "bootstrap_full_refit_audit":
        opts["method"] = "bootstrap_full_refit"
    widget = _widget("kin" if tech == "kin" else "binding")
    out = _quiet(widget._compute_errors_from_context, ctx, options=opts)
    se = out.get("se_log10", out.get("se"))
    fixed = np.asarray(ctx.get("fixed_mask") or [False] * len(out["k_hat"]), dtype=bool)
    if fixed.size != len(out["k_hat"]):
        fixed = np.zeros(len(out["k_hat"]), dtype=bool)
    return (np.asarray(out["k_hat"], float), np.asarray(se, float), np.asarray(out["ci_2p5"], float),
            np.asarray(out["ci_97p5"], float), fixed, out.get("n_fail"))


GENERATORS = {"uvvis": _uvvis_data, "nmr": _nmr_data, "ab_nmr": _ab_data, "ab_spec": _ab_data,
              "pot": _pot_data, "kin": _kin_data}


def seed_for(cell_id: str, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([BASE_SEED, zlib.crc32(cell_id.encode()), int(rep)])


def run_replicate(cell: dict[str, Any], rep: int, methods: list[str] | None = None) -> dict[str, Any]:
    ss = seed_for(cell.get("seed_id", cell["id"]), rep)
    data_ss, *method_ss = ss.spawn(1 + len(ALL_METHODS))
    rng = np.random.default_rng(data_ss)
    row: dict[str, Any] = {"cell": cell["id"], "rep": int(rep), "entropy": [int(v) for v in np.atleast_1d(ss.entropy)]}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        cfg, truth = GENERATORS[cell["technique"]](cell, rng, workdir)
        row["truth"] = truth.tolist()
        try:
            ctx, est, ok, _ = _fit(cell, cfg)
        except Exception as exc:  # recorded as a fit failure, never silently dropped
            ctx, est, ok = {}, None, False
            row["fit_error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["fit_ok"] = bool(ok)
        row["fit_seconds"] = time.perf_counter() - t0
        row["methods"] = {}
        if not ok:
            return row
        for idx, mkey in enumerate(ALL_METHODS):
            if mkey not in (methods or cell["methods"]):
                continue
            B = int(cell["B_refit"] if mkey.startswith("fr_") else cell["B"])
            seed = int(method_ss[idx].generate_state(1)[0] % (2**31 - 1))
            t1 = time.perf_counter()
            try:
                est_m, se, lo, hi, fixed, n_fail = _errors(cell, ctx, mkey, B, seed)
                row["methods"][mkey] = {
                    "est": est_m.tolist(), "se": se.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                    "fixed": fixed.tolist(), "n_fail": None if n_fail is None else int(n_fail),
                    "B": B, "seconds": time.perf_counter() - t1,
                }
            except Exception as exc:
                row["methods"][mkey] = {"error": f"{type(exc).__name__}: {exc}"[:300],
                                        "seconds": time.perf_counter() - t1}
    row["seconds"] = time.perf_counter() - t0
    return row


# ============================================================================ runner
def _done_reps(path: Path) -> set[int]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(int(json.loads(line)["rep"]))
                except (ValueError, KeyError):
                    continue  # a torn last line is recomputed
    return done


def _worker(args):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cell, rep, methods = args
    return cell["id"], run_replicate(cell, rep, methods)


def _provenance() -> dict[str, Any]:
    import platform
    import subprocess

    import scipy

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except Exception:
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty, "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": _ver("pandas"), "PySide6": _ver("PySide6"),
            "base_seed": BASE_SEED, "argv": sys.argv}


def run(grid: list[dict[str, Any]], out: Path, *, workers: int = 1, n_rep: int | None = None,
        methods: list[str] | None = None, progress: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), **_provenance(),
                             "cells": [c["id"] for c in grid], "n_rep": n_rep, "methods": methods}) + "\n")
    tasks = []
    for cell in grid:
        target = int(n_rep if n_rep is not None else cell["N"])
        done = _done_reps(out / f"{cell['id']}.jsonl")
        tasks += [(cell, r, methods) for r in range(target) if r not in done]
    if progress:
        print(f"{len(tasks)} replicates to run in {len(grid)} cells", flush=True)
    t0 = time.time()
    handles: dict[str, Any] = {}
    try:
        if workers <= 1:
            iterator = (_worker(t) for t in tasks)
        else:
            import multiprocessing as mp

            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
            futures = [pool.submit(_worker, t) for t in tasks]
            iterator = (f.result() for f in as_completed(futures))
        for i, (cell_id, row) in enumerate(iterator, 1):
            fh = handles.get(cell_id)
            if fh is None:
                fh = handles[cell_id] = (out / f"{cell_id}.jsonl").open("a", encoding="utf-8")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if progress and (i % 50 == 0 or i == len(tasks)):
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        for fh in handles.values():
            fh.close()
        if workers > 1 and tasks:
            pool.shutdown(cancel_futures=True)


# ============================================================================ summary
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def classify(cov: float, se_sd: float) -> str:
    """Pre-registered thresholds (docs/PHASE2_PREREGISTRATION.md section 4)."""
    if not (np.isfinite(cov) and np.isfinite(se_sd)):
        return "undefined"
    if cov < 0.93 or se_sd < 0.85:
        return "anticonservative"
    if cov > 0.97 or se_sd > 1.15:
        return "conservative"
    return "calibrated"


def summarize_rows(rows: list[dict[str, Any]], cell_id: str, rng_seed: int = 0) -> list[dict[str, Any]]:
    n_total = len(rows)
    ok_rows = [r for r in rows if r.get("fit_ok")]
    out = []
    if not ok_rows:
        return [{"cell": cell_id, "method": "-", "param": -1, "n_rep": n_total, "fit_fail_rate": 1.0}]
    p = len(ok_rows[0]["truth"])
    methods = sorted({m for r in ok_rows for m in r["methods"]}, key=ALL_METHODS.index)
    rng = np.random.default_rng(rng_seed)
    for m in methods:
        for j in range(p):
            est, se, lo, hi, nfail, secs, errs, fixed = [], [], [], [], [], [], 0, False
            for r in ok_rows:
                rec = r["methods"].get(m)
                if rec is None:
                    continue
                if "error" in rec:
                    errs += 1
                    continue
                fixed = fixed or bool(rec["fixed"][j])
                est.append(rec["est"][j]); se.append(rec["se"][j]); lo.append(rec["lo"][j]); hi.append(rec["hi"][j])
                nfail.append(rec["n_fail"] if rec["n_fail"] is not None else 0); secs.append(rec["seconds"])
            truth = float(ok_rows[0]["truth"][j])
            est, se, lo, hi = map(lambda a: np.asarray(a, dtype=float), (est, se, lo, hi))
            n = est.size
            row: dict[str, Any] = {"cell": cell_id, "method": m, "param": j, "truth": truth, "n_rep": n_total,
                                   "n_used": n, "fit_fail_rate": 1 - len(ok_rows) / n_total, "method_errors": errs,
                                   "fixed": fixed}
            if n == 0 or fixed:
                out.append(row)
                continue
            finite = np.isfinite(lo) & np.isfinite(hi)
            sd = float(np.std(est, ddof=1)) if n > 1 else math.nan
            bias = float(np.mean(est) - truth)
            miss_left = int(np.sum(finite & (truth < lo)))
            miss_right = int(np.sum(finite & (truth > hi)))
            covered = int(np.sum(finite & (lo <= truth) & (truth <= hi)))
            n_def = int(np.sum(finite))
            se_f = se[np.isfinite(se)]
            # SE/SD with a percentile bootstrap CI over replicates (2000 resamples)
            ratio = float(np.mean(se_f) / sd) if se_f.size and sd > 0 else math.nan
            if n > 2 and se_f.size == n and sd > 0:
                idx = rng.integers(0, n, size=(2000, n))
                rb = se[idx].mean(axis=1) / est[idx].std(axis=1, ddof=1)
                r_lo, r_hi = (float(v) for v in np.percentile(rb, [2.5, 97.5]))
            else:
                r_lo = r_hi = math.nan
            cov = covered / n_def if n_def else math.nan
            w_lo, w_hi = wilson(covered, n_def)
            row.update({
                "bias": bias, "sd": sd, "bias_over_sd": bias / sd if sd > 0 else math.nan,
                "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
                "se_mean": float(np.mean(se_f)) if se_f.size else math.nan,
                "se_median": float(np.median(se_f)) if se_f.size else math.nan,
                "se_undefined": int(n - se_f.size),
                "se_over_sd": ratio, "se_over_sd_lo": r_lo, "se_over_sd_hi": r_hi,
                "coverage": cov, "coverage_lo": w_lo, "coverage_hi": w_hi, "n_ci_defined": n_def,
                "miss_left": miss_left, "miss_right": miss_right,
                "width_median": float(np.median((hi - lo)[finite])) if n_def else math.nan,
                "boot_fail_mean": float(np.mean(nfail)) if nfail else math.nan,
                "seconds_mean": float(np.mean(secs)) if secs else math.nan,
                "class": classify(cov, ratio),
                "relevant_bias": bool(sd > 0 and abs(bias) / sd > 0.25),
                "asymmetric": bool((miss_left + miss_right) >= 10
                                   and max(miss_left, miss_right) >= 3 * max(1, min(miss_left, miss_right))),
            })
            out.append(row)
    return out


def summarize(out: Path, csv: Path | None = None) -> list[dict[str, Any]]:
    import pandas as pd

    rows_all: list[dict[str, Any]] = []
    for path in sorted(out.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn last line of a run in progress
        if rows:
            rows_all += summarize_rows(rows, path.stem)
    if csv is not None:
        pd.DataFrame(rows_all).to_csv(csv, index=False, float_format="%.6g")
    return rows_all


# ============================================================================ CLI
def _load_grid(path: str) -> list[dict[str, Any]]:
    name = str(path)
    if name in ("primary", "diagnostic", "smoke"):
        return {"primary": primary_grid, "diagnostic": diagnostic_grid, "smoke": smoke_grid}[name]()
    return json.loads(Path(path).read_text(encoding="utf-8"))["cells"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m validation.coverage_campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--write", required=True)
    r = sub.add_parser("run")
    r.add_argument("grid", help="grid JSON, or one of: primary, diagnostic, smoke")
    r.add_argument("out")
    r.add_argument("--select", default=None, help="regex on cell id")
    r.add_argument("--tier", type=int, default=None)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--n-rep", type=int, default=None)
    r.add_argument("--methods", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("out")
    s.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "grid":
        cells = primary_grid() + diagnostic_grid()
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(json.dumps({"base_seed": BASE_SEED, "cells": cells}, indent=1), encoding="utf-8")
        print(f"{len(cells)} cells -> {a.write}")
    elif a.cmd == "run":
        grid = _load_grid(a.grid)
        if a.select:
            grid = [c for c in grid if re.search(a.select, c["id"])]
        if a.tier is not None:
            grid = [c for c in grid if int(c.get("tier", 0)) == a.tier]
        run(grid, Path(a.out), workers=a.workers, n_rep=a.n_rep,
            methods=a.methods.split(",") if a.methods else None)
    else:
        rows = summarize(Path(a.out), Path(a.csv) if a.csv else None)
        print(f"{len(rows)} summary rows")


if __name__ == "__main__":
    main()
