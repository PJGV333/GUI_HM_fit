"""Deterministic synthetic validation cases for Phase 1 (known true parameters).

Each case returns a dict with: case, technique, true, estimate, se, dof, seed, noise.
Used by tests/test_phase1_validation.py and by ``python tests/phase1_cases.py`` (prints a
Markdown table for docs/STABILITY_PHASE1_REPORT.md).
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ----------------------------------------------------------------------------- spectro
def _spectro_dataset(path, k_true, model, n_points=10, noise=2e-4, seed=300, g_max=2e-3):
    from hmfit_core.solvers import NewtonRaphson

    wl = np.linspace(240.0, 600.0, 31)
    c = np.column_stack([np.full(n_points, 1e-3), np.linspace(0.0, g_max, n_points)])
    M = np.asarray(model, dtype=float).T  # (n_comp, n_species)
    solver = NewtonRaphson(c, M, nas=[], model_sett="Free")
    C, _ = solver.concentraciones(np.asarray(k_true, dtype=float))
    centers = np.linspace(265.0, 380.0, C.shape[1])
    eps = np.vstack([1.2e3 * np.exp(-((wl - c0) ** 2) / (2 * 25.0**2)) for c0 in centers])
    y = C @ eps
    if noise > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, y.shape)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(y.T, index=wl, columns=[f"p{i+1}" for i in range(n_points)]).to_excel(w, sheet_name="Spectra")
        pd.DataFrame({"H": c[:, 0], "G": c[:, 1]}).to_excel(w, sheet_name="Conc", index=False)


def _spectro_cfg(path, model, k0, bounds):
    return {
        "file_path": str(path), "spectra_sheet": "Spectra", "conc_sheet": "Conc",
        "column_names": ["H", "G"], "receptor_label": "H", "guest_label": "G",
        "efa_enabled": False, "efa_eigenvalues": 0, "modelo": model, "non_abs_species": [],
        "algorithm": "Newton-Raphson", "model_settings": "Free", "optimizer": "powell",
        "initial_k": k0, "bounds": bounds, "channels_raw": "All", "channels_mode": "all",
        "channels_resolved": [], "show_stability_diagnostics": False, "multi_start_runs": 1,
    }


def spectro_case(name, k_true, model, k0, bounds, noise=2e-4, seed=300, g_max=2e-3):
    from hmfit_core import run_spectroscopy

    path = os.path.join(tempfile.mkdtemp(), "s.xlsx")
    _spectro_dataset(path, k_true, model, noise=noise, seed=seed, g_max=g_max)
    out = _quiet(run_spectroscopy, _spectro_cfg(path, model, k0, bounds))
    ed = out["export_data"]
    return {
        "case": name, "technique": "UV-Vis", "true": list(k_true), "estimate": list(ed["k"]),
        "se": list(ed["SE_log10K"]), "dof": None, "seed": seed, "noise": f"N(0,{noise:g}) abs",
        "notes": list(ed.get("uncertainty_notes") or []),
    }


# ----------------------------------------------------------------------------- NMR
def nmr_case(name, k_true, k0, bounds, fixed=None, noise=2e-3, seed=4, model=None, n=12):
    from hmfit_core.processors.nmr_processor import process_nmr_data
    from hmfit_core.solvers import NewtonRaphson

    model = model or [[1, 0], [0, 1], [1, 1]]
    M = np.asarray(model, dtype=float).T
    H = np.full(n, 1e-3)
    G = np.linspace(0.0, 5e-3, n)
    solver = NewtonRaphson(np.column_stack([H, G]), M, nas=[1], model_sett="Free")
    C, Co = solver.concentraciones(np.asarray(k_true, dtype=float))
    # host-bound fraction for each host-containing species -> shift
    host_species = [i for i, row in enumerate(model) if row[0] > 0]
    frac = Co[:, host_species] * np.asarray([model[i][0] for i in host_species]) / H[:, None]
    deltas1 = np.linspace(7.0, 7.6, len(host_species))
    deltas2 = np.linspace(8.0, 7.5, len(host_species))
    rng = np.random.default_rng(seed)
    s1 = frac @ deltas1 + rng.normal(0, noise, n)
    s2 = frac @ deltas2 + rng.normal(0, noise, n)
    path = os.path.join(tempfile.mkdtemp(), "n.xlsx")
    pd.DataFrame({"H": H, "G": G, "s1": s1, "s2": s2}).to_excel(path, index=False, sheet_name="S")
    out = _quiet(
        process_nmr_data, path, "S", "S", ["H", "G"], ["s1", "s2"], "H", "G", model,
        list(k0), bounds, "Newton-Raphson", "powell", "Free", [1], k_fixed=fixed,
    )
    ed = out["export_data"]
    return {
        "case": name, "technique": "1H NMR", "true": list(k_true), "estimate": list(ed["k"]),
        "se": list(ed["SE_log10K"]), "dof": None, "seed": seed, "noise": f"N(0,{noise:g}) ppm",
    }


# ----------------------------------------------------------------------------- acid-base
def acid_base_nmr_case(name, pka_true, pka0, noise=3e-3, seed=0, n=25, ph=(2.0, 10.0), fit_mask=None):
    from hmfit_core.acid_base import distribution_fractions_from_pH, pka_to_log_beta
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from hmfit_core.run_acid_base import run_acid_base

    pH = np.linspace(ph[0], ph[1], n)
    frac = distribution_fractions_from_pH(pH, pka_to_log_beta(list(pka_true)))
    shifts = np.linspace(7.9, 7.0, frac.shape[1])
    y = frac @ shifts + np.random.default_rng(seed).normal(0.0, noise, n)
    path = os.path.join(tempfile.mkdtemp(), "ab.csv")
    pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
    cfg = {"file_path": path, "data_type": "nmr", "pka_initial": ",".join(str(v) for v in pka0),
           "analyte_concentration": 1e-3, "base_charge": -len(pka_true)}
    if fit_mask is not None:
        cfg["pka_fit_mask"] = fit_mask
    res = run_acid_base(cfg, progress_cb=lambda _m: None)
    out = compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    P = out["export_frames"]["Parameters"]
    corr = np.asarray(out["corr"], dtype=float)
    return {
        "case": name, "technique": "acid-base NMR", "true": list(pka_true),
        "estimate": P["Estimate"].astype(float).tolist(), "se": pd.to_numeric(P["SE"], errors="coerce").tolist(),
        "dof": out["dof"], "seed": seed, "noise": f"N(0,{noise:g}) ppm",
        "max_abs_corr": float(np.nanmax(np.abs(corr - np.eye(corr.shape[0])))) if corr.size > 1 else 0.0,
    }


def potentiometry_case(name, pka_true, pka0, noise=2e-3, seed=11):
    from hmfit_core.acid_base import make_simple_acid_base_system
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
    from hmfit_core.run_acid_base import run_acid_base

    system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=list(pka_true),
                                          base_charge=-len(pka_true))
    vol = np.linspace(0.0, 10.0 * len(pka_true) * 1.8, 60)
    exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                  titrant_concentration=1e-2, titrant_type="base")
    pH = simulate_pH_titration(exp, system) + np.random.default_rng(seed).normal(0.0, noise, vol.size)
    path = os.path.join(tempfile.mkdtemp(), "pot.csv")
    pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
    res = run_acid_base({"file_path": path, "data_type": "potentiometry",
                         "pka_initial": ",".join(str(v) for v in pka0), "analyte_concentration": 1e-2,
                         "titrant_concentration": 1e-2, "initial_volume": 10.0, "titrant_type": "base",
                         "base_charge": -len(pka_true)}, progress_cb=lambda _m: None)
    out = compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    P = out["export_frames"]["Parameters"]
    return {
        "case": name, "technique": "potentiometry", "true": list(pka_true),
        "estimate": P["Estimate"].astype(float).tolist(), "se": pd.to_numeric(P["SE"], errors="coerce").tolist(),
        "dof": out["dof"], "seed": seed, "noise": f"N(0,{noise:g}) pH",
    }


# ----------------------------------------------------------------------------- kinetics
def kinetics_case(name, mech, k_true, k0, y0, t, n_wl=12, noise=1e-2, seed=0, scale=1.0, fixed=None):
    from hmfit.kinetics.data.dataset import KineticsDataset
    from hmfit.kinetics.fit.objective import GlobalKineticsObjective
    from hmfit.kinetics.fit.optimizer import fit_global
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel

    model = KineticsModel(parse_mechanism(mech))
    names = list(k_true.keys())
    C = model.solve_concentrations(t, y0, k_true)
    A = np.random.default_rng(1).random((C.shape[1], n_wl)) * scale
    D = C @ A
    if noise > 0:
        D = D + np.random.default_rng(seed).normal(0.0, noise, D.shape)
    obj = GlobalKineticsObjective(model, [KineticsDataset(t=t, y0=y0, D=D)], param_names=names)
    res = fit_global(obj, np.asarray(k0, dtype=float), bounds=(0.0, np.inf), fixed_mask=fixed)
    return {
        "case": name, "technique": "kinetics", "true": [k_true[n] for n in names],
        "estimate": [res.params[n] for n in names], "se": list(res.errors), "dof": res.result.dof,
        "seed": seed, "noise": f"N(0,{noise:g})",
    }


FIRST_ORDER = "species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"
SECOND_ORDER = "species: A, B, C\nA + B -> C ; k1\n"


def all_cases():
    t1 = np.linspace(0.0, 4.0, 40)
    t2 = np.linspace(0.0, 20.0, 60)
    y01 = {"A": 1.0, "B": 0.0, "C": 0.0}
    y02 = {"A": 1e-4, "B": 2e-4, "C": 0.0}
    m11 = [[1, 0], [0, 1], [1, 1]]
    m12 = [[1, 0], [0, 1], [1, 1], [1, 2]]
    return [
        spectro_case("UV-Vis 1:1, log K = 3", [3.0], m11, [2.5], [(0.0, 6.0)]),
        spectro_case("UV-Vis 1:1, no noise", [3.0], m11, [2.5], [(0.0, 6.0)], noise=0.0),
        spectro_case("UV-Vis 1:2, log b2 fixed", [4.0, 7.0], m12, [3.5, 7.0], [(0.0, 8.0), (7.0, 7.0)], g_max=4e-3),
        nmr_case("NMR 1:1, log K = 3", [3.0], [2.5], [{"min": 0, "max": 8}]),
        nmr_case("NMR 1:2, log b2 fixed", [4.0, 7.0], [3.5, 7.0], [{"min": 0, "max": 8}, {"min": 7.0, "max": 7.0}],
                 fixed=[False, True], model=m12, n=15),
        # HM fit convention: pKa_n = log beta_n - log beta_(n-1), so pKa1 belongs to HL <=> L + H+
        # (the most basic site) and a physical diprotic ligand has pKa1 > pKa2. The Phase 1
        # versions of these three cases used the inverted order (cooperative protonation).
        acid_base_nmr_case("Acid-base NMR, pKa 7.5/4.0", [7.5, 4.0], [7.2, 4.3]),
        acid_base_nmr_case("Acid-base NMR, close pKa 4.4/4.0", [4.4, 4.0], [4.6, 3.8]),
        acid_base_nmr_case("Acid-base NMR, pKa1 fixed", [9.0, 4.0], [9.0, 4.2], ph=(2.0, 7.0), n=15,
                           fit_mask=[False, True]),
        potentiometry_case("Potentiometry, pKa 5.0", [5.0], [4.4]),
        kinetics_case("Kinetics 1st order A->B->C", FIRST_ORDER, {"k1": 1.2, "k2": 0.4}, [1.0, 0.5], y01, t1),
        kinetics_case("Kinetics 1st order, no noise", FIRST_ORDER, {"k1": 1.2, "k2": 0.4}, [1.0, 0.5], y01, t1, noise=0.0),
        kinetics_case("Kinetics 1st order, k2 fixed", FIRST_ORDER, {"k1": 1.2, "k2": 0.4}, [1.0, 0.4], y01, t1,
                      fixed=[False, True]),
        kinetics_case("Kinetics 2nd order, k = 5e3 M^-1 s^-1", SECOND_ORDER, {"k1": 5e3}, [3e3], y02, t2,
                      noise=2e-3, scale=1e4),
    ]


def markdown_table(cases):
    lines = ["| Caso | Técnica | Verdadero | Estimado | SE | |Δ|/SE | g.l. | Semilla / ruido |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cases:
        est = np.asarray(c["estimate"], float)
        tru = np.asarray(c["true"], float)
        se = np.asarray(c["se"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, np.abs(est - tru) / se, np.nan)
        fmt = lambda a: ", ".join(f"{v:.4g}" for v in a)
        lines.append(
            f"| {c['case']} | {c['technique']} | {fmt(tru)} | {fmt(est)} | {fmt(se)} | "
            f"{', '.join('fijo' if s == 0 else f'{v:.2f}' for v, s in zip(z, se))} | "
            f"{c['dof'] if c['dof'] is not None else '—'} | {c['seed']} / {c['noise']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table(all_cases()))
