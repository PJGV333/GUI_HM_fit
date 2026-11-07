# SPDX-License-Identifier: GPL-3.0-or-later
"""Literature benchmark: HM fit against published HypNMR results (Roelens/Vacca 2009).

Cases live in ``tests/data/literature/<case>/`` (see SOURCES.md there). Every fit
goes through the public entry point the GUI uses (``hmfit_core.api.run_nmr_fit``),
with the configuration the NMR tab builds; the analytic error analysis of the
Errors tab is reproduced with the same core call and arguments
(``compute_errors_nmr_varpro``, as in ``ModelOptPlotsWidget._compute_errors_from_context``)
to obtain the unit-weight sigma, the correlations and the limiting shifts.

Pre-registered criteria and their justification: ``docs/LITERATURE_BENCHMARK.md``.

Commands::

    python -m validation.literature_benchmark run [--json OUT] [--md OUT]
    python -m validation.literature_benchmark sensitivity [--n 50] [--workers 30] [--json OUT]
    python -m validation.literature_benchmark baseline [--write]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data" / "literature"

# Starting values (pre-registered): refined log beta = reference + START_OFFSET,
# fixed constants at their printed value; no bounds; GUI defaults otherwise.
START_OFFSET = 0.2

# ----------------------------------------------------------------------------- criteria
# Pre-registered in docs/LITERATURE_BENCHMARK.md (stage B). Do not edit without a
# justified amendment committed BEFORE comparing new results.
CRITERIA = {
    "L1_rms_rel": 0.03,          # |RMS_hm / RMS_ref - 1|
    "L2_abs_factor": 0.25,       # |dlogb| <= max(0.25 * SD_ref, 0.005)
    "L2_abs_floor": 0.005,
    "L3_se_rel": 0.15,           # |SE_hm / SD_ref - 1|
    "L4_corr_abs": 0.05,         # |r_hm - r_ref|
}


def case_ids(kind: str = "fit") -> list[str]:
    out = []
    for p in sorted(DATA_DIR.iterdir()):
        if (p / "case.json").is_file():
            c = json.loads((p / "case.json").read_text())
            if c.get("kind") == kind:
                out.append(p.name)
    return out


def load_case(cid: str) -> dict[str, Any]:
    d = DATA_DIR / cid
    case = json.loads((d / "case.json").read_text())
    with (d / "data.csv").open() as fh:
        rows = list(csv.reader(fh))
    case["_header"] = rows[0]
    case["_rows"] = rows[1:]
    bl = d / "baseline.json"
    case["_baseline"] = json.loads(bl.read_text()) if bl.is_file() else None
    tl = d / "tolerances.json"
    case["_tolerances"] = json.loads(tl.read_text()) if tl.is_file() else None
    return case


def tolerances_from_sensitivity(case, sens_case) -> dict[str, Any]:
    """Pre-registered rule (stage B): each tolerance is the proposed one unless the
    rounding dispersion exceeds it, in which case it becomes 3 x that dispersion."""
    ref = case["reference"]["log_beta"]
    out: dict[str, Any] = {"rule": "proposed tolerance, or 3 x rounding dispersion if larger",
                           "L1_rms_rel": CRITERIA["L1_rms_rel"], "L2_abs": {}, "L3_se_rel": {},
                           "widened": []}
    cv = sens_case["cv_rms"]
    if cv > CRITERIA["L1_rms_rel"]:
        out["L1_rms_rel"] = 3 * cv
        out["widened"].append(f"L1: cv(RMS) = {cv:.4f} > {CRITERIA['L1_rms_rel']}")
    for name, free, sd, cvse in zip(sens_case["names"], sens_case["free"], sens_case["sd_log_beta"],
                                    sens_case["cv_se"]):
        if not free:
            continue
        t2 = max(CRITERIA["L2_abs_factor"] * ref[name]["sd"], CRITERIA["L2_abs_floor"])
        if sd > t2:
            out["widened"].append(f"L2 {name}: SD = {sd:.5f} > {t2:.5f}")
            t2 = 3 * sd
        out["L2_abs"][name] = t2
        t3 = CRITERIA["L3_se_rel"]
        if cvse > t3:
            out["widened"].append(f"L3 {name}: cv(SE) = {cvse:.4f} > {t3}")
            t3 = 3 * cvse
        out["L3_se_rel"][name] = t3
    return out


def supported(case) -> tuple[bool, str]:
    for n in case["nuclei"]:
        for sp, st in n["limiting_shifts"].items():
            if st["status"] == "fixed":
                return False, ("fixed limiting shifts are not supported by HM fit "
                               f"({n['name']} in {sp}); docs/LITERATURE_BENCHMARK.md §6")
            if st["status"] == "absent":
                return False, f"absent limiting shift ({n['name']} in {sp}) not representable"
    return True, ""


# ----------------------------------------------------------------------------- model
def model_matrices(case):
    """Species x reagent matrix (free reagents first, then the complexes in the
    order of the results page) and the species x nucleus stoichiometry map the
    NMR tab injects (column of the nucleus' parent reagent)."""
    reagents = case["reagents"]
    nr = len(reagents)
    M = [[int(i == j) for j in range(nr)] for i in range(nr)]
    M += [list(s["stoich"]) for s in case["species"]]
    S = [[row[reagents.index(n["reagent"])] for n in case["nuclei"]] for row in M]
    return M, S


def numeric_table(case, perturb: dict[str, Any] | None = None):
    """(conc[m, nr], shifts[m, nn]) as floats; ``perturb`` = {"rng": Generator} applies
    uniform rounding perturbations within half a unit of the last printed digit."""
    nr = len(case["reagents"])
    rows = case["_rows"]
    conc = np.zeros((len(rows), nr))
    shifts = np.full((len(rows), len(case["nuclei"])), np.nan)
    rng = perturb["rng"] if perturb else None
    salt_draw: dict[str, float] = {}
    # salt columns (cation and anion of TMAC): one physical quantity, so one draw per
    # printed value shared by both columns
    if nr == 3:
        salt_cols = {1, 2}
    elif case["reagents"] == ["Q", "X"]:
        salt_cols = {0, 1}
    else:
        salt_cols = {1}
    for i, r in enumerate(rows):
        for j in range(nr):
            s = r[1 + j]
            v = float(s)
            if rng is not None and v != 0.0:
                half = _half_ulp(s)
                if j in salt_cols:
                    key = s
                    if key not in salt_draw:
                        salt_draw[key] = rng.uniform(-half, half)
                    v += salt_draw[key]
                else:
                    v += rng.uniform(-half, half)
            conc[i, j] = v
        for j in range(len(case["nuclei"])):
            s = r[1 + nr + j]
            if s != "":
                v = float(s)
                if rng is not None:
                    v += rng.uniform(-_half_ulp(s), _half_ulp(s))
                shifts[i, j] = v
    return conc, shifts


def _half_ulp(s: str) -> float:
    exp = Decimal(s).as_tuple().exponent
    return 0.5 * 10.0 ** int(exp)


def gui_config(case, xlsx_path: str, conc, shifts, start: str = "offset") -> dict[str, Any]:
    """Write the workbook the NMR tab would read and build the same config dict
    (``NMRTab._collect_config`` + ``_inject_stoichiometry_map``)."""
    import pandas as pd

    reagents = case["reagents"]
    cols = [f"T_{r}" for r in reagents]
    sig = [n["name"] for n in case["nuclei"]]
    df = pd.DataFrame(conc, columns=cols)
    for j, s in enumerate(sig):
        df[s] = shifts[:, j]
    df.to_excel(xlsx_path, index=False, sheet_name="data")
    M, S = model_matrices(case)
    ref = case["reference"]["log_beta"]
    k0, fixed = [], []
    for sp in case["species"]:
        if sp["status"] == "fixed":
            k0.append(float(sp["log_beta"]))
            fixed.append(True)
        else:
            lb = float(ref[sp["name"]]["log_beta"])
            k0.append(lb + (START_OFFSET if start == "offset" else 0.0))
            fixed.append(False)
    return {
        "file_path": xlsx_path, "nmr_sheet": "data", "conc_sheet": "data",
        "column_names": cols, "signal_names": sig,
        "signal_data": [{"col_name": s, "parent": f"T_{n['reagent']}"} for s, n in zip(sig, case["nuclei"])],
        "receptor_label": cols[0], "guest_label": cols[-1],
        "modelo": M, "non_abs_species": [], "algorithm": "Newton-Raphson",
        "model_settings": "Free", "optimizer": "powell",
        "initial_k": k0, "bounds": [{"min": None, "max": None} for _ in k0],
        "fixed_mask": fixed, "k_fixed": fixed, "show_stability_diagnostics": False,
        "multi_start_runs": 1, "multi_start_parallel": False, "multi_start_max_workers": None,
        "equation_text": "", "stoichiometry_map": S,
    }


def errors_tab_analytic(result, cfg) -> dict[str, Any]:
    """The Errors tab 'analytic' computation (unit weights) at the fitted constants,
    with the arguments of ModelOptPlotsWidget._compute_errors_from_context."""
    from hmfit_core.processors.nmr_processor import build_D_cols
    from hmfit_core.solvers import NewtonRaphson
    from hmfit_core.utils.errors import compute_errors_nmr_varpro, correlation_from_covariance

    ed = result["export_data"]
    k_hat = np.asarray(ed["k"], dtype=float)
    C_T = np.asarray(ed["C_T"], dtype=float)
    dq = np.asarray(ed["Chemical_Shifts"], dtype=float)
    modelo = np.asarray(ed["modelo"], dtype=float).T
    fixed_mask = np.asarray(ed["fixed_mask"], dtype=bool)
    stoich = np.asarray(ed["stoichiometry_map"], dtype=float)
    res = NewtonRaphson(C_T, modelo, list(ed.get("non_absorbent_species") or []), "Free")
    D_cols, _ = build_D_cols(C_T, list(ed["column_names"]), list(ed["signal_names"]), default_idx=0)
    mask = np.isfinite(dq) & np.isfinite(D_cols) & (np.abs(D_cols) > 0)
    m = compute_errors_nmr_varpro(
        k_hat, res, dq, D_cols, modelo, [], stoichiometry=stoich, mask=mask, fixed_mask=fixed_mask,
        rcond=1e-10, rcond_cov=1e-10, ridge=1e-8, ridge_cov=0.0, use_projector=True,
        param_names=[f"K{i + 1}" for i in range(k_hat.size)],
    )
    p = k_hat.size
    free_idx = np.asarray(m["free_idx"], dtype=int)
    cov = np.zeros((p, p))
    cf = np.asarray(m["Cov_log10K_free"], dtype=float)
    if cf.size:
        cov[np.ix_(free_idx, free_idx)] = cf
    corr = correlation_from_covariance(cov, m.get("nonidentifiable"))
    return {"se": np.asarray(m["SE_log10K"], float), "corr": corr, "s2": float(m["covfit"]),
            "dof": int(m["dof"]), "nobs": int(m["nobs"]), "n_linear": int(m["n_linear"]),
            "rms": float(m["rms"]), "coef": np.asarray(m["coef"], float)}


def fit_case(case, *, perturb_seed: int | None = None, start: str = "offset",
             fix_all_at_reference: bool = False, with_errors_tab: bool = True) -> dict[str, Any]:
    from hmfit_core.api import run_nmr_fit

    rng = np.random.default_rng(perturb_seed) if perturb_seed is not None else None
    conc, shifts = numeric_table(case, {"rng": rng} if rng is not None else None)
    with tempfile.TemporaryDirectory() as td:
        cfg = gui_config(case, os.path.join(td, "case.xlsx"), conc, shifts,
                         start="reference" if fix_all_at_reference else start)
        if fix_all_at_reference:
            cfg["fixed_mask"] = cfg["k_fixed"] = [True] * len(cfg["initial_k"])
        t0 = time.perf_counter()
        result = run_nmr_fit(cfg, progress_cb=lambda _m: None)
        elapsed = time.perf_counter() - t0
    if not result.get("success", False):
        return {"success": False, "error": result.get("error"), "elapsed_s": elapsed}
    ed = result["export_data"]
    stats = dict((r[0], r[1]) for r in ed["stats_table"])
    names = [s["name"] for s in case["species"]]
    out = {
        "success": True,
        "elapsed_s": elapsed,
        "objective_evaluations": int(result.get("objective_evaluations") or 0),
        "names": names,
        "log_beta": [float(x) for x in ed["k"]],
        "se": [None if c["SE_log10K"] is None else float(c["SE_log10K"]) for c in result["constants"]],
        "fixed": [bool(c["fixed"]) for c in result["constants"]],
        "rms": float(stats["RMS"]),
        "stability_status": result.get("stability_status"),
    }
    if with_errors_tab:
        et = errors_tab_analytic(result, cfg)
        M, S = model_matrices(case)
        species_names = list(case["reagents"]) + names
        shifts_out = {}
        for j, n in enumerate(case["nuclei"]):
            shifts_out[n["name"]] = {sp: float(et["coef"][i, j]) for i, sp in enumerate(species_names)
                                     if S[i][j] > 0}
        out["errors_tab"] = {"se": et["se"].tolist(), "sigma": math.sqrt(et["s2"]) if et["s2"] > 0 else None,
                             "dof": et["dof"], "nobs": et["nobs"], "n_linear": et["n_linear"],
                             "corr": et["corr"].tolist()}
        out["limiting_shifts"] = shifts_out
    return out


# ----------------------------------------------------------------------------- comparison
def compare(case, fit) -> dict[str, Any]:
    """Levels 1-4 against the literature reference (pre-registered tolerances)."""
    ref = case["reference"]
    c = dict(CRITERIA)
    tolc = case.get("_tolerances") or {}
    if "L1_rms_rel" in tolc:
        c["L1_rms_rel"] = tolc["L1_rms_rel"]
    out: dict[str, Any] = {"levels": {}}
    rms_rel = fit["rms"] / ref["rms_weighted_residual"] - 1.0
    out["levels"]["L1"] = {"pass": abs(rms_rel) <= c["L1_rms_rel"], "rms_hm": fit["rms"],
                           "rms_ref": ref["rms_weighted_residual"], "rel": rms_rel}
    if fit.get("errors_tab", {}).get("sigma") is not None:
        out["levels"]["L1"]["sigma_hm"] = fit["errors_tab"]["sigma"]
        out["levels"]["L1"]["sigma_ref"] = ref["sigma"]
        out["levels"]["L1"]["sigma_rel"] = fit["errors_tab"]["sigma"] / ref["sigma"] - 1.0
    l2, l3 = [], []
    for name, lb, se, fx in zip(fit["names"], fit["log_beta"], fit["se"], fit["fixed"]):
        r = ref["log_beta"][name]
        if r["status"] == "fixed" or fx:
            continue
        tol2 = (tolc.get("L2_abs") or {}).get(name, max(c["L2_abs_factor"] * r["sd"], c["L2_abs_floor"]))
        tol3 = (tolc.get("L3_se_rel") or {}).get(name, c["L3_se_rel"])
        d = lb - r["log_beta"]
        l2.append({"name": name, "hm": lb, "ref": r["log_beta"], "delta": d, "tol": tol2,
                   "pass": abs(d) <= tol2})
        rel = (se / r["sd"] - 1.0) if (se is not None and r["sd"]) else float("nan")
        l3.append({"name": name, "hm": se, "ref": r["sd"], "rel": rel, "tol": tol3,
                   "pass": bool(np.isfinite(rel) and abs(rel) <= tol3)})
    out["levels"]["L2"] = {"pass": all(x["pass"] for x in l2), "items": l2}
    out["levels"]["L3"] = {"pass": all(x["pass"] for x in l3), "items": l3}
    # level 4 (reported, not used for the classification)
    l4s = []
    for nuc, per in (fit.get("limiting_shifts") or {}).items():
        for sp, v in per.items():
            r = ref["limiting_shifts"].get(nuc, {}).get(sp)
            if r is None or r.get("error") is None:
                continue
            l4s.append({"nucleus": nuc, "species": sp, "hm": v, "ref": r["value"], "err": r["error"],
                        "pass": abs(v - r["value"]) <= r["error"]})
    l4c = []
    cref = ref.get("correlation_x1000")
    if cref and fit.get("errors_tab"):
        idx = {n: i for i, n in enumerate(fit["names"])}
        corr = np.asarray(fit["errors_tab"]["corr"], float)
        for i, row in enumerate(cref["lower_x1000"]):
            for j, v in enumerate(row):
                a, b = idx[cref["names"][i]], idx[cref["names"][j]]
                hm = corr[a, b]
                l4c.append({"pair": f"{cref['names'][i]}/{cref['names'][j]}", "hm": hm, "ref": v / 1000.0,
                            "pass": bool(np.isfinite(hm) and abs(hm - v / 1000.0) <= c["L4_corr_abs"])})
    out["levels"]["L4"] = {"shifts_pass": sum(x["pass"] for x in l4s), "shifts_n": len(l4s),
                           "corr_pass": sum(x["pass"] for x in l4c), "corr_n": len(l4c),
                           "shifts": l4s, "corr": l4c}
    L = out["levels"]
    if not L["L1"]["pass"]:
        cls = "no reproduce"
    elif L["L2"]["pass"] and L["L3"]["pass"]:
        cls = "reproduce"
    else:
        cls = "parcial"
    out["classification"] = cls
    return out


def forward_check(case) -> dict[str, Any]:
    """Level 0 (diagnostic): all constants fixed at the reference values; HM fit
    solves only the limiting shifts. Tests the transcription and the model
    specification independently of the optimiser."""
    fit = fit_case(case, fix_all_at_reference=True, with_errors_tab=False)
    if not fit.get("success"):
        return {"success": False, "error": fit.get("error")}
    rref = case["reference"]["rms_weighted_residual"]
    return {"success": True, "rms_hm": fit["rms"], "rms_ref": rref, "rel": fit["rms"] / rref - 1.0}


# ----------------------------------------------------------------------------- provenance
def environment() -> dict[str, Any]:
    import numpy
    import scipy

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    try:
        import pandas
        pd_v = pandas.__version__
    except Exception:
        pd_v = None
    return {"commit": commit, "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "pandas": pd_v, "platform": platform.platform()}


# ----------------------------------------------------------------------------- sensitivity
def _sens_one(args):
    cid, seed = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    case = load_case(cid)
    f = fit_case(case, perturb_seed=seed, with_errors_tab=False)
    return cid, seed, f


def sensitivity(cids, n: int, workers: int, base_seed: int = 20261107) -> dict[str, Any]:
    """Rounding sensitivity: n refits per case with uniform perturbations within half
    a unit of the last printed digit. Reports ONLY dispersions (no reference values)."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(cid, base_seed + 1000 * k + i) for k, cid in enumerate(cids) for i in range(n)]
    results: dict[str, list] = {cid: [] for cid in cids}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for cid, seed, f in ex.map(_sens_one, jobs, chunksize=1):
            results[cid].append(f)
    out = {}
    for cid, fs in results.items():
        ok = [f for f in fs if f.get("success")]
        if not ok:
            out[cid] = {"n": len(fs), "n_ok": 0}
            continue
        lb = np.array([f["log_beta"] for f in ok], float)
        se = np.array([[np.nan if s is None else s for s in f["se"]] for f in ok], float)
        rms = np.array([f["rms"] for f in ok], float)
        free = ~np.array(ok[0]["fixed"], bool)
        out[cid] = {
            "n": len(fs), "n_ok": len(ok), "names": ok[0]["names"], "free": free.tolist(),
            "sd_log_beta": lb.std(axis=0, ddof=1).tolist(),
            "range_log_beta": (lb.max(axis=0) - lb.min(axis=0)).tolist(),
            "cv_se": (se.std(axis=0, ddof=1) / np.nanmean(se, axis=0)).tolist(),
            "cv_rms": float(rms.std(ddof=1) / rms.mean()),
            "median_elapsed_s": float(np.median([f["elapsed_s"] for f in ok])),
        }
    return {"n": n, "base_seed": base_seed, "environment": environment(), "cases": out}


# ----------------------------------------------------------------------------- reporting
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def run_all(cids=None) -> dict[str, Any]:
    cids = cids or case_ids()
    out = {"environment": environment(), "criteria": CRITERIA, "start_offset": START_OFFSET, "cases": {}}
    for cid in cids:
        case = load_case(cid)
        ok, why = supported(case)
        rec: dict[str, Any] = {"source": case["source"], "supported": ok}
        if not ok:
            rec["reason"] = why
            rec["classification"] = "no soportado"
        else:
            rec["forward"] = forward_check(case)
            fit = fit_case(case)
            rec["fit"] = fit
            if fit.get("success"):
                rec.update(compare(case, fit))
            else:
                rec["classification"] = "no reproduce"
        out["cases"][cid] = rec
    return out


def markdown_table(run: dict[str, Any]) -> str:
    lines = ["| Caso | Fuente (SI) | L0 RMS fijo Δrel | L1 RMS HM / ref (Δrel) | L2 log β | L3 SE | L4 desplaz. | L4 correl. | Clasificación |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cid, r in run["cases"].items():
        src = f"{r['source']['article']} {', '.join(r['source']['si_pages'])}"
        if not r["supported"]:
            lines.append(f"| `{cid}` | {src} | — | — | — | — | — | — | no soportado |")
            continue
        fw = r.get("forward", {})
        l0 = f"{fw.get('rel', float('nan')):+.1e}" if fw.get("success") else "fallo"
        L = r.get("levels", {})
        if not L:
            lines.append(f"| `{cid}` | {src} | {l0} | fallo del ajuste | — | — | — | — | {r['classification']} |")
            continue
        l1 = f"{L['L1']['rms_hm']:.3e} / {L['L1']['rms_ref']:.3e} ({L['L1']['rel']:+.1%})"
        l2 = f"{sum(x['pass'] for x in L['L2']['items'])}/{len(L['L2']['items'])}"
        l3 = f"{sum(x['pass'] for x in L['L3']['items'])}/{len(L['L3']['items'])}"
        l4s = f"{L['L4']['shifts_pass']}/{L['L4']['shifts_n']}"
        l4c = f"{L['L4']['corr_pass']}/{L['L4']['corr_n']}" if L['L4']['corr_n'] else "—"
        lines.append(f"| `{cid}` | {src} | {l0} | {l1} | {l2} | {l3} | {l4s} | {l4c} | {r['classification']} |")
    return "\n".join(lines)


def detail_tables(run: dict[str, Any]) -> str:
    parts = []
    for cid, r in run["cases"].items():
        L = r.get("levels")
        if not L:
            continue
        src = r["source"]
        parts.append(f"#### `{cid}` — {src['article']}, SI {', '.join(src['si_pages'])}"
                     + (f"; {src['article_table']}" if src.get("article_table") else ""))
        parts.append("")
        parts.append("| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |")
        parts.append("|---|---|---|---|---|---|---|---|---|")
        et = (r["fit"].get("errors_tab") or {}).get("se") or []
        names = r["fit"]["names"]
        for x2, x3 in zip(L["L2"]["items"], L["L3"]["items"]):
            i = names.index(x2["name"])
            ets = et[i] if i < len(et) else None
            parts.append(f"| {x2['name']} | {x2['hm']:.4f} | {x2['ref']:.4f} | {x2['delta']:+.4f}"
                         f"{'' if x2['pass'] else ' ✗'} | {x2['tol']:.4f} | {_fmt(x3['hm'])} | {x3['ref']:.4f} | "
                         f"{x3['rel']:+.1%}{'' if x3['pass'] else ' ✗'} | {_fmt(ets)} |")
        l1 = L["L1"]
        parts.append("")
        parts.append(f"RMS: HM fit {l1['rms_hm']:.5e}, ref. {l1['rms_ref']:.5e} ({l1['rel']:+.2%}). "
                     + (f"σ (pestaña Errors, ν = {r['fit']['errors_tab']['dof']}): HM fit {l1['sigma_hm']:.5e}, "
                        f"ref. {l1['sigma_ref']:.5e} ({l1['sigma_rel']:+.2%})." if 'sigma_hm' in l1 else ""))
        parts.append("")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--json", type=Path)
    p_run.add_argument("--md", type=Path)
    p_run.add_argument("cases", nargs="*")
    p_s = sub.add_parser("sensitivity")
    p_s.add_argument("--n", type=int, default=50)
    p_s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p_s.add_argument("--json", type=Path)
    p_s.add_argument("cases", nargs="*")
    p_t = sub.add_parser("tolerances")
    p_t.add_argument("sensitivity_json", type=Path)
    p_t.add_argument("--write", action="store_true")
    p_b = sub.add_parser("baseline")
    p_b.add_argument("--write", action="store_true")
    p_b.add_argument("cases", nargs="*")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run = run_all(args.cases or None)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(run, indent=1, default=float) + "\n")
        md = markdown_table(run) + "\n\n" + detail_tables(run)
        if args.md:
            args.md.write_text(md + "\n")
        print(md)
    elif args.cmd == "sensitivity":
        cids = args.cases or [c for c in case_ids() if supported(load_case(c))[0]]
        s = sensitivity(cids, args.n, args.workers)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(s, indent=1) + "\n")
        for cid, r in s["cases"].items():
            if not r.get("n_ok"):
                print(cid, "no successful refit")
                continue
            fr = np.array(r["free"])
            print(f"{cid:15s} ok={r['n_ok']}/{r['n']} cv_rms={r['cv_rms']:.2e} "
                  f"max SD(logb)={max(np.array(r['sd_log_beta'])[fr]):.2e} "
                  f"max range(logb)={max(np.array(r['range_log_beta'])[fr]):.2e} "
                  f"max cv(SE)={np.nanmax(np.array(r['cv_se'])[fr]):.2e} t={r['median_elapsed_s']:.1f}s")
    elif args.cmd == "tolerances":
        sens = json.loads(args.sensitivity_json.read_text())
        for cid, sc in sens["cases"].items():
            t = tolerances_from_sensitivity(load_case(cid), sc)
            t["sensitivity"] = {"file": str(args.sensitivity_json), "n": sens["n"],
                                "base_seed": sens["base_seed"], "commit": sens["environment"]["commit"]}
            print(cid, t["L1_rms_rel"], t["widened"])
            if args.write:
                (DATA_DIR / cid / "tolerances.json").write_text(json.dumps(t, indent=1) + "\n")
    elif args.cmd == "baseline":
        cids = args.cases or case_ids()
        for cid in cids:
            case = load_case(cid)
            if not supported(case)[0]:
                continue
            f = fit_case(case)
            bl = {"case": cid, "log_beta": dict(zip(f["names"], f["log_beta"])),
                  "se": dict(zip(f["names"], f["se"])), "rms": f["rms"],
                  "sigma_errors_tab": f["errors_tab"]["sigma"],
                  "errors_tab_se": dict(zip(f["names"], f["errors_tab"]["se"])),
                  "limiting_shifts": f["limiting_shifts"],
                  "objective_evaluations": f["objective_evaluations"],
                  "settings": {"optimizer": "powell", "algorithm": "Newton-Raphson",
                               "start_offset": START_OFFSET},
                  "environment": environment()}
            print(cid, json.dumps({k: bl[k] for k in ("log_beta", "rms")}))
            if args.write:
                (DATA_DIR / cid / "baseline.json").write_text(json.dumps(bl, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
