# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E of the literature benchmark: hmfit_core.bc50 against published BC50 values.

1. R09b SI S20-S23 tables (T_C = T_A), x grid exact: BC50, T_B, free concentrations and
   fractions compared with the printed digits.
2. R09a: [TMA] = 1 mol/L and T_TMA = T_Cl for receptors 1 and 2 (A = receptor, B = Cl-).
3. R09b: BC50^0 of X (R, D) and of Q (B, D) at T_X = T_Q.

Criteria: docs/LITERATURE_BENCHMARK.md §4.7.  ``python -m validation.literature_bc50 [--json OUT]``
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from hmfit_core.bc50 import bc50_curve
from validation import literature_speciation as ls

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data" / "literature"

ARTICLE_R09A = {  # R09a Tables 1 and 2 (log beta, rounded as printed)
    "r09a-r1-5sp": {"RQX": 6.121, "RX": 2.946, "R2": 0.783, "R2X": 4.513, "QX": 4.090},
    "r09a-p2-058": {"PQX": 7.19, "PX": 3.58, "P2": 0.1, "P2X": 6.26, "P2QX": 9.47, "QX": 4.06},
}


def _ulp(s: str) -> float:
    return 10.0 ** Decimal(s).as_tuple().exponent


def _system(fit_case_id: str, constants: str):
    fc = json.loads((DATA_DIR / fit_case_id / "case.json").read_text())
    names = [s["name"] for s in fc["species"]]
    stoich = [s["stoich"] for s in fc["species"]]
    if constants == "si":
        lb = [s["log_beta"] if s["status"] == "fixed" else fc["reference"]["log_beta"][s["name"]]["log_beta"]
              for s in fc["species"]]
    else:
        art = ARTICLE_R09A.get(fit_case_id) or ls.ARTICLE_LOG_BETA[fit_case_id]
        lb = [art[n] for n in names]
    return fc["reagents"], stoich, lb, names


def tables(constants: str) -> dict[str, Any]:
    out = {}
    for sid in ("r09b-spec-r", "r09b-spec-b", "r09b-spec-d-x", "r09b-spec-d-q"):
        case = ls.load(sid)
        reagents, stoich, lb, names = _system(case["source"]["constants_from"], constants)
        A, B, C = case["analysed"], case["bound"], case["ancillary"]
        hdr, rows = case["_header"], case["_rows"]
        col = {h: i for i, h in enumerate(hdr)}
        xs = [float(r[col[f"xbound({A})"]]) for r in rows]
        res = bc50_curve(reagents, stoich, lb, A, B, C, x=xs, condition="electroneutral", counterion=A,
                         species_names=names)
        allnames = res.species
        cells, within_half, within_one = 0, 0, 0
        worst = []
        for i, r in enumerate(rows):
            calc = {"BC50": res.bc50[i], f"T({B})": res.totals[B][i], f"T({C})": res.totals[C][i]}
            for x in reagents:
                calc[f"[{x}]"] = res.free[x][i]
            for h in hdr:
                m = re.fullmatch(r"x\((.+)\)", h)
                if m:
                    calc[h] = res.fractions[i, allnames.index(m.group(1))]
            for h, v in calc.items():
                s = r[col[h]]
                d = abs(v - float(s)) / _ulp(s)
                cells += 1
                within_half += d <= 0.5 + 1e-9
                within_one += d <= 1.0 + 1e-9
                worst.append((d, i, h, float(v), s))
        worst.sort(reverse=True)
        out[sid] = {"cells": cells, "within_half_ulp": within_half, "within_one_ulp": within_one,
                    "reproduce": within_one == cells and within_half >= 0.98 * cells,
                    "worst": [{"ulp": w[0], "row": w[1], "column": w[2], "hm": w[3], "printed": w[4]}
                              for w in worst[:5]],
                    "warnings": res.warnings}
    return out


def r09a(constants: str) -> dict[str, Any]:
    xs = np.round(np.arange(0.0, 0.99, 0.02), 2)
    curves = {}
    for cid in ("r09a-r1-5sp", "r09a-p2-058"):
        reagents, stoich, lb, names = _system(cid, constants)
        A = reagents[0]
        fixed = bc50_curve(reagents, stoich, lb, A, "X", "Q", x=xs, condition="fixed_free",
                           free_ancillary=1.0, species_names=names)
        neutral = bc50_curve(reagents, stoich, lb, A, "X", "Q", x=xs, condition="electroneutral",
                             counterion="X", species_names=names)
        curves[cid] = {"fixed": fixed, "neutral": neutral}
    r1f, r2f = curves["r09a-r1-5sp"]["fixed"].bc50, curves["r09a-p2-058"]["fixed"].bc50
    r1n, r2n = curves["r09a-r1-5sp"]["neutral"].bc50, curves["r09a-p2-058"]["neutral"].bc50
    ratio_f, ratio_n = r1f / r2f, r1n / r2n
    imax = int(np.nanargmax(ratio_n))
    checks = {
        "BC50_0_receptor1_TMA1": (float(r1f[0]), "1.03E-02"),
        "BC50_0_receptor2_TMA1": (float(r2f[0]), "6.7E-04"),
        "ratio_TMA1_min": (float(np.nanmin(ratio_f)), "15.4"),
        "ratio_TMA1_max": (float(np.nanmax(ratio_f)), "15.4"),
        "ratio_neutral_max": (float(ratio_n[imax]), "16.5"),
        "ratio_neutral_x0": (float(ratio_n[0]), "4.6"),
    }
    out = {k: {"hm": v, "printed": s, "pass": abs(v - float(s)) <= 0.5 * _ulp(s) + 1e-12}
           for k, (v, s) in checks.items()}
    out["ratio_neutral_argmax_x"] = float(xs[imax])
    out["BC50_0_receptor1_neutral"] = float(r1n[0])
    out["BC50_0_receptor2_neutral"] = float(r2n[0])
    out["warnings"] = {cid: {k: c.warnings for k, c in v.items()} for cid, v in curves.items()}
    return out


def r09b(constants: str) -> dict[str, Any]:
    targets = [("X", "R", "Q", "r09b-r-qx", "5.0E-03"), ("X", "D", "Q", "r09b-d-qx", "1.05E-02"),
               ("Q", "B", "X", "r09b-b-qx", "1.95E-01"), ("Q", "D", "X", "r09b-d-qx", "1.19E-02")]
    out = {}
    for A, B, C, cid, printed in targets:
        reagents, stoich, lb, names = _system(cid, constants)
        r = bc50_curve(reagents, stoich, lb, A, B, C, x=[0.0], condition="electroneutral", counterion=A,
                       species_names=names)
        v = float(r.bc50[0])
        out[f"{A} vs {B}"] = {"hm": v, "printed": printed, "pass": abs(v - float(printed)) <= 0.5 * _ulp(printed),
                              "warnings": r.warnings}
    return out


def run_all() -> dict[str, Any]:
    return {c: {"tables": tables(c), "r09a": r09a(c), "r09b": r09b(c)} for c in ("si", "article")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    run = run_all()
    for c, rec in run.items():
        print(f"== constants: {c}")
        for sid, t in rec["tables"].items():
            print(f"  {sid}: {t['within_half_ulp']}/{t['cells']} <= 0.5 ulp, {t['within_one_ulp']}/{t['cells']} <= 1 ulp"
                  f" -> {'reproduce' if t['reproduce'] else 'NO'}; worst {t['worst'][0]}")
        for k, v in rec["r09a"].items():
            print(f"  R09a {k}: {v}")
        for k, v in rec["r09b"].items():
            print(f"  R09b {k}: {v['hm']:.4g} vs {v['printed']} pass={v['pass']}")
    if args.json:
        args.json.write_text(json.dumps(run, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
