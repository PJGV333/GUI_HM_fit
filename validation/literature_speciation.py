# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage D of the literature benchmark: HM fit's mass-balance solver against the
published speciation tables of R09b (SI S20-S23), without fitting or noise.

Column meaning (inferred from the R09a SI appendix, S22-S26, and R09b; checked on the
tables themselves, docs/LITERATURE_BENCHMARK.md §7):

* ``xbound(A)``: fraction of the analysed reagent A bound in species that contain both A
  and the bound analyte B (A = X in "X vs R", A = Q in "Q vs B");
* ``BC50``: total concentration of A at which 50 % of B is bound; ``T(A) = BC50``;
* ``T(B)``: total of B at that point; ``T(C)``: total of the ancillary ion, equal to
  ``T(A)`` (electroneutrality, T_Q = T_X);
* ``[.]``: free concentrations; ``x(A)`` = [A]/T_A and ``x(S)`` = a_S [S]/T_A, with a_S the
  stoichiometric coefficient of A in species S (fractions of the total A; they sum to 1).

Each row is solved with ``hmfit_core.solvers.NewtonRaphson`` (the GUI default) and
``LevenbergMarquardt`` from the printed totals, with the constants of the results page
(``si``) or the rounded constants of the article tables (``article``).

    python -m validation.literature_speciation [--json OUT] [--md OUT]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data" / "literature"

# Pre-registered (docs/LITERATURE_BENCHMARK.md §4.6)
CONC_REL_TOL = 0.005
FRAC_ABS_TOL = 0.0005

# Rounded constants of the article tables (log beta), used by the BC50 Calculator
# according to the figure captions of R09b ("from the formation constants of Tables 1-3").
ARTICLE_LOG_BETA = {
    "r09b-r-qx": {"RQX": 6.121, "RX": 2.946, "R2": 0.783, "R2X": 4.513, "QX": 4.090},
    "r09b-b-qx": {"BQX": 4.767, "BQ": 1.373, "QX": 4.090},
    "r09b-d-qx": {"DQX": 5.371, "DQ2X2": 12.061, "D2": 0.189, "DQ": 0.983, "DX": 2.132, "QX": 4.090},
}


def load(sid: str) -> dict[str, Any]:
    d = DATA_DIR / sid
    case = json.loads((d / "case.json").read_text())
    with (d / "table.csv").open() as fh:
        rows = list(csv.reader(fh))
    case["_header"], case["_rows"] = rows[0], rows[1:]
    case["_fit_case"] = json.loads((DATA_DIR / case["source"]["constants_from"] / "case.json").read_text())
    return case


def model(case, constants: str):
    fc = case["_fit_case"]
    reagents = fc["reagents"]
    nr = len(reagents)
    species = [s["name"] for s in fc["species"]]
    M = [[int(i == j) for j in range(nr)] for i in range(nr)] + [list(s["stoich"]) for s in fc["species"]]
    if constants == "si":
        lb = [s["log_beta"] if s["status"] == "fixed" else fc["reference"]["log_beta"][s["name"]]["log_beta"]
              for s in fc["species"]]
    else:
        art = ARTICLE_LOG_BETA[case["source"]["constants_from"]]
        lb = [art[n] for n in species]
    return reagents, list(reagents) + species, np.asarray(M, float), np.asarray(lb, float)


def solve(case, constants: str = "si", algorithm: str = "Newton-Raphson") -> dict[str, Any]:
    from hmfit_core.solvers import LevenbergMarquardt, NewtonRaphson

    reagents, names, M, lb = model(case, constants)
    hdr, rows = case["_header"], case["_rows"]
    A, B, C = case["analysed"], case["bound"], case["ancillary"]
    col = {h: i for i, h in enumerate(hdr)}
    totals = np.zeros((len(rows), len(reagents)))
    for i, r in enumerate(rows):
        t = {A: float(r[col["BC50"]]), B: float(r[col[f"T({B})"]]), C: float(r[col[f"T({C})"]])}
        totals[i] = [t[x] for x in reagents]
    Solver = NewtonRaphson if algorithm == "Newton-Raphson" else LevenbergMarquardt
    _, Co = Solver(totals, M.T, nas=[], model_sett="Free").concentraciones(lb)
    Co = np.asarray(Co, float)
    ia = reagents.index(A)
    out_rows = []
    for i, r in enumerate(rows):
        TA = totals[i, ia]
        calc: dict[str, float] = {}
        for x in reagents:
            calc[f"[{x}]"] = Co[i, names.index(x)]
        for h in hdr:
            m = re.fullmatch(r"x\((.+)\)", h)
            if m:
                sp = m.group(1)
                k = names.index(sp)
                calc[h] = M[k, ia] * Co[i, k] / TA if TA > 0 else float("nan")
        # xbound(A): species containing A and B
        ib = reagents.index(B)
        bound = sum(M[k, ia] * Co[i, k] for k in range(len(names)) if M[k, ia] > 0 and M[k, ib] > 0)
        calc[f"xbound({A})"] = bound / TA if TA > 0 else float("nan")
        # half of B bound (BC50 definition): bound B / T_B
        TB = totals[i, ib]
        bB = sum(M[k, ib] * Co[i, k] for k in range(len(names)) if M[k, ib] > 0 and M[k, ia] > 0)
        calc["_bound_fraction_B"] = bB / TB if TB > 0 else float("nan")
        out_rows.append(calc)
    return {"rows": out_rows, "totals": totals}


def compare(case, constants="si", algorithm="Newton-Raphson") -> dict[str, Any]:
    res = solve(case, constants, algorithm)
    hdr, rows = case["_header"], case["_rows"]
    col = {h: i for i, h in enumerate(hdr)}
    A = case["analysed"]
    items = []
    worst = {"conc_rel": 0.0, "frac_abs": 0.0, "xbound_abs": 0.0}
    n_pass = 0
    for i, (r, calc) in enumerate(zip(rows, res["rows"])):
        row_ok = True
        detail = {}
        for h in hdr:
            if h.startswith("[") and h in calc:
                tab = float(r[col[h]])
                v = calc[h]
                if tab == 0.0:
                    ok = abs(v) <= 1e-12
                    d = abs(v)
                else:
                    d = abs(v / tab - 1.0)
                    ok = d <= CONC_REL_TOL
                    worst["conc_rel"] = max(worst["conc_rel"], d)
                detail[h] = (v, tab, d, ok)
                row_ok &= ok
            elif h.startswith("x(") and h in calc:
                tab = float(r[col[h]])
                v = calc[h]
                if not np.isfinite(v):     # T_A = 0 cannot happen in these tables; keep loud
                    ok, d = False, float("nan")
                else:
                    d = abs(v - tab)
                    ok = d <= FRAC_ABS_TOL
                    worst["frac_abs"] = max(worst["frac_abs"], d)
                detail[h] = (v, tab, d, ok)
                row_ok &= ok
        xb = float(r[col[f"xbound({A})"]])
        worst["xbound_abs"] = max(worst["xbound_abs"], abs(calc[f"xbound({A})"] - xb))
        n_pass += row_ok
        items.append({"row": i, "xbound": xb, "pass": bool(row_ok),
                      "fails": {h: v for h, v in detail.items() if not v[3]},
                      "bound_fraction_B": calc["_bound_fraction_B"]})
    return {"constants": constants, "algorithm": algorithm, "n_rows": len(rows), "n_pass": n_pass,
            "worst": worst, "rows": items}


def rounding_explains(case, constants="si", n=2000, seed=7) -> dict[str, Any]:
    """Diagnostic for failing rows: perturb the printed totals within half a unit of their
    last digit (T_C = T_A kept equal) and record whether some perturbation brings the whole
    row within tolerance, i.e. whether the miss is explained by the rounding of the inputs."""
    from decimal import Decimal

    from hmfit_core.solvers import NewtonRaphson

    reagents, names, M, lb = model(case, constants)
    hdr, rows = case["_header"], case["_rows"]
    col = {h: i for i, h in enumerate(hdr)}
    A, B, C = case["analysed"], case["bound"], case["ancillary"]
    base = compare(case, constants)
    rng = np.random.default_rng(seed)
    explained, total = 0, 0
    unexplained = []
    for it in base["rows"]:
        if it["pass"]:
            continue
        r = rows[it["row"]]
        strs = {A: r[col["BC50"]], B: r[col[f"T({B})"]]}
        def half(s):
            return 0.5 * 10.0 ** Decimal(s).as_tuple().exponent
        T = np.zeros((n, len(reagents)))
        for k in range(n):
            ta = float(strs[A]) + rng.uniform(-half(strs[A]), half(strs[A]))
            tb = float(strs[B]) + (rng.uniform(-half(strs[B]), half(strs[B])) if float(strs[B]) else 0.0)
            t = {A: ta, B: tb, C: ta}
            T[k] = [t[x] for x in reagents]
        _, Co = NewtonRaphson(T, M.T, nas=[], model_sett="Free").concentraciones(lb)
        Co = np.asarray(Co, float)
        ia = reagents.index(A)
        # joint per row: ONE perturbation of the totals must bring every cell of the row
        # (not only the failing ones) within tolerance
        ok_all = np.ones(n, dtype=bool)
        for h in hdr:
            if h.startswith("[") and h[1:-1] in reagents:
                tab = float(r[col[h]])
                vals = Co[:, names.index(h[1:-1])]
                ok_all &= (np.abs(vals / tab - 1.0) <= CONC_REL_TOL) if tab else (np.abs(vals) <= 1e-12)
            elif h.startswith("x(") and h[2:-1] in names:
                kk = names.index(h[2:-1])
                ok_all &= np.abs(M[kk, ia] * Co[:, kk] / T[:, ia] - float(r[col[h]])) <= FRAC_ABS_TOL
        total += 1
        hit = bool(ok_all.any())
        explained += hit
        if not hit:
            unexplained.append({"row": it["row"], "fails": it["fails"]})
    return {"failing_rows": total, "explained_by_input_rounding": explained, "unexplained": unexplained,
            "n_perturbations": n}


def run_all() -> dict[str, Any]:
    out = {}
    for sid in sorted(p.name for p in DATA_DIR.iterdir() if (p / "table.csv").is_file()):
        case = load(sid)
        rec = {"source": case["source"], "results": {}}
        for constants in ("si", "article"):
            for alg in ("Newton-Raphson", "Levenberg-Marquardt"):
                rec["results"][f"{constants}/{alg}"] = compare(case, constants, alg)
            rec["results"][f"{constants}/rounding"] = rounding_explains(case, constants)
        out[sid] = rec
    return out


def markdown(run) -> str:
    lines = ["| Tabla | Fuente | Constantes | Filas que pasan (NR / LM) | máx. Δrel conc. | máx. \\|Δ\\| fracción | máx. \\|Δ\\| xbound | Filas fallidas explicadas por el redondeo de los totales |",
             "|---|---|---|---|---|---|---|---|"]
    for sid, rec in run.items():
        src = f"{rec['source']['article']} SI {', '.join(rec['source']['si_pages'])}"
        for constants in ("si", "article"):
            nr = rec["results"][f"{constants}/Newton-Raphson"]
            lm = rec["results"][f"{constants}/Levenberg-Marquardt"]
            ro = rec["results"][f"{constants}/rounding"]
            w = nr["worst"]
            lab = "página de resultados" if constants == "si" else "artículo (redondeadas)"
            lines.append(f"| `{sid}` | {src} | {lab} | {nr['n_pass']}/{nr['n_rows']} / {lm['n_pass']}/{lm['n_rows']} | "
                         f"{w['conc_rel']:.2%} | {w['frac_abs']:.4f} | {w['xbound_abs']:.4f} | "
                         f"{ro['explained_by_input_rounding']}/{ro['failing_rows']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    run = run_all()
    md = markdown(run)
    print(md)
    if args.json:
        args.json.write_text(json.dumps(run, indent=1, default=float) + "\n")
    if args.md:
        args.md.write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
