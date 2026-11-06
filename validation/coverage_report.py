"""Tables for the Phase 2 coverage campaign (docs/PHASE2_COVERAGE_TABLES.md).

    python -m validation.coverage_report RUN_DIR --csv validation/results/coverage/summary.csv \
        --md docs/PHASE2_COVERAGE_TABLES.md

Applies, mechanically, the aggregation and decision inputs pre-registered in
docs/PHASE2_PREREGISTRATION.md section 4. Every number comes from RUN_DIR (raw JSONL).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.coverage_campaign import ALL_METHODS, primary_grid, summarize

REALISTIC_NOISE = {"iid", "hetero", "iid1", "iid5"}
CORRELATED_NOISE = {"offset", "point", "drift"}


def _meta() -> pd.DataFrame:
    rows = []
    for c in primary_grid():
        rows.append({"cell": c["id"], "technique": c["technique"], "design": c["design"],
                     "n_points": c["n_points"], "noise": c["noise"], "saturation": c["saturation"],
                     "estimator": c.get("estimator", "soft_penalty"), "tier": c["tier"],
                     "pass": "fr" if c["id"].endswith("~fr") else "main"})
    return pd.DataFrame(rows)


def load(run_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(summarize(run_dir))
    if df.empty:
        return df
    df = df.merge(_meta(), on="cell", how="left")
    df["diagnostic"] = df["cell"].str.startswith("diag-")
    return df


def _evidence_anticonservative(r) -> bool:
    return (r["class"] == "anticonservative") and (
        (np.isfinite(r["coverage_hi"]) and r["coverage_hi"] < 0.95)
        or (np.isfinite(r["se_over_sd_hi"]) and r["se_over_sd_hi"] < 1.0)
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per technique x estimator x method over the realistic grid (pre-registration s.4)."""
    d = df[(~df["diagnostic"]) & df["noise"].isin(REALISTIC_NOISE) & (~df["fixed"].astype(bool))
           & df["coverage"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["evid_anti"] = d.apply(_evidence_anticonservative, axis=1)
    out = []
    def _verdict(evid_anti: bool, frac: float) -> str:
        return ("anticonservative" if evid_anti
                else "calibrated" if frac >= 0.75 else "not calibrated (mixed/conservative)")

    for (tech, est, m), g in d.groupby(["technique", "estimator", "method"]):
        n = len(g)
        frac_cal = float((g["class"] == "calibrated").mean())
        verdict = _verdict(bool(g["evid_anti"].any()), frac_cal)
        # Amendment 1: is the verdict robust to the pairs whose Wilson interval contains a
        # coverage threshold? Worst case: none of them calibrated; best case: all of them
        # calibrated when SE/SD is inside [0.85, 1.15].
        cls_ci = g["class_ci"] if "class_ci" in g else g["class"]
        ind = cls_ci == "indeterminate"
        se_ok = g["se_over_sd"].between(0.85, 1.15)
        frac_lo = float((cls_ci == "calibrated").mean())
        frac_hi = float(((cls_ci == "calibrated") | (ind & se_ok)).mean())
        v_lo, v_hi = _verdict(bool(g["evid_anti"].any()), frac_lo), _verdict(bool(g["evid_anti"].any()), frac_hi)
        verdict_ci = v_lo if v_lo == v_hi else "indeterminate"
        out.append({
            "technique": tech, "estimator": est, "method": m, "pairs": n,
            "n_rep_median": int(g["n_used"].median()),
            "calibrated": f"{frac_cal:.0%}", "conservative": f"{(g['class'] == 'conservative').mean():.0%}",
            "anticonservative": f"{(g['class'] == 'anticonservative').mean():.0%}",
            "anti_with_evidence": int(g["evid_anti"].sum()),
            "coverage_median": g["coverage"].median(), "coverage_min": g["coverage"].min(),
            "se_sd_median": g["se_over_sd"].median(), "se_sd_min": g["se_over_sd"].min(),
            "relevant_bias_pairs": int(g["relevant_bias"].sum()), "verdict": verdict,
            "indeterminate": f"{ind.mean():.0%}", "verdict_ci": verdict_ci,
        })
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    res = pd.DataFrame(out)
    res["_o"] = res["method"].map(order)
    return res.sort_values(["technique", "estimator", "_o"]).drop(columns="_o")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def cell_table(df: pd.DataFrame) -> str:
    lines = ["| celda | método | p | N | sesgo/SD | SE/SD [IC95] | cobertura [Wilson] | izq/der | ancho med. | fallos ajuste | clase |",
             "|---|---|---:|---:|---:|---|---|---|---:|---:|---|"]
    for _, r in df.iterrows():
        if r.get("fixed") or not r.get("n_used"):
            continue
        lines.append(
            f"| {r['cell']} | {r['method']} | {int(r['param'])} | {int(r['n_used'])} | {_fmt(r['bias_over_sd'], 2)} | "
            f"{_fmt(r['se_over_sd'], 2)} [{_fmt(r['se_over_sd_lo'], 2)}, {_fmt(r['se_over_sd_hi'], 2)}] | "
            f"{_fmt(r['coverage'])} [{_fmt(r['coverage_lo'])}, {_fmt(r['coverage_hi'])}] | "
            f"{int(r['miss_left'])}/{int(r['miss_right'])} | {_fmt(r['width_median'], 4)} | "
            f"{_fmt(r['fit_fail_rate'], 3)} | {r['class']}"
            f"{' (indet.)' if r.get('class_ci') == 'indeterminate' else ''}{' · sesgo' if r['relevant_bias'] else ''}"
            f"{' · asim.' if r['asymmetric'] else ''} |")
    return "\n".join(lines)


def by_points(df: pd.DataFrame) -> str:
    """Coverage vs number of titration points (few-block threshold, stage 2.4.2)."""
    d = df[(~df["diagnostic"]) & df["technique"].isin(["uvvis", "nmr"]) & df["noise"].isin(REALISTIC_NOISE)
           & (df["estimator"] == "soft_penalty") & (~df["fixed"].astype(bool)) & df["coverage"].notna()]
    if d.empty:
        return "(sin datos)"
    piv = d.groupby(["technique", "method", "n_points"])["coverage"].median().unstack("n_points")
    lines = ["| técnica | método | " + " | ".join(f"n={c}" for c in piv.columns) + " |",
             "|---|---|" + "---:|" * len(piv.columns)]
    for (tech, m), row in piv.iterrows():
        lines.append(f"| {tech} | {m} | " + " | ".join(_fmt(v) for v in row.values) + " |")
    return "\n".join(lines)


def estimator_table(df: pd.DataFrame) -> str:
    d = df[(df["technique"] == "uvvis") & (df["noise"] == "iid") & (~df["diagnostic"])
           & df["n_points"].isin([8, 12, 15]) & df["method"].isin(["analytic", "os_row", "fr_row"])
           & (df["param"] == 0) & df["coverage"].notna()]
    if d.empty:
        return "(sin datos)"
    lines = ["| diseño | n | sat. | estimador | método | N | sesgo | SD | RMSE | cobertura [Wilson] | SE/SD | clase |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---|"]
    d = d.sort_values(["design", "n_points", "saturation", "method", "estimator"])
    for _, r in d.iterrows():
        lines.append(f"| {r['design']} | {int(r['n_points'])} | {r['saturation']} | {r['estimator']} | {r['method']} | "
                     f"{int(r['n_used'])} | {_fmt(r['bias'], 4)} | {_fmt(r['sd'], 4)} | {_fmt(r['rmse'], 4)} | "
                     f"{_fmt(r['coverage'])} [{_fmt(r['coverage_lo'])}, {_fmt(r['coverage_hi'])}] | "
                     f"{_fmt(r['se_over_sd'], 2)} | {r['class']} |")
    return "\n".join(lines)


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(sin datos)"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_markdown(df: pd.DataFrame, run_dir: Path) -> str:
    manifests = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    commits = sorted({m.get("commit") for m in manifests if m.get("commit")})
    env = manifests[-1] if manifests else {}
    n_cells = df["cell"].nunique() if not df.empty else 0
    parts = [
        "# Fase 2 — tablas de la campaña de cobertura (generado)",
        "",
        "> Generado por `python -m validation.coverage_report` a partir de los JSONL crudos. No editar a mano.",
        "",
        f"- Celdas con datos: {n_cells} de {len(primary_grid())} pre-registradas.",
        f"- Commits de código usados: {', '.join(c[:10] for c in commits)}.",
        f"- Entorno: Python {env.get('python')}, NumPy {env.get('numpy')}, SciPy {env.get('scipy')}, "
        f"pandas {env.get('pandas')}, PySide6 {env.get('PySide6')}; {env.get('platform')}.",
        "",
        "## 1. Agregado por técnica y método (rejilla realista, reglas del pre-registro §4)",
        "",
        md_table(aggregate(df)),
        "",
        "## 2. Cobertura mediana frente al número de puntos (UV-Vis y RMN, ruido no correlacionado)",
        "",
        by_points(df),
        "",
        "## 3. Comparación de estimadores de absortividades (etapa 2.3, UV-Vis, ruido iid, parámetro log K₁)",
        "",
        estimator_table(df),
        "",
        "## 4. Celdas con IC asimétricos",
        "",
        md_table(df[df.get("asymmetric", False) == True][["cell", "method", "param", "miss_left", "miss_right", "coverage"]])  # noqa: E712
        if "asymmetric" in df else "(sin datos)",
        "",
        "## 5. Tabla completa por celda",
        "",
        cell_table(df[~df["diagnostic"]]) if not df.empty else "(sin datos)",
        "",
        "## 6. Diagnóstico: error sistemático de ±3 % en la concentración del titulante",
        "",
        cell_table(df[df["diagnostic"]]) if not df.empty else "(sin datos)",
        "",
    ]
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    ap.add_argument("--aggregate-csv")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    df = load(run_dir)
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.csv, index=False, float_format="%.6g")
    if a.aggregate_csv:
        aggregate(df).to_csv(a.aggregate_csv, index=False, float_format="%.6g")
    if a.md:
        Path(a.md).write_text(build_markdown(df, run_dir), encoding="utf-8")
    print(aggregate(df).to_string(index=False))


if __name__ == "__main__":
    main()
