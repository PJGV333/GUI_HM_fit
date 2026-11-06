"""Secondary analyses of the Phase 2 campaign (pre-registration amendment 2, item 5).

Declared as *secondary*: they were specified after tier-1 results existed and do not feed
the 2.3/2.4 decision rules.

    python -m validation.coverage_secondary RUN_DIR [--md OUT.md] [--csv OUT.csv]

1. Pooled coverage per technique x method x parameter over the realistic grid restricted to
   8-15 titration points (main passes, GUI default estimator), with a Wilson interval and a
   chi-square test of homogeneity of the per-cell coverage proportions (2 x k table).
2. Coverage of the t interval with the bootstrap SE, theta_hat +/- t_{0.975,nu} * SE_boot,
   next to the percentile interval the GUI reports. theta_hat and t_{0.975,nu} are taken from
   the analytic record of the *same dataset* (same seed): t = (theta_hat - lo_analytic) /
   SE_analytic. Full-refit passes (``~fr``) share datasets with their main pass, so they are
   joined by replicate index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from validation.coverage_campaign import ALL_METHODS, wilson
from validation.coverage_report import REALISTIC_NOISE, _meta

BOOT = [m for m in ALL_METHODS if m != "analytic"]


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def per_replicate(run_dir: Path) -> pd.DataFrame:
    """One row per cell x replicate x method x parameter: covered by the percentile CI and,
    for bootstraps, by the t interval with bootstrap SE."""
    meta = _meta().set_index("cell")
    by_cell = {p.stem: _rows(p) for p in sorted(run_dir.glob("*.jsonl")) if p.stem != "manifest"}
    recs = []
    for cell, rows in by_cell.items():
        if cell not in meta.index:
            continue
        main_cell = cell[:-3] if cell.endswith("~fr") else cell
        analytic_rows = {r["rep"]: r for r in by_cell.get(main_cell, []) if r.get("fit_ok")}
        for r in rows:
            if not r.get("fit_ok"):
                continue
            truth = np.asarray(r["truth"], float)
            an = (analytic_rows.get(r["rep"]) or r).get("methods", {}).get("analytic")
            an_ok = an is not None and "error" not in an
            for m, rec in r["methods"].items():
                if "error" in rec:
                    continue
                for j in range(truth.size):
                    if rec["fixed"][j]:
                        continue
                    lo, hi = rec["lo"][j], rec["hi"][j]
                    cov_p = (lo <= truth[j] <= hi) if np.isfinite(lo) and np.isfinite(hi) else np.nan
                    cov_t = np.nan
                    if m != "analytic" and an_ok:
                        th, se_a, lo_a, se_b = an["est"][j], an["se"][j], an["lo"][j], rec["se"][j]
                        if all(np.isfinite(v) for v in (th, se_a, lo_a, se_b)) and se_a > 0:
                            t = (th - lo_a) / se_a
                            cov_t = float(abs(truth[j] - th) <= t * se_b)
                    recs.append({"cell": cell, "rep": r["rep"], "method": m, "param": j,
                                 "cov_pct": cov_p, "cov_t": cov_t})
    df = pd.DataFrame(recs)
    return df.merge(meta.reset_index(), on="cell", how="left") if not df.empty else df


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["noise"].isin(REALISTIC_NOISE) & df["n_points"].between(8, 15) & (df["estimator"] == "soft_penalty")
           & (df["pass"] == "main") & df["cov_pct"].notna()]
    out = []
    for (tech, m, j), g in d.groupby(["technique", "method", "param"]):
        per_cell = g.groupby("cell")["cov_pct"].agg(["sum", "count"])
        k, n = int(per_cell["sum"].sum()), int(per_cell["count"].sum())
        lo, hi = wilson(k, n)
        p_het = np.nan
        if len(per_cell) > 1 and 0 < k < n:
            table = np.vstack([per_cell["sum"].to_numpy(), (per_cell["count"] - per_cell["sum"]).to_numpy()])
            p_het = float(stats.chi2_contingency(table, correction=False)[1])
        out.append({"technique": tech, "method": m, "param": int(j), "cells": len(per_cell), "N": n,
                    "coverage": k / n, "wilson_lo": lo, "wilson_hi": hi,
                    "cell_cov_min": float((per_cell["sum"] / per_cell["count"]).min()),
                    "cell_cov_max": float((per_cell["sum"] / per_cell["count"]).max()),
                    "p_heterogeneity": p_het})
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    res = pd.DataFrame(out)
    return res.sort_values(["technique", "param", "method"], key=lambda s: s.map(order) if s.name == "method" else s)


def t_vs_percentile(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["method"] != "analytic") & df["noise"].isin(REALISTIC_NOISE) & df["cov_t"].notna()
           & df["cov_pct"].notna()]
    out = []
    for (tech, est, m), g in d.groupby(["technique", "estimator", "method"]):
        n = len(g)
        kp, kt = int(g["cov_pct"].sum()), int(g["cov_t"].sum())
        out.append({"technique": tech, "estimator": est, "method": m, "cells": g["cell"].nunique(), "N": n,
                    "cov_percentile": kp / n, "pct_lo": wilson(kp, n)[0], "pct_hi": wilson(kp, n)[1],
                    "cov_t_bootSE": kt / n, "t_lo": wilson(kt, n)[0], "t_hi": wilson(kt, n)[1]})
    return pd.DataFrame(out)


def _md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(sin datos)"
    cols = list(frame.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else ("—" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in frame.iterrows()])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    df = per_replicate(Path(a.run_dir))
    pool, tv = pooled(df), t_vs_percentile(df)
    text = "\n".join([
        "# Fase 2 — análisis secundarios (generado)", "",
        "> Secundarios (enmienda 2 del pre-registro): especificados con resultados del tier 1 ya existentes;",
        "> no alimentan las reglas de decisión. Generado por `python -m validation.coverage_secondary`.", "",
        "## 1. Cobertura agregada, rejilla realista, 8–15 puntos, estimador por defecto", "",
        "`p_heterogeneity`: χ² de homogeneidad de las proporciones de cobertura entre celdas (2 × k).", "",
        _md(pool), "",
        "## 2. IC t con SE bootstrap frente al IC de percentiles (rejilla realista)", "",
        "θ̂ ± t₀.₉₇₅,ν · SE_boot, con θ̂ y t de la réplica analítica del mismo conjunto de datos.", "",
        _md(tv), "",
    ])
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
    if a.csv:
        pool.assign(table="pooled").to_csv(a.csv, index=False, float_format="%.6g")
        tv.to_csv(Path(a.csv).with_name(Path(a.csv).stem + "_t_vs_percentile.csv"), index=False, float_format="%.6g")
    print(text)


if __name__ == "__main__":
    main()
