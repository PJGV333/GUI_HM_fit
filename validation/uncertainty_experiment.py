"""Stage E/F campaign: same datasets, estimator x uncertainty-method matrix.

For each replicate the dataset is generated once; log K is estimated with the
selected linear projector (exact scalar argmin of the production-equivalent
profiled objective; equivalence with production Powell is locked by
tests/test_uvvis_bias_reference.py), then every uncertainty method is applied
to the SAME fit via validation.uncertainty_methods.

Resumable: per-replicate JSONL checkpoint (rows.jsonl); summaries are computed
from rows on demand. Manifest records commit, versions, config.

Usage:
  QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      -m validation.uncertainty_experiment validation/configs/unc/<x>.json <outdir>

Scenario keys: name, base_seed, n_rep, points, noise, truth, g_max, c_H,
estimator ("ols" | "nnls" | float mu), B_wild, B_refit, refit_span.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from validation.bias_experiment import (
    argmin_1d,
    dataset,
    profile_objective,
    seed_for,
)

METHOD_KEYS = ("analytic", "hc1", "lin_wild_el", "lin_wild_spec",
               "onestep_spec", "onestep_wl",
               "refit_wild_el", "refit_wild_spec", "refit_resample")


def run_replicate(seed: int, scen: dict) -> dict:
    from validation.uncertainty_methods import replicate_uncertainty

    d = dataset(seed, scen)
    truth = float(scen.get("truth", 3.0))
    est = scen.get("estimator", 1e-2)
    mode, mu = ("ols", None) if est == "ols" else \
               ("nnls_hard", None) if est == "nnls" else ("soft_penalty", float(est))
    f = profile_objective(d["c"], d["y"], mode=mode, mu=mu if mu is not None else 1e-2)
    t_fit = time.perf_counter()
    theta = argmin_1d(f, float(scen.get("scan_lo", 1.6)), float(scen.get("scan_hi", 6.0)))[0]
    fit_s = time.perf_counter() - t_fit
    res = replicate_uncertainty(d, theta, est, truth,
                                B_wild=int(scen.get("B_wild", 1000)),
                                B_refit=int(scen.get("B_refit", 0)),
                                seed=seed,
                                refit_span=float(scen.get("refit_span", 0.05)))
    row = {"seed": int(seed), "theta": float(theta)}
    for k in METHOD_KEYS:
        if k in res:
            row[k] = res[k]
    row["_fit_s"] = float(fit_s)
    row["_unc_s"] = float(res["_seconds"]["total"])
    return row


def summarize(rows: list[dict], scen: dict) -> dict:
    truth = float(scen.get("truth", 3.0))
    th = np.array([r["theta"] for r in rows])
    out = {"n": len(rows), "truth": truth,
           "theta": {"bias": float(th.mean() - truth), "sd": float(th.std(ddof=1)),
                     "rmse": float(np.sqrt(((th - truth) ** 2).mean())),
                     "mc_se_bias": float(th.std(ddof=1) / np.sqrt(len(rows)))}}
    methods = [k for k in METHOD_KEYS if k in rows[0]]
    for k in methods:
        se = np.array([r[k]["se"] for r in rows], dtype=float)
        se = se[np.isfinite(se)]
        w = np.array([r[k]["width95"] for r in rows], dtype=float)
        w = w[np.isfinite(w)]
        cov = np.array([bool(r[k]["covered95"]) for r in rows if r[k]["covered95"] is not None])
        covb = np.array([bool(r[k].get("covered95_basic", False)) for r in rows
                         if r[k].get("covered95_basic") is not None])
        wb = np.array([r[k]["width95_basic"] for r in rows
                       if r[k].get("width95_basic") is not None])
        bb = np.array([r[k]["boot_bias"] for r in rows if "boot_bias" in r[k]])
        m = {"mean_se": float(se.mean()) if se.size else None,
             "se_over_mc_sd": float(se.mean() / th.std(ddof=1)) if se.size and len(rows) > 1 else None,
             "cov95": float(cov.mean()) if cov.size else None,
             "cov68": None, "n_cov": int(cov.size)}
        if cov.size:
            c68 = np.array([bool(r[k]["covered68"]) for r in rows if r[k]["covered68"] is not None])
            m["cov68"] = float(c68.mean()) if c68.size else None
            m["cov95_mc_se"] = float(np.sqrt(cov.mean() * (1 - cov.mean()) / cov.size))
        if w.size:
            m["mean_width95"] = float(w.mean())
        if covb.size:
            m["cov95_basic"] = float(covb.mean())
        if wb.size:
            m["mean_width95_basic"] = float(wb.mean())
        if bb.size:
            m["mean_boot_bias"] = float(bb.mean())
        covbc = np.array([bool(r[k]["covered95_bca"]) for r in rows
                          if r[k].get("covered95_bca") is not None])
        wbc = np.array([r[k]["width95_bca"] for r in rows
                        if r[k].get("width95_bca") is not None])
        if covbc.size:
            m["cov95_bca"] = float(covbc.mean())
        if wbc.size:
            m["mean_width95_bca"] = float(wbc.mean())
        if "n_fail" in rows[0][k]:
            fails = np.array([r[k]["n_fail"] for r in rows])
            B = np.array([r[k]["B"] for r in rows])
            m["refit_fail_rate"] = float(fails.sum() / B.sum())
        out[k] = m
    out["_time"] = {"mean_fit_s": float(np.mean([r["_fit_s"] for r in rows])),
                    "mean_unc_s": float(np.mean([r["_unc_s"] for r in rows]))}
    return out


def provenance() -> dict:
    import scipy
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    return {"commit": commit, "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("outdir")
    args = ap.parse_args()
    scen = json.loads(Path(args.config).read_text())
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done: dict[int, dict] = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done[r["seed"]] = r
    todo = [seed_for(int(scen["base_seed"]), r) for r in range(int(scen["n_rep"]))
            if seed_for(int(scen["base_seed"]), r) not in done]
    if todo:
        with rows_path.open("a") as fh:
            for s in todo:
                row = run_replicate(s, scen)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"rep seed={s} theta={row['theta']:.6f} "
                      f"({row['_fit_s'] + row['_unc_s']:.1f}s)", flush=True)
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    summ = summarize(rows, scen)
    prov = provenance()
    import hashlib
    (out / "manifest.json").write_text(json.dumps(
        {"config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
         "scen": scen, "provenance": prov, "n_rows": len(rows)},
        sort_keys=True, indent=1) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"scen": scen, "summary": summ, "provenance": prov},
        sort_keys=True, indent=1) + "\n")
    print(json.dumps(summ, sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
