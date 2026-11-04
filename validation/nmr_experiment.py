"""Stage G: Monte Carlo campaign for 1H NMR 1:1 (production path).

Same generator equations as tests/phase1_cases.nmr_case (shifts from
host-bound fractions, two signals, ppm noise), fitted through the
production NMR processor. Resumable JSONL checkpoints; manifest with
commit/versions/config sha.

Usage:
  QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.nmr_experiment \
      validation/configs/nmr/nmr_11_iid.json <outdir>

Scenario keys: name, base_seed, n_rep, n_points, noise (ppm), truth (log K),
g_max, k0, bounds_lo, bounds_hi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def nmr_dataset(path: Path, seed: int, scen: dict) -> None:
    """Generator identical to tests/phase1_cases.nmr_case (same equations)."""
    from hmfit_core.solvers import NewtonRaphson

    n = int(scen.get("n_points", 12))
    noise = float(scen.get("noise", 2e-3))
    g_max = float(scen.get("g_max", 5e-3))
    k_true = np.asarray([float(scen.get("truth", 3.0))])
    model = [[1, 0], [0, 1], [1, 1]]
    M = np.asarray(model, dtype=float).T
    H = np.full(n, 1e-3)
    G = np.linspace(0.0, g_max, n)
    solver = NewtonRaphson(np.column_stack([H, G]), M, nas=[1], model_sett="Free")
    C, Co = solver.concentraciones(k_true)
    host_species = [i for i, row in enumerate(model) if row[0] > 0]
    frac = Co[:, host_species] * np.asarray([model[i][0] for i in host_species]) / H[:, None]
    deltas1 = np.linspace(7.0, 7.6, len(host_species))
    deltas2 = np.linspace(8.0, 7.5, len(host_species))
    rng = np.random.default_rng(seed)
    s1 = frac @ deltas1 + rng.normal(0, noise, n)
    s2 = frac @ deltas2 + rng.normal(0, noise, n)
    pd.DataFrame({"H": H, "G": G, "s1": s1, "s2": s2}).to_excel(path, index=False,
                                                                sheet_name="S")


def seed_for(base: int, r: int) -> int:
    from numpy.random import SeedSequence
    return int(SeedSequence([base, r]).generate_state(1)[0])


def run_replicate(seed: int, scen: dict) -> dict:
    from hmfit_core.processors.nmr_processor import process_nmr_data

    tmp = Path(tempfile.mkdtemp())
    path = tmp / "n.xlsx"
    nmr_dataset(path, seed, scen)
    lo = float(scen.get("bounds_lo", 0.0))
    hi = float(scen.get("bounds_hi", 6.0))
    t0 = time.perf_counter()
    out = process_nmr_data(
        str(path), "S", "S", ["H", "G"], ["s1", "s2"], "H", "G", MODEL,
        [float(scen.get("k0", 2.5))], [{"min": lo, "max": hi}],
        "Newton-Raphson", "powell", "Free", [1], k_fixed=None)
    ed = out["export_data"]
    return {"seed": int(seed), "theta": float(ed["k"][0]),
            "se": float(ed["SE_log10K"][0]),
            "ok": bool(ed.get("converged", True)),
            "seconds": time.perf_counter() - t0}


def summarize(rows: list[dict], scen: dict) -> dict:
    from scipy import stats
    truth = float(scen.get("truth", 3.0))
    th = np.array([r["theta"] for r in rows])
    se = np.array([r["se"] for r in rows])
    df = int(scen.get("n_points", 12)) * 2 - 3  # 2 signals x n obs - (1 K + 2 shifts)
    q95, q68 = stats.t.ppf(0.975, df), stats.t.ppf(0.84, df)
    cov95 = ((th - q95 * se <= truth) & (truth <= th + q95 * se))
    cov68 = ((th - q68 * se <= truth) & (truth <= th + q68 * se))
    n = len(rows)
    return {
        "n": n, "truth": truth,
        "bias": float(th.mean() - truth),
        "mc_se_bias": float(th.std(ddof=1) / np.sqrt(n)),
        "sd": float(th.std(ddof=1)),
        "rmse": float(np.sqrt(((th - truth) ** 2).mean())),
        "mean_se": float(se.mean()),
        "se_over_mc_sd": float(se.mean() / th.std(ddof=1)),
        "cov95": float(cov95.mean()), "cov68": float(cov68.mean()),
        "cov95_mc_se": float(np.sqrt(cov95.mean() * (1 - cov95.mean()) / n)),
        "dof_assumed": df,
    }


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
    done: set[int] = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            done.add(json.loads(line)["seed"])
    todo = [seed_for(int(scen["base_seed"]), r) for r in range(int(scen["n_rep"]))
            if seed_for(int(scen["base_seed"]), r) not in done]
    if todo:
        with rows_path.open("a") as fh:
            for s in todo:
                row = run_replicate(s, scen)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"rep seed={s} theta={row['theta']:.6f} se={row['se']:.5f}",
                      flush=True)
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    summ = summarize(rows, scen)
    prov = provenance()
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
