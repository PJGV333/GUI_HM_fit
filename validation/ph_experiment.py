"""Stage G: Monte Carlo campaigns for acid-base (NMR shifts and pH
potentiometry) through the production run_acid_base + analytic errors.

Resumable JSONL; manifest with config sha / commit / versions. Generator
equations mirror tests/phase1_cases.acid_base_nmr_case and
.potentiometry_case.

Usage:
  QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.ph_experiment \
      validation/configs/ph/<x>.json <outdir>

Scenario keys: kind ("nmr"|"potentiometry"), base_seed, n_rep, pka_true
(list), pka0 (list), noise, n_points (nmr: pH points; pot: volumes),
ph_lo, ph_hi (nmr), analyte_concentration.
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


def _dataset(path: Path, seed: int, scen: dict) -> None:
    kind = scen["kind"]
    pka = list(scen["pka_true"])
    noise = float(scen.get("noise", 3e-3))
    if kind == "nmr":
        from hmfit_core.acid_base import (distribution_fractions_from_pH,
                                          pka_to_log_beta)
        n = int(scen.get("n_points", 25))
        pH = np.linspace(float(scen.get("ph_lo", 2.0)), float(scen.get("ph_hi", 10.0)), n)
        frac = distribution_fractions_from_pH(pH, pka_to_log_beta(pka))
        shifts = np.linspace(7.9, 7.0, frac.shape[1])
        y = frac @ shifts + np.random.default_rng(seed).normal(0.0, noise, n)
        pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
    elif kind == "potentiometry":
        from hmfit_core.acid_base import make_simple_acid_base_system
        from hmfit_core.potentiometry import (PotentiometryExperiment,
                                              simulate_pH_titration)
        ca = float(scen.get("analyte_concentration", 1e-2))
        system = make_simple_acid_base_system(analytical_concentration=ca,
                                              pka=pka, base_charge=-len(pka))
        vol = np.linspace(0.0, 10.0 * len(pka) * 1.8, int(scen.get("n_points", 60)))
        exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol,
                                      analyte_concentration=ca,
                                      titrant_concentration=ca, titrant_type="base")
        pH = simulate_pH_titration(exp, system) + np.random.default_rng(seed).normal(0.0, noise, vol.size)
        pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
    else:
        raise ValueError(f"unknown kind {kind!r}")


def _cfg(scen: dict, path: Path) -> dict:
    cfg = {"file_path": str(path), "data_type": scen["kind"],
           "pka_initial": ",".join(str(v) for v in scen["pka0"]),
           "analyte_concentration": float(scen.get("analyte_concentration",
                                                   1e-3 if scen["kind"] == "nmr" else 1e-2))}
    if scen["kind"] == "potentiometry":
        ca = float(scen.get("analyte_concentration", 1e-2))
        cfg.update({"titrant_concentration": ca, "initial_volume": 10.0,
                    "titrant_type": "base"})
    cfg["base_charge"] = -len(scen["pka_true"])
    return cfg


def seed_for(base: int, r: int) -> int:
    from numpy.random import SeedSequence
    return int(SeedSequence([base, r]).generate_state(1)[0])


def run_replicate(seed: int, scen: dict) -> dict:
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from hmfit_core.run_acid_base import run_acid_base

    tmp = Path(tempfile.mkdtemp())
    path = tmp / ("d" + ("_.csv"))
    _dataset(path, seed, scen)
    t0 = time.perf_counter()
    res = run_acid_base(_cfg(scen, path), progress_cb=lambda _m: None)
    out = compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    P = out["export_frames"]["Parameters"]
    est = P["Estimate"].astype(float).tolist()
    se = pd.to_numeric(P["SE"], errors="coerce").astype(float).tolist()
    return {"seed": int(seed), "theta": est, "se": se, "dof": out["dof"],
            "seconds": time.perf_counter() - t0}


def summarize(rows: list[dict], scen: dict) -> dict:
    from scipy import stats
    pka = np.asarray(scen["pka_true"], dtype=float)
    TH = np.array([r["theta"] for r in rows], dtype=float)
    SE = np.array([r["se"] for r in rows], dtype=float)
    dof = int(rows[0]["dof"])
    q95, q68 = stats.t.ppf(0.975, dof), stats.t.ppf(0.84, dof)
    n = len(rows)
    per = {}
    for j, name in enumerate([f"pKa{j+1}" for j in range(len(pka))]):
        th, se = TH[:, j], SE[:, j]
        cov95 = ((th - q95 * se <= pka[j]) & (pka[j] <= th + q95 * se))
        cov68 = ((th - q68 * se <= pka[j]) & (pka[j] <= th + q68 * se))
        per[name] = {
            "bias": float(th.mean() - pka[j]),
            "median_bias": float(np.median(th) - pka[j]),
            "mc_se_bias": float(th.std(ddof=1) / np.sqrt(n)),
            "sd": float(th.std(ddof=1)),
            "rmse": float(np.sqrt(((th - pka[j]) ** 2).mean())),
            "mean_se": float(se.mean()),
            "se_over_mc_sd": float(se.mean() / th.std(ddof=1)),
            "cov95": float(cov95.mean()), "cov68": float(cov68.mean()),
        }
    swapped = float(np.mean([1.0 if (len(pka) == 2 and TH[i][0] > TH[i][1]) else 0.0
                             for i in range(n)])) if len(pka) == 2 else None
    return {"n": n, "dof": dof, "per_parameter": per,
            "label_swap_rate": swapped}


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
                print(f"rep seed={s} theta={row['theta']}", flush=True)
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
