"""Stage G: Monte Carlo campaign for reaction kinetics (first order,
second order, consecutive, reversible) through the production
hmfit.kinetics fit_global + its analytic parameter errors.

Generator mirrors tests/phase1_cases.kinetics_case: concentrations from
KineticsModel.solve_concentrations, linear spectra mixing (fixed seed 1),
i.i.d. noise on the observable matrix.

Usage:
  QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.kinetics_experiment \
      validation/configs/kin/<x>.json <outdir>

Scenario keys: name, base_seed, n_rep, mech (mechanism text), k_true (dict
name->value), k0 (list matching k_true order), y0 (dict), t (list),
n_wl, noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def seed_for(base: int, r: int) -> int:
    from numpy.random import SeedSequence
    return int(SeedSequence([base, r]).generate_state(1)[0])


def run_replicate(seed: int, scen: dict) -> dict:
    from hmfit.kinetics.data.dataset import KineticsDataset
    from hmfit.kinetics.fit.objective import GlobalKineticsObjective
    from hmfit.kinetics.fit.optimizer import fit_global
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    from hmfit.kinetics.model.kinetics_model import KineticsModel

    t = np.asarray(scen["t"], dtype=float)
    y0 = {k: float(v) for k, v in scen["y0"].items()}
    k_true = {k: float(v) for k, v in scen["k_true"].items()}
    names = list(k_true.keys())
    model = KineticsModel(parse_mechanism(scen["mech"]))
    C = model.solve_concentrations(t, y0, k_true)
    A = np.random.default_rng(1).random((C.shape[1], int(scen.get("n_wl", 12)))) \
        * float(scen.get("scale", 1.0))
    D = C @ A
    D = D + np.random.default_rng(seed).normal(0.0, float(scen.get("noise", 1e-2)), D.shape)
    obj = GlobalKineticsObjective(model, [KineticsDataset(t=t, y0=y0, D=D)],
                                  param_names=names)
    t0 = time.perf_counter()
    res = fit_global(obj, np.asarray(scen["k0"], dtype=float),
                     bounds=(0.0, np.inf), fixed_mask=None)
    return {"seed": int(seed),
            "theta": [float(res.params[n]) for n in names],
            "se": [float(e) if np.isfinite(e) else None for e in res.errors],
            "dof": int(res.result.dof),
            "seconds": time.perf_counter() - t0}


def summarize(rows: list[dict], scen: dict) -> dict:
    from scipy import stats
    names = list(scen["k_true"].keys())
    truth = np.asarray([float(scen["k_true"][n]) for n in names])
    TH = np.array([[np.nan if v is None else v for v in r["theta"]] for r in rows], dtype=float)
    n = len(rows)
    per = {}
    dof = int(rows[0]["dof"])
    for j, name in enumerate(names):
        th = TH[:, j]
        ok = np.isfinite(th)
        tr = truth[j]
        bias = float(th[ok].mean() - tr)
        sd = float(th[ok].std(ddof=1))
        per[name] = {
            "truth": tr,
            "bias": bias,
            "rel_bias": float(bias / tr),
            "mc_se_bias": float(sd / np.sqrt(ok.sum())),
            "sd": sd,
            "rmse": float(np.sqrt(np.nanmean((th[ok] - tr) ** 2))),
            "fail_rate": float(1.0 - ok.mean()),
        }
    return {"n": n, "dof": dof, "per_parameter": per}


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
