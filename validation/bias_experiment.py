"""Paired-estimator bias experiments for UV-Vis 1:1 (Stage B/C/D evidence).

Compares three absorptivity estimators on identical synthetic datasets and an
identical exact scalar argmin of the production profiled objective, so all
differences are paired and free of optimizer or Monte-Carlo confounding:

  ols      A = argmin ||C a - y||^2                       (unconstrained)
  pen(mu)  A = argmin 1/2||C a - y||^2 + mu/2||min(a,0)||^2  (production
           _exact_soft_penalty_columns; legacy objective excludes the penalty)
  nnls     A = argmin_{A>=0} ||C a - y||^2                (production
           _exact_nnls_columns)

theta = argmin over log K of the profiled residual RMS, matching the
production legacy objective bit-for-bit (verified in tests).

Independent references (not copies of production code paths):
  reference_C          closed-form 1:1 mass-balance equilibrium
  reference_rss_profile  per-wavelength lstsq/nnls RSS using reference_C
  theta_ref            scipy bounded minimize_scalar on that RSS

Usage:
  QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python -m validation.bias_experiment <config.json> <outdir>

Configs are JSON scenario dicts (see configs/*.json). Runs are resumable:
each replicate row is appended to <outdir>/<name>.rows.jsonl and skipped on
re-invocation with the same scenario. Provenance (commit, config digest,
source digest, dependency versions) is written to <outdir>/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar, nnls as sp_nnls

ROOT = Path(__file__).resolve().parents[1]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source_digest():
    tracked = subprocess.run(["git", "ls-files", "--", "*.py"], cwd=ROOT, check=True,
                             stdout=subprocess.PIPE, text=True).stdout.splitlines()
    paths = {ROOT / name for name in tracked}
    paths.update((ROOT / "validation").glob("*.py"))
    h = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            name = str(path.relative_to(ROOT)).encode()
            data = path.read_bytes()
            h.update(len(name).to_bytes(8, "big")); h.update(name)
            h.update(len(data).to_bytes(8, "big")); h.update(data)
    return h.hexdigest()


def _environment():
    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__}


MODEL = [[1, 0], [0, 1], [1, 1]]


def dataset(seed: int, scen: dict) -> dict:
    """Same equations as tests/phase1_cases._spectro_dataset, parameterized."""
    from hmfit_core.solvers import NewtonRaphson
    n_points = int(scen.get("points", 8))
    g_max = float(scen.get("g_max", 2e-3))
    truth = float(scen.get("truth", 3.0))
    c_H = float(scen.get("c_H", 1e-3))
    amp = float(scen.get("amp", 1.2e3))
    sig = float(scen.get("sig", 25.0))
    n_wl = int(scen.get("n_wl", 31))
    wl = np.linspace(float(scen.get("wl_lo", 240.0)), float(scen.get("wl_hi", 600.0)), n_wl)
    c = np.column_stack([np.full(n_points, c_H), np.linspace(0.0, g_max, n_points)])
    M = np.asarray(MODEL, dtype=float).T
    solver = NewtonRaphson(c, M, nas=[], model_sett="Free")
    C = solver.concentraciones(np.asarray([truth], dtype=float))[0]
    centers = np.linspace(float(scen.get("c0_lo", 265.0)), float(scen.get("c0_hi", 380.0)), C.shape[1])
    eps = np.vstack([amp * np.exp(-((wl - cc) ** 2) / (2 * sig ** 2)) for cc in centers])
    kind = scen.get("eps_kind", "default")
    if kind == "G0":
        eps[1] = 0.0
    elif kind == "HG0":
        eps[2] = 0.0
    y = C @ eps
    noise = float(scen.get("noise", 2e-4))
    model = scen.get("noise_model", "iid")
    if noise > 0:
        rng = np.random.default_rng(seed)
        if model == "iid":
            y = y + rng.normal(0.0, noise, y.shape)
        elif model == "hetero":
            # per-point scale ramp: sigma_i = noise * (1 + ramp * i/(m-1))
            ramp = float(scen.get("noise_ramp", 2.0))
            scale = 1.0 + ramp * np.linspace(0.0, 1.0, n_points)
            y = y + rng.normal(0.0, 1.0, y.shape) * (noise * scale)[:, None]
        elif model == "ar1":
            # AR(1) correlation across wavelengths within each spectrum
            rho = float(scen.get("noise_rho", 0.6))
            e = rng.normal(0.0, 1.0, y.shape)
            z = np.empty_like(e)
            z[:, 0] = e[:, 0]
            for j in range(1, n_wl):
                z[:, j] = rho * z[:, j - 1] + np.sqrt(1.0 - rho * rho) * e[:, j]
            y = y + noise * z
        else:
            raise ValueError(f"unknown noise_model {model!r}")
    return {"c": c, "M": M, "wl": wl, "eps": eps, "y": y, "C": C}


def profile_objective(c, Y, mode="soft_penalty", mu=1e-2):
    """theta -> production-equivalent profiled RMS (legacy objective, unit weights)."""
    from hmfit_core.solvers import NewtonRaphson
    from hmfit_core.processors import spectroscopy_processor as sp
    solver = NewtonRaphson(c, np.asarray(MODEL, dtype=float).T, nas=[], model_sett="Free")

    def f(logK):
        C = solver.concentraciones(np.asarray([logK], dtype=float))[0]
        if mode == "ols":
            A = np.linalg.lstsq(C, Y, rcond=None)[0]
        elif mode == "soft_penalty":
            A = sp._exact_soft_penalty_columns(C, Y, mu=float(mu))
        elif mode == "nnls_hard":
            A = sp._exact_nnls_columns(C, Y)
        else:
            raise ValueError(mode)
        r = C @ A - Y
        return float(np.sqrt(np.mean(r ** 2)))
    return f


def argmin_1d(f, lo, hi, coarse=401, tol=1e-9):
    xs = np.linspace(lo, hi, coarse)
    vs = np.array([f(x) for x in xs])
    i = int(np.argmin(vs))
    edge = i in (0, len(xs) - 1)
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    c0, d0 = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c0), f(d0)
    while b - a > tol:
        if fc < fd:
            b, d0, fd = d0, c0, fc
            c0 = b - gr * (b - a)
            fc = f(c0)
        else:
            a, c0, fc = c0, d0, fd
            d0 = a + gr * (b - a)
            fd = f(d0)
    x = 0.5 * (a + b)
    return x, float(f(x)), edge


def reference_C(c, theta):
    """Closed-form 1:1 equilibrium (independent of the NewtonRaphson solver)."""
    K = 10.0 ** float(theta)
    HT, GT = c[:, 0], c[:, 1]
    S = HT + GT + 1.0 / K
    disc = np.maximum(S * S - 4.0 * HT * GT, 0.0)
    x = 0.5 * (S - np.sqrt(disc))
    return np.column_stack([HT - x, GT - x, x])


def reference_rss_profile(c, Y, kind="ols"):
    def rss(theta):
        C = reference_C(c, theta)
        total = 0.0
        for lam in range(Y.shape[1]):
            y = Y[:, lam]
            a = np.linalg.lstsq(C, y, rcond=None)[0] if kind == "ols" else sp_nnls(C, y)[0]
            total += float(np.dot(y - C @ a, y - C @ a))
        return total
    return rss


def seed_for(base_seed: int, replicate: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), replicate]).generate_state(1)[0])


def run_replicate(seed: int, scen: dict) -> dict:
    d = dataset(seed, scen)
    c, Y, truth = d["c"], d["y"], float(scen.get("truth", 3.0))
    lo, hi = scen.get("scan_lo", truth - 1.0), scen.get("scan_hi", truth + 1.0)
    out = {"seed": seed, "truth": truth, "edge": {}}
    estimators = {"ols": ("ols", 0.0), "nnls": ("nnls_hard", 0.0)}
    for mu in scen.get("mus", [1e-2]):
        estimators[f"pen{mu:g}"] = ("soft_penalty", float(mu))
    for name, (mode, mu) in estimators.items():
        f = profile_objective(c, Y, mode=mode, mu=mu)
        x, fx, edge = argmin_1d(f, lo, hi)
        out[name] = x
        out[name + "_f"] = fx
        out[name + "_ftrue"] = f(truth)
        if edge:
            out["edge"][name] = True
    mu0 = float(scen.get("mus", [1e-2])[0])
    from hmfit_core.solvers import NewtonRaphson
    solver = NewtonRaphson(c, np.asarray(MODEL, dtype=float).T, nas=[], model_sett="Free")
    Cstar = solver.concentraciones(np.asarray([out[f"pen{mu0:g}"]], dtype=float))[0]
    A_ols = np.linalg.lstsq(Cstar, Y, rcond=None)[0]
    out["act_frac"] = float((A_ols < 0.0).mean())
    out["max_solver_vs_ref_C"] = float(np.max(np.abs(Cstar - reference_C(c, out[f"pen{mu0:g}"]))))
    for kind, key in (("ols", "ols"), ("nnls", "nnls")):
        res = minimize_scalar(reference_rss_profile(c, Y, kind=kind), bounds=(lo, hi),
                              method="bounded", options={"xatol": 1e-9})
        out[key + "_ref"] = float(res.x)
    return out


def summarize(rows, scen):
    n = len(rows)
    truth = float(scen.get("truth", 3.0))
    summ = {"n": n, "truth": truth}
    skip = {"seed", "rep", "truth"}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))
            and not isinstance(rows[0][k], bool) and k not in skip]
    for k in keys:
        v = np.array([row[k] for row in rows], dtype=float)
        summ[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else None,
                   "bias": float(v.mean() - truth),
                   "rmse": float(np.sqrt(np.mean((v - truth) ** 2))),
                   "frac_pos_dev": float(np.mean(v > truth))}
    est_keys = [k for k in keys if not k.endswith(("_f", "_ftrue"))]
    for k in [k for k in est_keys if k.startswith(("pen", "nnls"))]:
        dv = np.array([row[k] - row["ols"] for row in rows])
        summ[k + "_minus_ols"] = {"mean": float(dv.mean()),
                                  "sd": float(dv.std(ddof=1)) if n > 1 else None,
                                  "frac_pos_dev": float(np.mean(dv > 0))}
    return summ


def run_experiment(scen: dict, outdir: str, workers: int = 1) -> dict:
    name = scen["name"]
    nrep = int(scen["n_rep"])
    identity = {"scenario": scen, "scenario_sha256": _digest(scen),
                "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                         stdout=subprocess.PIPE, text=True).stdout.strip(),
                "source_sha256": _source_digest(), "versions": _environment()}
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir_path / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("scenario", "scenario_sha256", "commit", "source_sha256", "versions"):
            if previous.get(key) != identity[key]:
                raise ValueError(f"manifest incompatible: {key} changed")
    else:
        (outdir_path / "manifest.json.tmp").write_text(_canonical(identity).decode() + "\n")
        os.replace(outdir_path / "manifest.json.tmp", manifest_path)
    rows_path = outdir_path / f"{name}.rows.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            done[row["rep"]] = row
    todo = [r for r in range(nrep) if r not in done]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    if todo:
        jobs = [(seed_for(scen.get("base_seed", 0), r), r, scen) for r in todo]
        if workers > 1:
            with mp.get_context("fork").Pool(min(workers, len(todo))) as pool:
                results = pool.map(_rep_worker, jobs)
        else:
            results = [_rep_worker(j) for j in jobs]
        with rows_path.open("a") as fh:
            for row in results:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    rows.sort(key=lambda r: r["rep"])
    summ = summarize(rows, scen)
    with open(outdir_path / f"{name}.json", "w") as fh:
        json.dump({"summary": summ, "scen": scen, "rows": rows,
                   "provenance": {k: identity[k] for k in ("commit", "source_sha256", "versions")}},
                  fh, sort_keys=True)
    edge_reps = [row["rep"] for row in rows if row.get("edge")]
    return {"scenario": name, "completed": len(rows), "summary": summ,
            "edge_flagged_replicates": edge_reps}


def _rep_worker(job):
    seed, rep, scen = job
    t0 = time.perf_counter()
    row = run_replicate(seed, scen)
    row["rep"] = rep
    row["seconds"] = time.perf_counter() - t0
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_experiment(json.loads(args.config.read_text()), str(args.outdir), args.workers)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
