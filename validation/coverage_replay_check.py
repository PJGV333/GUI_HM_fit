"""Re-run saved campaign replicates and compare them with the stored records.

    python -m validation.coverage_replay_check RUN_DIR --per-cell 2 --workers 8 [--select REGEX] [--json OUT]

For each selected cell, ``--per-cell`` replicate indices (spread over the stored ones,
deterministic) are recomputed with the *current* code via ``run_replicate`` (same seeds)
and every numeric field (est, se, lo, hi per method and parameter) is compared with the
stored value. Reports the maximum absolute and relative differences per method and the
number of mismatches beyond ``--rtol``/``--atol``. Used (i) to check that commits after
the data commit do not change campaign numerics and (ii) to verify the one-step
optimisation (docs/STABILITY_PHASE2_REPORT.md).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from validation import coverage_campaign as cc

FIELDS = ("est", "se", "lo", "hi")


def _load(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[int(r["rep"])] = r
    return rows


def _job(args):
    cell, rep, methods = args
    return cell["id"], rep, cc.run_replicate(cell, rep, methods)


def compare(old: dict, new: dict, rtol: float, atol: float) -> list[dict]:
    out = []
    if bool(old.get("fit_ok")) != bool(new.get("fit_ok")):
        return [{"method": "-", "field": "fit_ok", "abs": math.inf, "rel": math.inf, "bad": True}]
    for m, rec in (old.get("methods") or {}).items():
        new_rec = (new.get("methods") or {}).get(m)
        if new_rec is None or ("error" in rec) != ("error" in new_rec):
            out.append({"method": m, "field": "presence/error", "abs": math.inf, "rel": math.inf, "bad": True})
            continue
        if "error" in rec:
            continue
        for f in FIELDS:
            a = np.asarray(rec[f], float)
            b = np.asarray(new_rec[f], float)
            both_nan = np.isnan(a) & np.isnan(b)
            if np.any(np.isnan(a) != np.isnan(b)):
                out.append({"method": m, "field": f, "abs": math.inf, "rel": math.inf, "bad": True})
                continue
            d = np.where(both_nan, 0.0, np.abs(a - b))
            scale = np.where(both_nan, 1.0, np.maximum(np.abs(a), 1e-300))
            bad = bool(np.any(d > atol + rtol * np.abs(np.where(both_nan, 0.0, a))))
            out.append({"method": m, "field": f, "abs": float(d.max(initial=0.0)),
                        "rel": float((d / scale).max(initial=0.0)), "bad": bad})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--grid", default="validation/configs/coverage/primary_grid.json")
    ap.add_argument("--select", default=None)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--methods", default=None, help="restrict recomputation to these methods")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    grid = {c["id"]: c for c in json.loads(Path(a.grid).read_text(encoding="utf-8"))["cells"]}
    run_dir = Path(a.run_dir)
    jobs, stored = [], {}
    methods = a.methods.split(",") if a.methods else None
    for path in sorted(run_dir.glob("*.jsonl")):
        cid = path.stem
        if cid == "manifest" or cid not in grid or (a.select and not re.search(a.select, cid)):
            continue
        rows = _load(path)
        reps = sorted(rows)
        if not reps:
            continue
        pick = sorted({reps[int(round(i * (len(reps) - 1) / max(1, a.per_cell - 1)))] for i in range(a.per_cell)})
        for rep in pick:
            stored[(cid, rep)] = rows[rep]
            jobs.append((grid[cid], rep, methods))
    results = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as ex:
        for cid, rep, new in ex.map(_job, jobs):
            old = stored[(cid, rep)]
            if methods:
                old = {**old, "methods": {m: v for m, v in old["methods"].items() if m in methods}}
            for d in compare(old, new, a.rtol, a.atol):
                results.append({"cell": cid, "rep": rep, **d})
    by_m: dict[str, dict] = {}
    for r in results:
        s = by_m.setdefault(r["method"], {"n": 0, "bad": 0, "max_abs": 0.0, "max_rel": 0.0})
        s["n"] += 1
        s["bad"] += int(r["bad"])
        s["max_abs"] = max(s["max_abs"], r["abs"])
        s["max_rel"] = max(s["max_rel"], r["rel"])
    summary = {"replicates": len(jobs), "cells": len({j[0]["id"] for j in jobs}), "rtol": a.rtol, "atol": a.atol,
               "by_method": by_m, "mismatches": [r for r in results if r["bad"]][:50]}
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=1))
    if summary["mismatches"]:
        print("first mismatches:", json.dumps(summary["mismatches"][:5], indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
