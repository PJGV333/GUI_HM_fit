"""Resumable, process-parallel pilot runner (not a confirmatory campaign).

Example: python -m validation.run_campaign validation/configs/uvvis_pilot.json /path/out
Only the explicitly supported scenario/method is accepted. Outputs are
append-safe per-replicate JSON checkpoints plus a deterministically sorted CSV.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys

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
    import PySide6
    from hmfit_core.utils.np_backend import USE_JAX
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pyside6": PySide6.__version__, "backend": "jax" if USE_JAX else "numpy"}


def _atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _worker(config, replicate):
    import numpy as np
    from validation.adapters import run_one
    seed = int(np.random.SeedSequence([int(config["seed"]), replicate]).generate_state(1)[0])
    return run_one(config, replicate, seed)


def run_campaign(config, directory):
    config = dict(config)
    required = {"scenario", "replicates", "seed", "points", "noise", "method", "workers"}
    if set(config) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    if config["scenario"] != "uvvis_11_iid" or config["method"] != "analytic":
        raise ValueError("Only uvvis_11_iid / analytic is implemented")
    if int(config["replicates"]) < 1 or int(config["workers"]) < 1 or int(config["points"]) < 4:
        raise ValueError("replicates/workers must be positive and points must be >=4")
    if not (float(config["noise"]) >= 0.0):
        raise ValueError("noise must be nonnegative")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, text=True).stdout.strip()
    identity = {"config": config, "config_sha256": _digest(config), "commit": commit,
                "source_sha256": _source_digest(), "versions": _environment()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("manifest incompatible with config, code or environment")
    else:
        _atomic_json(manifest_path, {**identity, "completed": 0})

    rows = {}
    for path in sorted(checkpoints.glob("*.json")):
        wrapped = json.loads(path.read_text())
        row = wrapped["row"]
        idx = int(row["replicate"])
        if wrapped.get("sha256") != _digest(row) or idx in rows or not 0 <= idx < int(config["replicates"]):
            raise ValueError(f"corrupt or duplicate checkpoint: {path.name}")
        rows[idx] = row
    pending = [i for i in range(int(config["replicates"])) if i not in rows]

    def save(row):
        if _source_digest() != identity["source_sha256"]:
            raise ValueError("source changed while campaign was running")
        idx = int(row["replicate"])
        if idx in rows or idx not in pending:
            raise ValueError("duplicate or unexpected worker result")
        _atomic_json(checkpoints / f"{idx:08d}.json", {"row": row, "sha256": _digest(row)})
        rows[idx] = row
        _atomic_json(manifest_path, {**identity, "completed": len(rows)})

    old_threads = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        for key in old_threads:
            os.environ[key] = "1"
        if int(config["workers"]) == 1:
            for idx in pending:
                save(_worker(config, idx))
        elif pending:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(config["workers"]),
                                                         mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(_worker, config, idx) for idx in pending]
                for future in concurrent.futures.as_completed(futures):
                    save(future.result())
    finally:
        for key, value in old_threads.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    _atomic_json(manifest_path, {**identity, "completed": len(rows)})
    result_path = directory / "results.csv"
    tmp = result_path.with_suffix(".csv.tmp")
    ordered = [rows[idx] for idx in sorted(rows)]
    if ordered:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(ordered[0]))
            writer.writeheader(); writer.writerows(ordered)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, result_path)
    return {"manifest": str(manifest_path), "results": str(result_path), "completed": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_campaign(json.loads(args.config.read_text()), args.output), sort_keys=True))


if __name__ == "__main__":
    main()
