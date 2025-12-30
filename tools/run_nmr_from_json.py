#!/usr/bin/env python3
import json
import os
import sys


def _default_config_path() -> str:
    return "/mnt/data/hmfit_confignmrT2.json"


def _fallback_config_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "hmfit_confignmrT2.json")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_stat(stats_table, key):
    for row in stats_table or []:
        if len(row) >= 2 and row[0] == key:
            return row[1]
    return None


def main() -> int:
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = _default_config_path()

    if not os.path.exists(cfg_path):
        fallback = _fallback_config_path()
        if os.path.exists(fallback):
            cfg_path = fallback
        else:
            print(f"[skip] Config not found: {cfg_path}")
            return 0

    cfg = _load_config(cfg_path)
    data_path = cfg.get("file_path")
    if not data_path or not os.path.exists(data_path):
        print(f"[skip] Data file not found: {data_path}")
        return 0

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from hmfit_core.run_nmr import run_nmr

    result = run_nmr(cfg, progress_cb=print)
    if result.get("error"):
        print(f"[error] {result['error']}")
        return 1

    export = result.get("export_data") or {}
    stats_table = export.get("stats_table") or []
    rms = _extract_stat(stats_table, "RMS")

    print("\nResults")
    print(f"Config: {cfg_path}")
    print(f"RMS: {rms}")

    constants = result.get("constants") or []
    if constants:
        print("\nSE / %Error")
        for c in constants:
            name = c.get("name")
            log10k = c.get("log10K")
            se = c.get("SE_log10K")
            perc = c.get("percent_error")
            print(f"{name}: log10K={log10k} SE_log10K={se} %Error={perc}")

    se_list = export.get("SE_log10K") or []
    se_vals = [abs(x) for x in se_list if isinstance(x, (int, float))]
    max_se = max(se_vals) if se_vals else None

    if rms is not None and rms > 3e-4:
        print(f"[warn] RMS > 3e-4: {rms}")
    if max_se is not None and max_se > 0.1:
        print(f"[warn] max(SE_log10K) > 0.1: {max_se}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
