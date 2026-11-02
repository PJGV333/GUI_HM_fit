"""Summarize completed pilot rows; all estimates are conditional on fit success."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def summarize(rows):
    attempted = len(rows)
    valid = [r for r in rows if str(r.get("success")).lower() == "true"]
    n = len(valid)
    out = {"attempted": attempted, "converged": n, "failures": attempted - n,
           "failure_fraction": (attempted - n) / attempted if attempted else None,
           "covered": 0, "coverage95": None, "coverage_mc_se": None,
           "coverage_wilson_low": None, "coverage_wilson_high": None,
           "bias": None, "bias_mc_se": None, "rmse": None, "sd": None,
           "mean_se": None, "se_over_sd": None, "mean_interval_width": None,
           "median_interval_width": None, "warning": "pilot only; metrics conditional on successful fits"}
    if not n:
        return out
    truth = np.array([float(r["truth"]) for r in valid])
    est = np.array([float(r["estimate"]) for r in valid])
    se = np.array([float(r["se"]) for r in valid])
    lo = np.array([float(r["ci_low"]) for r in valid])
    hi = np.array([float(r["ci_high"]) for r in valid])
    if not np.all(np.isfinite(np.concatenate((truth, est, se, lo, hi)))) or np.any(lo > hi):
        raise ValueError("successful rows must contain finite estimates and ordered intervals")
    errors = est - truth
    covered = int(np.count_nonzero((lo <= truth) & (truth <= hi)))
    coverage = covered / n
    interval = binomtest(covered, n).proportion_ci(confidence_level=0.95, method="wilson")
    sd = float(np.std(est, ddof=1)) if n >= 2 else None
    out.update(covered=covered, coverage95=coverage,
               coverage_mc_se=math.sqrt(coverage * (1.0 - coverage) / n),
               coverage_wilson_low=float(interval.low), coverage_wilson_high=float(interval.high),
               bias=float(np.mean(errors)), bias_mc_se=(float(np.std(errors, ddof=1)) / math.sqrt(n) if n >= 2 else None),
               rmse=float(np.sqrt(np.mean(errors ** 2))), sd=sd, mean_se=float(np.mean(se)),
               se_over_sd=(float(np.mean(se)) / sd if sd is not None and sd > 0 else None),
               mean_interval_width=float(np.mean(hi - lo)), median_interval_width=float(np.median(hi - lo)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print(json.dumps(summarize(rows), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
