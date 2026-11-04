"""Analytic SE calibration vs Monte-Carlo scatter (Stage D evidence).

For the archived pilot scenario (base seed 20261102, 8 points, sigma=2e-4):
computes for every replicate the production analytic standard error
(compute_errors_spectro_varpro, unconstrained VarPro Jacobian) evaluated at
both the unconstrained-OLS argmin and the production soft-penalty argmin, then
compares mean SE with the Monte-Carlo SD of each estimator and reports 95%
Student-t coverage of log K = 3 for both estimators.

Writes validation/results/bias/se_calibration_pilot20.json.

  QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.se_calibration [n_reps]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "results" / "bias" / "se_calibration_pilot20.json"
ROWS = ROOT / "validation" / "results" / "bias" / "pilot20.rows.jsonl"


def main(n_reps: int = 20) -> None:
    from hmfit_core.solvers import NewtonRaphson
    from hmfit_core.utils.errors import compute_errors_spectro_varpro
    from validation.bias_experiment import MODEL, dataset, seed_for

    truth = 3.0
    scen = {"points": 8, "noise": 2e-4, "truth": truth, "g_max": 2e-3}
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    rows = rows[:n_reps]
    est = {"ols": [], "pen0.01": []}
    se = {"ols": [], "pen0.01": []}
    dof = None
    for row in rows:
        d = dataset(int(row["seed"]), scen)
        solver = NewtonRaphson(d["c"], np.asarray(MODEL, dtype=float).T, nas=[], model_sett="Free")
        for key in est:
            theta = float(row[key])
            m = compute_errors_spectro_varpro(np.asarray([theta]), solver, d["y"].T,
                                              np.asarray(MODEL, dtype=float).T, [],
                                              weights=np.ones_like(d["y"]))
            est[key].append(theta)
            se[key].append(float(np.asarray(m["SE_log10K"])[0]))
            dof = int(m["dof"])
    out = {"n": len(rows), "truth": truth, "dof": dof}
    for key in est:
        e = np.array(est[key])
        s = np.array(se[key])
        q = float(t.ppf(0.975, dof))
        covered = int(np.count_nonzero((e - q * s <= truth) & (truth <= e + q * s)))
        ci = binomtest(covered, len(rows)).proportion_ci(confidence_level=0.95, method="wilson")
        sd = float(np.std(e, ddof=1))
        out[key] = {"bias": float(e.mean() - truth), "mc_sd": sd,
                    "mean_se": float(s.mean()), "se_over_sd": float(s.mean() / sd),
                    "coverage95": covered / len(rows), "covered": covered,
                    "wilson_low": float(ci.low), "wilson_high": float(ci.high),
                    "mean_width": float(np.mean(2 * q * s))}
    dv = np.array(est["pen0.01"]) - np.array(est["ols"])
    out["pen_minus_ols"] = {"mean": float(dv.mean()), "sd": float(dv.std(ddof=1)),
                            "frac_pos": float(np.mean(dv > 0))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(json.dumps(out, sort_keys=True, indent=1))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
