"""Mechanism evidence: expected profiled RSS vs theta (Stage C).

For a grid of log K, average over M noise realizations of the FIXED pilot
signal: E[RSS_ols(theta)], E[RSS_pen(theta)] and the clipping-induced excess
E[RSS_pen - RSS_ols](theta), plus the active (negative) coefficient fraction.
The excess term tilts the penalized profile toward larger log K, which is the
mechanism behind the positive bias measured by validation/bias_experiment.py.

Writes validation/results/bias/mechanism_profile.csv and prints a table.

  QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.mechanism_profile
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "results" / "bias" / "mechanism_profile.csv"


def main() -> None:
    from hmfit_core.processors import spectroscopy_processor as sp
    from validation.bias_experiment import dataset, reference_C

    scen = {"points": 8, "noise": 2e-4, "truth": 3.0, "g_max": 2e-3}
    d = dataset(1, scen)                 # noise-free signal (noise=0 branch)
    c, eps, base = d["c"], d["eps"], d["y"]
    sigma, M_real = 2e-4, 300
    rng = np.random.default_rng(4242)
    rows = []
    for theta in np.linspace(2.90, 3.10, 11):
        C = reference_C(c, theta)
        rss_o = rss_p = negf = 0.0
        for _ in range(M_real):
            y = base + rng.normal(0.0, sigma, base.shape)
            a_o = np.linalg.lstsq(C, y, rcond=None)[0]
            a_p = sp._exact_soft_penalty_columns(C, y, mu=1e-2)
            ro, rp = y - C @ a_o, y - C @ a_p
            rss_o += float((ro * ro).sum())
            rss_p += float((rp * rp).sum())
            negf += float((a_o < 0.0).mean())
        rss_o /= M_real; rss_p /= M_real; negf /= M_real
        rows.append((theta, rss_o, rss_p, rss_p - rss_o, negf))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        fh.write("theta,E_RSS_ols,E_RSS_pen,E_excess,negfrac\n")
        for r in rows:
            fh.write(",".join(f"{v:.6e}" for v in r) + "\n")
    th = np.array([r[0] for r in rows])
    for j, name in ((1, "ols"), (2, "pen")):
        v = np.array([r[j] for r in rows])
        p = np.polyfit(th, v, 2)
        print(f"argmin E[RSS_{name}] (quadratic fit): {-p[1] / (2 * p[0]):.5f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
