"""Reproducible figures for docs/UVVIS_BIAS_INVESTIGATION.md.

Reads the scenario summaries committed under validation/results/bias/ and
writes PNG figures under docs/figures/uvvis_bias/.

  QT_QPA_PLATFORM=offscreen .venv/bin/python -m validation.make_bias_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results" / "bias"
FIGS = ROOT / "docs" / "figures" / "uvvis_bias"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    # Fig 1: bias vs noise level (paired estimators, 40 replicates each).
    noise_scen = [("noise2e5", 2e-5), ("pilot40", 2e-4), ("noise2e3", 2e-3), ("noise2e2", 2e-2)]
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [n for _, n in noise_scen]
    for key, label, style in (("ols", "OLS (unconstrained)", "o-"),
                              ("pen0.01", "soft penalty mu=1e-2", "s-"),
                              ("nnls", "NNLS", "^--")):
        ys = [load(s)["summary"][key]["bias"] for s, _ in noise_scen]
        ax.plot(xs, ys, style, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("noise sigma (absorbance)")
    ax.set_ylabel("bias of log K (log10)"); ax.legend()
    ax.set_title("log K bias vs noise (1:1 UV-Vis, 8 points, 40 reps, paired)")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_noise_bias.png"); plt.close(fig)

    # Fig 2: bias vs penalty strength mu.
    mu = load("mu40")["summary"]
    mus = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    biases = [mu[f"pen{m:g}"]["bias"] for m in mus]
    sds = [mu[f"pen{m:g}"]["sd"] for m in mus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([m if m > 0 else 1e-5 for m in mus], biases, yerr=None, marker="o", capsize=0)
    for m, b, s in zip(mus, biases, sds):
        ax.plot([m if m > 0 else 1e-5] * 2, [b - 1.96 * s / np.sqrt(40), b + 1.96 * s / np.sqrt(40)],
                color="C0", lw=1)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(mu["ols"]["bias"], color="C1", ls=":",
                                                 label="OLS (mu = 0)")
    ax.set_xscale("log"); ax.set_xlabel("penalty mu")
    ax.set_ylabel("bias of log K (log10)"); ax.legend()
    ax.set_title("bias vs soft-penalty strength (40 reps, 95% MC error bars)")
    fig.tight_layout(); fig.savefig(FIGS / "fig2_mu_bias.png"); plt.close(fig)

    # Fig 3: paired differences pen - OLS per replicate.
    diffs = [json.loads(l)["pen0.01"] - json.loads(l)["ols"] for l in
             (RESULTS / "pilot40.rows.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diffs, bins=12)
    ax.axvline(float(np.mean(diffs)), color="r", label=f"mean {np.mean(diffs):+.5f}")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta_r=\hat\theta_{\rm pen}-\hat\theta_{\rm OLS}$")
    ax.set_ylabel("replicates"); ax.legend()
    ax.set_title("Paired estimator difference (pilot scenario, 40 reps)")
    fig.tight_layout(); fig.savefig(FIGS / "fig3_paired_diff.png"); plt.close(fig)

    # Fig 4: design/regime comparison of soft-penalty bias.
    scen_names = ["pts4", "pilot40", "pts16", "pts32", "gmax05", "gmax5", "gmax10",
                  "truth15", "truth2", "truth4", "truth5", "cH5", "epsG0", "epsSmall",
                  "epsOverlap"]
    labels = ["4 pts", "8 pts", "16 pts", "32 pts", "G<=0.5mM", "G<=5mM", "G<=10mM",
              "logK=1.5", "logK=2", "logK=4", "logK=5", "H=5mM", "epsG=0", "eps small",
              "overlap"]
    have = [n for n in scen_names if (RESULTS / f"{n}.json").exists()]
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, off, label in (("ols", -0.25, "OLS"), ("pen0.01", 0.25, "soft penalty")):
        xs = [i + off for i, n in enumerate(have)]
        ys = [load(n)["summary"][key]["bias"] for n in have]
        ax.bar(xs, ys, width=0.4, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(have))); ax.set_xticklabels([n for n in have], rotation=45, ha="right")
    ax.set_ylabel("bias of log K (log10)"); ax.legend()
    ax.set_title("bias by design and regime (40 reps each, sigma=2e-4 unless noted)")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_designs.png"); plt.close(fig)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
