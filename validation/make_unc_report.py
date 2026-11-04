"""Reproducible tables + figures for docs/MONTE_CARLO_RESULTS.md (Stage F).

Reads curated summaries from validation/results/unc/*/summary.json and emits
a markdown comparison table per scenario plus figures under
docs/figures/uncertainty/.

  .venv/bin/python -m validation.make_unc_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "validation" / "results" / "unc"
FIGS = ROOT / "docs" / "figures" / "uncertainty"

METHODS = ("analytic", "hc1", "hc2", "hc3", "profile",
           "lin_wild_el", "lin_wild_spec",
           "onestep_spec", "onestep_wl",
           "refit_wild_el", "refit_wild_spec", "refit_resample")


def f(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    lines = []
    for sp in sorted(RES.glob("*/summary.json")):
        d = json.loads(sp.read_text())
        s, scen = d["summary"], d["scen"]
        lines.append(f"\n### {scen['name']} (n={s['n']}, estimador="
                     f"{'OLS' if scen['estimator']=='ols' else 'pen mu='+str(scen['estimator'])}, "
                     f"ruido={scen.get('noise_model','iid')})\n")
        t = s["theta"]
        lines.append(f"log K: sesgo {t['bias']:+.5f} ± {t['mc_se_bias']:.5f} (MC), "
                     f"SD {t['sd']:.5f}, RMSE {t['rmse']:.5f}\n")
        lines.append("| método | SE medio | SE/SD | cov95 | cov68 | anchura95 | cov95 basic | cov95 bca | sesgo boot |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in METHODS:
            m = s.get(k)
            if not m:
                continue
            lines.append(f"| {k} | {f(m.get('mean_se'))} | {f(m.get('se_over_mc_sd'),3)} | "
                         f"{f(m.get('cov95'),3)} | {f(m.get('cov68'),3)} | "
                         f"{f(m.get('mean_width95'))} | {f(m.get('cov95_basic'),3)} | "
                         f"{f(m.get('cov95_bca'),3)} | {f(m.get('mean_boot_bias'))} |")
        # figure: coverage vs width scatter
        pts = [(s[k]["mean_width95"], s[k]["cov95"], k) for k in METHODS
               if s.get(k) and s[k].get("cov95") is not None and s[k].get("mean_width95")]
        if pts:
            fig, ax = plt.subplots(figsize=(5.2, 4))
            for x, y, k in pts:
                ax.scatter(x, y, s=40)
                ax.annotate(k.replace("_", " "), (x, y), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            ax.axhline(0.95, color="k", lw=0.6, ls="--")
            ax.set_xlabel("mean 95% interval width (log10 K)")
            ax.set_ylabel("empirical coverage")
            ax.set_title(f"{scen['name']}: coverage vs width")
            fig.tight_layout()
            fig.savefig(FIGS / f"cov_{scen['name']}.png")
            plt.close(fig)
    (RES / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("figures in", FIGS)


if __name__ == "__main__":
    main()
