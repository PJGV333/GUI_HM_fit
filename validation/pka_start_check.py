"""Do correctly ordered data converge to the true pKa when the fit starts from the old,
inverted starting points? (Phase 2 report, section "Defectos en versiones publicadas".)

    QT_QPA_PLATFORM=offscreen python -m validation.pka_start_check [--seeds 20] [--json OUT]

Data: diprotic ligand with pKa1 > pKa2 (HM fit convention), simulated for acid-base NMR,
UV-Vis and potentiometry with the GUI entry point ``run_acid_base``. Starting points:
- "old_default": 5, 7 (ascending GUI default before Phase 2),
- "old_doc_example": 4.50, 8.90 (stepwise example in the GUI / docs before Phase 2; the
  cumulative example logB 4.50 / 13.40 is the same pair),
- "new_default": 7, 5 (descending default),
- "truth_order": the true values + 0.3.
A fit "recovers" the truth if every |pKa_hat - pKa_true| < 0.1 (noise SD ~ 0.003 ppm /
0.001 AU / 0.005 pH gives SE ~ 0.01-0.03); "swapped" if the estimate is close to the
reversed pair; otherwise "other".
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd

from hmfit_core.acid_base import distribution_fractions_from_pH, make_simple_acid_base_system, pka_to_log_beta
from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
from hmfit_core.run_acid_base import run_acid_base

TRUTHS = {"doc_ligand": [8.9, 4.5], "sep": [8.0, 4.0]}
STARTS = {"old_default": [5.0, 7.0], "old_doc_example": [4.5, 8.9], "new_default": [7.0, 5.0]}


def _data(kind, truth, seed, work):
    rng = np.random.default_rng(seed)
    if kind == "pot":
        system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=truth, base_charge=-2)
        vol = np.linspace(0.0, 25.0, 40)
        exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                      titrant_concentration=1e-2, titrant_type="base")
        pH = simulate_pH_titration(exp, system) + rng.normal(0.0, 0.005, vol.size)
        path = os.path.join(work, "pot.csv")
        pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
        return {"file_path": path, "data_type": "potentiometry", "analyte_concentration": 1e-2,
                "titrant_concentration": 1e-2, "initial_volume": 10.0, "titrant_type": "base", "base_charge": -2}
    pH = np.linspace(2.0, 11.0, 25)
    frac = distribution_fractions_from_pH(pH, pka_to_log_beta(truth))
    if kind == "nmr":
        y = frac @ np.array([[7.9, 8.4], [7.4, 8.1], [7.0, 7.6]]) + rng.normal(0.0, 3e-3, (pH.size, 2))
        cols, dtype = ["H1", "H2"], "nmr"
    else:
        wl = np.linspace(250.0, 400.0, 8)
        eps = np.vstack([0.9 * np.exp(-((wl - c) ** 2) / (2 * 35.0**2)) + 0.05 for c in (270.0, 320.0, 370.0)])
        y = frac @ eps + rng.normal(0.0, 1e-3, (pH.size, wl.size))
        cols, dtype = [f"A{int(w)}" for w in wl], "spectroscopy"
    path = os.path.join(work, "ab.csv")
    pd.DataFrame(np.column_stack([pH, y]), columns=["pH"] + cols).to_csv(path, index=False)
    cfg = {"file_path": path, "data_type": dtype, "analyte_concentration": 1e-3, "base_charge": -2}
    if dtype == "spectroscopy":
        cfg["baseline"] = False
    return cfg


def classify(est, truth):
    est, truth = np.asarray(est, float), np.asarray(truth, float)
    if not np.all(np.isfinite(est)):
        return "failed"
    if np.all(np.abs(est - truth) < 0.1):
        return "recovered"
    if np.all(np.abs(est - truth[::-1]) < 0.1):
        return "swapped"
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rows = []
    for kind in ("nmr", "spec", "pot"):
        for tname, truth in TRUTHS.items():
            starts = dict(STARTS, truth_order=[v + 0.3 for v in truth])
            for seed in range(a.seeds):
                with tempfile.TemporaryDirectory() as work:
                    cfg = _data(kind, truth, 20261106 + seed, work)
                    for sname, start in starts.items():
                        try:
                            res = run_acid_base({**cfg, "pka_initial": ",".join(map(str, start))},
                                                progress_cb=lambda _m: None)
                            est = [r["value"] for r in res["export_data"]["pka_table"]]
                            chi2 = float(res["statistics"]["chi_square"])
                        except Exception as exc:  # recorded, not hidden
                            est, chi2 = [float("nan")] * 2, float("nan")
                        rows.append({"technique": kind, "truth": tname, "seed": seed, "start": sname,
                                     "estimate": est, "chi_square": chi2, "outcome": classify(est, truth)})
    df = pd.DataFrame(rows)
    table = df.groupby(["technique", "truth", "start"])["outcome"].value_counts().unstack(fill_value=0)
    print(table.to_string())
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "table": table.reset_index().to_dict(orient="records")}, fh, indent=1)


if __name__ == "__main__":
    main()
