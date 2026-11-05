"""Export main synthetic campaign cases for manual comparison in BindFit / HypNMR.

    python -m validation.export_external_cases validation/results/coverage/external_cases

Each case is replicate 0 of a pre-registered cell (same seed as the campaign), written
as CSV and XLSX with columns: Host (M), Guest (M), then one column per signal
(absorbance per wavelength for UV-Vis, ppm per nucleus for NMR). README.md lists the
true constants and the HM fit estimate for that dataset.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from validation import coverage_campaign as cc

CASES = [
    "uvvis-11zero-10-iid-rec",
    "uvvis-11zero-10-iid-poor",
    "uvvis-12zero-12-iid-rec",
    "nmr-11-10-iid1-rec",
    "nmr-12-12-iid1-rec",
]


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = {c["id"]: c for c in cc.primary_grid()}
    readme = ["# Synthetic cases for external comparison (BindFit / HypNMR)", "",
              "Replicate 0 of each pre-registered cell (`validation/coverage_campaign.py`). "
              "Concentrations in M; UV-Vis signals are absorbances at the wavelength in the "
              "column name (nm); NMR signals in ppm.", "",
              "| case | model | true log K | HM fit log K (GUI default settings) | file |", "|---|---|---|---|---|"]
    for cid in CASES:
        cell = grid[cid]
        ss = cc.seed_for(cell["seed_id"], 0)
        data_ss = ss.spawn(1 + len(cc.ALL_METHODS))[0]
        rng = np.random.default_rng(data_ss)
        with tempfile.TemporaryDirectory() as work:
            cfg, truth = cc.GENERATORS[cell["technique"]](cell, rng, work)
            if cell["technique"] == "uvvis":
                spectra = pd.read_excel(cfg["file_path"], sheet_name="Spectra", index_col=0)
                conc = pd.read_excel(cfg["file_path"], sheet_name="Conc")
                signals = spectra.T.reset_index(drop=True)
                signals.columns = [f"A{float(w):g}nm" for w in spectra.index]
            else:
                sheet = pd.read_excel(cfg["file_path"], sheet_name="S")
                conc = sheet[["H", "G"]]
                signals = sheet[["s1", "s2"]].rename(columns={"s1": "delta1_ppm", "s2": "delta2_ppm"})
            _, est, ok, _ = cc._fit(cell, cfg)
        frame = pd.concat([conc.rename(columns={"H": "Host (M)", "G": "Guest (M)"}).reset_index(drop=True),
                           signals], axis=1)
        frame.to_csv(out_dir / f"{cid}.csv", index=False, float_format="%.8g")
        frame.to_excel(out_dir / f"{cid}.xlsx", index=False)
        model = "1:1" if cell["design"].startswith("11") else "1:2"
        readme.append(f"| {cid} | {model} | {', '.join(f'{v:g}' for v in truth)} | "
                      f"{', '.join(f'{v:.4f}' for v in np.atleast_1d(est)) if ok else 'fit failed'} | {cid}.csv / .xlsx |")
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export(Path(sys.argv[1] if len(sys.argv) > 1 else "validation/results/coverage/external_cases"))
