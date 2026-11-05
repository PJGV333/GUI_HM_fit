"""Potentiometry with fixed pKa (Phase 1 limitation 6, Phase 2 stage 2.1.4).

Potentiometry drops fixed pKa from the optimised vector. Before the fix the
error analysis crashed (IndexError in _acid_base_constant_info) as soon as one
pKa was fixed, so the GUI showed no error table and the export had no error
sheets. Fixed pKa must now appear as fixed (SE = 0, Fixed = Yes, NaN
correlations) in the GUI rows and in the export frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmfit_core.acid_base import make_simple_acid_base_system
from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
from hmfit_core.run_acid_base import run_acid_base


def _fit(tmp_path, mask, init="4.2, 8.0"):
    system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=[4.0, 8.0], base_charge=-2)
    vol = np.linspace(0.0, 25.0, 40)
    exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                  titrant_concentration=1e-2, titrant_type="base")
    pH = simulate_pH_titration(exp, system) + np.random.default_rng(1).normal(0.0, 0.005, vol.size)
    path = tmp_path / "pot.csv"
    pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
    cfg = {"file_path": str(path), "data_type": "potentiometry", "pka_initial": init, "pka_fit_mask": mask,
           "analyte_concentration": 1e-2, "titrant_concentration": 1e-2, "initial_volume": 10.0,
           "titrant_type": "base", "base_charge": -2}
    res = run_acid_base(cfg, progress_cb=lambda _m: None)
    assert res["success"] is True
    return res


@pytest.mark.parametrize("method", ["analytic", "bootstrap_linear", "bootstrap_full_refit_audit"])
@pytest.mark.parametrize("mask, fixed_idx", [([True, False], 1), ([False, True], 0)])
def test_fixed_pka_reported_as_fixed(tmp_path, method, mask, fixed_idx):
    res = _fit(tmp_path, mask, init="4.2, 8.0" if fixed_idx == 1 else "4.0, 8.3")
    opts = {"method": method, "B": 40, "seed": 3}
    out = compute_errors_acid_base_from_context(res["errors_context"], opts)
    names = [row[0] for row in out["results_rows"]]
    assert names == ["pKa1", "pKa2"]
    fixed_row = out["results_rows"][fixed_idx]
    free_row = out["results_rows"][1 - fixed_idx]
    assert fixed_row[-1] == "Yes" and free_row[-1] == ""
    assert float(fixed_row[2]) == 0.0
    assert float(fixed_row[3]) == float(fixed_row[1]) == float(fixed_row[4])
    assert float(fixed_row[1]) == pytest.approx(8.0 if fixed_idx == 1 else 4.0, abs=1e-12)
    assert np.isfinite(float(free_row[2])) and float(free_row[2]) > 0.0
    corr = np.asarray(out["corr"], dtype=float)
    assert corr.shape == (2, 2)
    assert np.all(np.isnan(corr[fixed_idx, :])) and np.all(np.isnan(corr[:, fixed_idx]))
    assert corr[1 - fixed_idx, 1 - fixed_idx] == pytest.approx(1.0)
    pka_table = out["export_frames"]["pKa"]
    assert list(pka_table["Parameter"]) == ["pKa1", "pKa2"]
    assert float(pka_table.loc[fixed_idx, "SE"]) == 0.0


def test_fixed_pka_reaches_export_sheets(tmp_path):
    res = _fit(tmp_path, [True, False])
    params = res["export_data"]["error_parameters"]
    assert list(params["Parameter"]) == ["pKa1", "pKa2"]
    assert list(params["Fixed"]) == ["", "Yes"]
    assert float(params.loc[1, "SE"]) == 0.0


def test_all_free_potentiometry_unchanged(tmp_path):
    res = _fit(tmp_path, [True, True], init="4.2, 8.1")
    assert res["errors_context"]["fixed_constants"] == []
    out = compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    assert [row[-1] for row in out["results_rows"]] == ["", ""]
    assert np.all(np.isfinite(np.asarray(out["corr"], dtype=float)))
