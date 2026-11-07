"""NMR export: the "Coefficients" sheet must hold the limiting shifts of the fitted model.

Defect (found while building the literature benchmark, docs/LITERATURE_BENCHMARK.md §6):
the exporter recomputed the limiting shifts with the alias-based parent column and
without the per-signal stoichiometry map used by the fit, so in a three-reagent
model (receptor, cation, anion) the exported shifts did not correspond to the fit
(e.g. Me of QX exported as -0.10 ppm instead of 3.22 ppm), although the constants
were right.
"""
import numpy as np
import pandas as pd

from hmfit_core.api import run_nmr_fit
from hmfit_core.exports import write_results_xlsx
from hmfit_core.solvers import NewtonRaphson

M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], float)
D_NH = np.array([4.3, 0, 0, 5.0, 5.5, 0])     # nucleus of R
D_ME = np.array([0, 3.0, 0, 2.98, 0, 3.22])   # nucleus of Q


def _case(tmp_path):
    rows = [(tr, tq, tq) for tq in (1.3e-4, 1.06e-3) for tr in np.r_[0, np.geomspace(6e-4, 2e-2, 15)]]
    CT = np.array(rows)
    _, Co = NewtonRaphson(CT, M.T, nas=[], model_sett="Free").concentraciones(np.array([6.0, 3.0, 4.0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        nh = (Co @ (M[:, 0] * D_NH)) / CT[:, 0]
    me = (Co @ (M[:, 1] * D_ME)) / CT[:, 1]
    nh[CT[:, 0] == 0] = np.nan
    path = tmp_path / "three.xlsx"
    pd.DataFrame({"R": CT[:, 0], "Q": CT[:, 1], "X": CT[:, 2], "NH": nh, "Me": me}).to_excel(
        path, index=False, sheet_name="S")
    return {"file_path": str(path), "nmr_sheet": "S", "conc_sheet": "S", "column_names": ["R", "Q", "X"],
            "signal_names": ["NH", "Me"], "receptor_label": "R", "guest_label": "X",
            "modelo": M.tolist(), "non_abs_species": [], "algorithm": "Newton-Raphson",
            "model_settings": "Free", "optimizer": "powell", "initial_k": [5.8, 3.2, 3.8],
            "bounds": [[None, None]] * 3, "k_fixed": [False] * 3,
            "stoichiometry_map": np.column_stack([M[:, 0], M[:, 1]]).tolist()}


def test_exported_limiting_shifts_follow_the_stoichiometry_map(tmp_path):
    res = run_nmr_fit(_case(tmp_path), progress_cb=lambda _m: None)
    assert res["success"]
    np.testing.assert_allclose([c["log10K"] for c in res["constants"]], [6.0, 3.0, 4.0], atol=1e-4)
    out = tmp_path / "export.xlsx"
    write_results_xlsx(out, constants=res["constants"], statistics={}, results_text=res["results_text"],
                       export_data=res["export_data"])
    co = pd.read_excel(out, sheet_name="Coefficients", header=None).iloc[-6:, 1:3].to_numpy(dtype=float)
    S = np.column_stack([M[:, 0], M[:, 1]])
    true = np.column_stack([D_NH, D_ME])
    np.testing.assert_allclose(co[S > 0], true[S > 0], atol=1e-4)
    # species without the nucleus' reagent have no limiting shift (not a fitted 0 or junk)
    assert np.isnan(co[S == 0]).all()
