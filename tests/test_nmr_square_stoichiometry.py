"""NMR analytic errors with a square stoichiometry map (as many signals as species).

Defect (literature benchmark, case r09a-r1-11-3n, docs/LITERATURE_BENCHMARK.md §6):
``compute_errors_nmr_varpro`` transposed the species x signals map whenever its
shape also matched signals x species, i.e. whenever both counts were equal. The
fit itself (projection) was right, but the residuals returned by the error
analysis, and hence the RMS reported by the NMR fit, the SE, the correlations
and the limiting shifts, used the wrong species for each signal.
"""
import numpy as np
import pandas as pd

from hmfit_core.api import run_nmr_fit
from hmfit_core.processors.nmr_processor import project_coeffs_block_onp_frac
from hmfit_core.solvers import NewtonRaphson
from hmfit_core.utils.errors import compute_errors_nmr_varpro

M = np.array([[1, 0], [0, 1], [1, 1]], float)          # R, G, RG
S = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], float)  # NH-a (R), NH-b (R), Me (G): 3 x 3
SHIFTS = np.array([[4.40, 4.20, 0.0], [0.0, 0.0, 3.10], [3.00, 4.10, 3.00]])  # species x signals


def _data(seed=5):
    TR = np.r_[0.0, np.geomspace(6e-4, 2e-2, 19)]
    TG = np.full_like(TR, 1.3e-4)
    CT = np.column_stack([TR, TG])
    C, _ = NewtonRaphson(CT, M.T, nas=[], model_sett="Free").concentraciones(np.array([2.4]))
    with np.errstate(invalid="ignore", divide="ignore"):
        D = C @ S
        y = (C[:, :, None] * S[None] * SHIFTS[None]).sum(axis=1) / D
    y += np.random.default_rng(seed).normal(0, 3e-3, y.shape)
    y[TR == 0, :2] = np.nan
    return CT, C, y


def test_error_analysis_uses_the_map_orientation_of_the_fit():
    CT, C, y = _data()
    res = NewtonRaphson(CT, M.T, nas=[], model_sett="Free")
    mask = np.isfinite(y)
    fit = project_coeffs_block_onp_frac(y, C, np.ones_like(y), mask, stoichiometry=S, rcond=1e-10, ridge=1e-8)
    rms_fit = np.sqrt(np.mean((y - fit)[mask] ** 2))
    err = compute_errors_nmr_varpro(np.array([2.4]), res, y, np.ones_like(y), M.T, [], stoichiometry=S,
                                    mask=mask)
    np.testing.assert_allclose(err["rms"], rms_fit, rtol=1e-9)
    # limiting shifts: zero-stoichiometry entries are not fitted, the rest are close to the truth
    assert np.all(err["coef"][S == 0] == 0.0)
    assert np.max(np.abs(err["coef"][S > 0] - SHIFTS[S > 0])) < 0.5


def test_reported_rms_equals_the_fit_residuals(tmp_path):
    CT, C, y = _data()
    path = tmp_path / "sq.xlsx"
    pd.DataFrame({"R": CT[:, 0], "G": CT[:, 1], "a": y[:, 0], "b": y[:, 1], "Me": y[:, 2]}).to_excel(
        path, index=False, sheet_name="S")
    cfg = {"file_path": str(path), "nmr_sheet": "S", "conc_sheet": "S", "column_names": ["R", "G"],
           "signal_names": ["a", "b", "Me"], "receptor_label": "R", "guest_label": "G",
           "modelo": M.tolist(), "non_abs_species": [], "algorithm": "Newton-Raphson",
           "model_settings": "Free", "optimizer": "powell", "initial_k": [2.4],
           "bounds": [[None, None]], "k_fixed": [True], "stoichiometry_map": S.tolist()}
    r = run_nmr_fit(cfg, progress_cb=lambda _m: None)
    ed = r["export_data"]
    resid = np.asarray(ed["Chemical_Shifts"], float) - np.asarray(ed["Calculated_Chemical_Shifts"], float)
    rms_fit = np.sqrt(np.nanmean(resid ** 2))
    rms_reported = dict((a, b) for a, b in ed["stats_table"])["RMS"]
    np.testing.assert_allclose(rms_reported, rms_fit, rtol=1e-9)
