"""Etapa G (NMR 1:2): documented production gap.

process_nmr_data with a 4-species 1:2 stoichiometry (H, G, HG, HG2) runs
without raising but does NOT actually optimize log K: on zero-noise synthetic
data generated with the repo's own NewtonRaphson solver, fitted constants
return exactly their initial values (echo), or Powell drives K2 into its
bound with a small SE, and a mixed model/K-length call raises an internal
argument mismatch. Verified on current HEAD; xfail-pinned so the gap stays
visible without breaking the suite.
"""
import numpy as np
import pandas as pd
import pytest

from hmfit_core.processors.nmr_processor import process_nmr_data
from hmfit_core.solvers import NewtonRaphson

M = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]]


def _dataset(tmp_path):
    k_true = np.array([3.0, 2.3])
    n = 16
    H = np.full(n, 2e-3)
    G = np.linspace(0, 2e-2, n)
    sol = NewtonRaphson(np.column_stack([H, G]), np.asarray(M, float).T,
                        nas=[1], model_sett="Free")
    _, Co = sol.concentraciones(k_true)
    frac = Co / np.column_stack([H, G, H, H])
    d1 = np.array([7.0, 8.2, 7.6, 6.4])
    d2 = np.array([8.0, 7.4, 8.4, 8.9])
    path = tmp_path / "nmr12.xlsx"
    pd.DataFrame({"H": H, "G": G,
                  "s1": frac @ d1, "s2": frac @ d2}).to_excel(path, index=False,
                                                              sheet_name="S")
    return path


@pytest.mark.xfail(reason="NMR 1:2 free fit does not move log K off its initial "
                         "value on zero-noise data (production gap, Etapa G)",
                   strict=False)
def test_nmr_1_2_fixed_K2_recovers_K1(tmp_path):
    path = _dataset(tmp_path)
    res = process_nmr_data(str(path), "S", "S", ["H", "G"], ["s1", "s2"],
                           "H", "G", M, [2.5, 2.3],
                           [{"min": 1, "max": 6}, {"min": 0.5, "max": 5}],
                           "Newton-Raphson", "powell", "Free", [1],
                           k_fixed=[False, True], stoichiometry_map=M)
    k1 = {c["name"]: c["log10K"] for c in res["constants"]}["K1"]
    assert abs(k1 - 3.0) < 0.05          # currently echoes 2.5 (initial value)
