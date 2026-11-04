"""Etapa G (NMR 1:2): convention and identifiability pins (was: xfail gap).

Investigation superseded the original "production gap" claim: with data
generated under the projector's own convention (stoichiometry_map counts
equivalent nuclei, so HG2 carries weight 2 in the G-parent signal, and the
signal-parent species stay OUT of non_absorbent_species), process_nmr_data
optimizes a 1:2 Free fit correctly on noisy two-nucleus data.

Pinned here:
1. K1 and K2 recovery in an identifiable regime (2*K2*C_H*C_G not << 1).
2. The identifiability self-diagnostic: when HG2 occupancy is negligible the
   K2 direction is flat and the analytic SE blows up (no silent confident fit).
3. The 'Free' model-set validation error when fewer constants than complexes
   are supplied (loud ValueError, not a silent mismatch).
"""
import numpy as np
import pandas as pd
import pytest

from hmfit_core.processors.nmr_processor import process_nmr_data
from hmfit_core.solvers import NewtonRaphson

M = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]]
D1 = np.array([7.0, 8.2, 7.6, 6.4])
D2 = np.array([8.0, 7.4, 8.4, 8.9])


def _dataset(tmp_path, k_true, seed):
    """Generator consistent with project_coeffs_block_onp_frac: signal =
    (Co @ (stoich_column * deltas)) / total_parent_concentration."""
    n = 24
    H = np.full(n, 2e-3)
    G = np.linspace(2e-4, 3e-2, n)
    _, Co = NewtonRaphson(np.column_stack([H, G]), np.asarray(M, float).T,
                          nas=[], model_sett="Free").concentraciones(np.asarray(k_true, float))
    rng = np.random.default_rng(seed)
    s1 = (Co @ (np.asarray(M)[:, 0] * D1)) / H + rng.normal(0, 2e-3, n)
    s2 = (Co @ (np.asarray(M)[:, 1] * D2)) / G + rng.normal(0, 2e-3, n)
    path = tmp_path / "nmr12.xlsx"
    pd.DataFrame({"H": H, "G": G, "s1": s1, "s2": s2}).to_excel(path, index=False, sheet_name="S")
    return path


def _fit(path, k0, bounds, k_fixed):
    return process_nmr_data(str(path), "S", "S", ["H", "G"], ["s1", "s2"], "H", "G", M,
                            list(k0), [{"min": b0, "max": b1} for b0, b1 in bounds],
                            "Newton-Raphson", "powell", "Free", [],
                            k_fixed=k_fixed, stoichiometry_map=M)


def test_nmr_1_2_free_recovers_both_constants(tmp_path):
    path = _dataset(tmp_path, [3.0, 4.5], seed=11)
    res = _fit(path, [2.5, 3.0], [(1, 6), (2, 7)], [False, False])
    ks = {c["name"]: (c["log10K"], c["SE_log10K"]) for c in res["constants"]}
    assert abs(ks["K1"][0] - 3.0) < 0.15 and ks["K1"][1] < 0.1
    assert abs(ks["K2"][0] - 4.5) < 0.15 and ks["K2"][1] < 0.1


def test_nmr_1_2_unidentifiable_K2_reports_huge_SE(tmp_path):
    # Same system, but K2 too weak for the concentration regime: 2*K2*C_H*C_G << 1.
    path = _dataset(tmp_path, [3.0, 2.3], seed=11)
    res = _fit(path, [2.5, 1.5], [(1, 6), (0.5, 5)], [False, False])
    ks = {c["name"]: (c["log10K"], c["SE_log10K"]) for c in res["constants"]}
    assert abs(ks["K1"][0] - 3.0) < 0.1          # K1 still identified
    assert ks["K2"][1] > 1.0                      # flat K2 direction, loud SE


def test_nmr_free_model_rejects_short_K_vector(tmp_path):
    path = _dataset(tmp_path, [3.0, 4.5], seed=3)
    with pytest.raises((ValueError, RuntimeError)):
        process_nmr_data(str(path), "S", "S", ["H", "G"], ["s1", "s2"], "H", "G", M,
                         [2.5], [{"min": 1, "max": 6}],
                         "Newton-Raphson", "powell", "Free", [],
                         k_fixed=[False], stoichiometry_map=M)


def test_nmr_unknown_optimizer_fails_honestly(tmp_path):
    # least_squares is not a scipy.optimize.minimize method: the run must be
    # reported as failed, NOT echo the initial constants with success=True.
    path = _dataset(tmp_path, [3.0, 4.5], seed=3)
    res = process_nmr_data(str(path), "S", "S", ["H", "G"], ["s1", "s2"], "H", "G", M,
                           [2.5, 3.0], [{"min": 1, "max": 6}, {"min": 2, "max": 7}],
                           "Newton-Raphson", "least_squares", "Free", [],
                           k_fixed=[False, False], stoichiometry_map=M)
    assert res["success"] is False
    assert "optimization failed" in (res.get("error") or "")
