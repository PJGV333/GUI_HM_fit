"""Literature benchmark, stage E: hmfit_core.bc50 against published BC50 values
(criteria pre-registered in docs/LITERATURE_BENCHMARK.md §4.7; results in §7.1)."""
from __future__ import annotations

import pytest

from validation import literature_bc50 as lbc

_TAB = {}


def _tables():
    if not _TAB:
        _TAB.update(lbc.tables("si"))
    return _TAB


def test_r09b_intrinsic_bc50_values():
    for name, rec in lbc.r09b("si").items():
        assert rec["pass"], f"{name}: {rec['hm']:.4g} vs {rec['printed']}"


def test_r09a_intrinsic_bc50_and_electroneutral_ratio():
    r = lbc.r09a("si")
    for key in ("BC50_0_receptor1_TMA1", "BC50_0_receptor2_TMA1", "ratio_neutral_max", "ratio_neutral_x0"):
        assert r[key]["pass"], f"{key}: {r[key]['hm']:.4g} vs {r[key]['printed']}"
    assert r["ratio_neutral_argmax_x"] == 0.5          # "maximum value of 16.5 at 50 % complexation"
    assert abs(r["ratio_TMA1_max"]["hm"] - r["ratio_TMA1_min"]["hm"]) < 1e-9   # constant along x


@pytest.mark.xfail(strict=True, reason=(
    "BC50(1)/BC50(2) at [TMA] = 1 M is constant along x as printed, but equals 15.52 with the "
    "results-page constants (15.70 with the article ones), not 15.4; 15.4 is the ratio of the "
    "rounded BC50^0 values 1.03e-2/6.7e-4 = 15.37 (docs/LITERATURE_BENCHMARK.md §7.1)"))
def test_r09a_ratio_at_fixed_cation_is_15_4():
    r = lbc.r09a("si")
    assert r["ratio_TMA1_min"]["pass"] and r["ratio_TMA1_max"]["pass"]


@pytest.mark.parametrize("sid", ["r09b-spec-r", "r09b-spec-b", "r09b-spec-d-x", "r09b-spec-d-q"])
def test_bc50_tables_within_one_printed_unit(sid):
    t = _tables()[sid]
    assert t["within_one_ulp"] == t["cells"], t["worst"][:3]


_HALF = "docs/LITERATURE_BENCHMARK.md §7.1: 2.7-4.6 % of the cells lie 0.50-0.55 printed units away (98 % required)"
@pytest.mark.parametrize("sid", [
    pytest.param("r09b-spec-r", marks=pytest.mark.xfail(strict=True, reason=_HALF)),
    pytest.param("r09b-spec-b", marks=pytest.mark.xfail(strict=True, reason=_HALF)),
    pytest.param("r09b-spec-d-x", marks=pytest.mark.xfail(strict=True, reason=_HALF)),
    "r09b-spec-d-q",
])
def test_bc50_tables_reproduce_preregistered(sid):
    """>= 98 % of the cells within half a printed unit and all within one."""
    assert _tables()[sid]["reproduce"]


def test_multinuclear_bound_analyte_is_warned():
    """R2X is binuclear in the bound analyte R (X vs R): BC50 must carry the note-18 warning."""
    assert any("note 18" in w for w in _tables()["r09b-spec-r"]["warnings"])
    assert _tables()["r09b-spec-b"]["warnings"] == []
