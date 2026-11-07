"""Literature benchmark, stage D: HM fit's mass-balance solver against the published
species distributions of R09b (SI S20-S23), without fitting or noise.

Pre-registered tolerance (docs/LITERATURE_BENCHMARK.md §4.6): 0.5 % relative on free
concentrations, 0.0005 absolute on fractions. Rows outside it must be explained by the
rounding of the printed totals (3 significant digits) used as inputs: one perturbation of
the totals within half a unit of their last digit must bring the whole row within
tolerance (§7). A row that cannot be explained that way would be a solver defect.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from validation import literature_speciation as ls

TABLES = ["r09b-spec-r", "r09b-spec-b", "r09b-spec-d-x", "r09b-spec-d-q"]


@pytest.mark.parametrize("sid", TABLES)
def test_table_columns_are_fractions_of_the_analysed_reagent(sid):
    """Interpretation check on the printed table itself: the x(.) columns sum to 1 and
    xbound(A) is the sum over species containing A and B."""
    case = ls.load(sid)
    hdr, rows = case["_header"], np.asarray(case["_rows"], dtype=float)
    xcols = [i for i, h in enumerate(hdr) if h.startswith("x(")]
    assert np.all(np.abs(rows[:, xcols].sum(axis=1) - 1.0) <= 0.00005 * len(xcols) + 1e-9)
    reagents, names, M, _ = ls.model(case, "si")
    ia, ib = reagents.index(case["analysed"]), reagents.index(case["bound"])
    bound = [i for i, h in enumerate(hdr) if h.startswith("x(")
             and M[names.index(re.fullmatch(r"x\((.+)\)", h).group(1)), ib] > 0]
    xb = rows[:, hdr.index(f"xbound({case['analysed']})")]
    assert np.all(np.abs(rows[:, bound].sum(axis=1) - xb) <= 0.00005 * len(bound) + 1e-9)
    # BC50 is the total of A, and the ancillary total equals it (electroneutrality)
    np.testing.assert_array_equal(rows[:, hdr.index("BC50")], rows[:, hdr.index(f"T({case['ancillary']})")])
    assert ia != ib


@pytest.mark.parametrize("sid", TABLES)
def test_solver_reproduces_speciation_within_printed_precision(sid):
    case = ls.load(sid)
    cmp = ls.compare(case, "si", "Newton-Raphson")
    explained = ls.rounding_explains(case, "si")
    assert explained["unexplained"] == [], explained["unexplained"][:3]
    assert explained["failing_rows"] == cmp["n_rows"] - cmp["n_pass"]
    # the misses are rounding-sized, never gross
    assert cmp["worst"]["conc_rel"] < 0.01 and cmp["worst"]["frac_abs"] < 0.002


@pytest.mark.parametrize("sid", TABLES)
def test_newton_raphson_and_levenberg_marquardt_agree(sid):
    case = ls.load(sid)
    a = ls.solve(case, "si", "Newton-Raphson")["rows"]
    b = ls.solve(case, "si", "Levenberg-Marquardt")["rows"]
    # Absolute floors: with T_B = 0 (first row) each solver leaves a different negligible
    # floor (1e-12 .. 1e-33) for the free concentration of the absent reagent.
    for ra, rb in zip(a, b):
        for k in ra:
            if not np.isfinite(ra[k]):
                continue
            floor = 1e-12 if k.startswith("[") else 1e-9
            assert abs(ra[k] - rb[k]) <= 1e-6 * abs(ra[k]) + floor, (k, ra[k], rb[k])


@pytest.mark.parametrize("sid", TABLES)
def test_half_of_the_bound_analyte_is_bound(sid):
    """BC50 definition: at every row 50 % of B is bound (within the rounding of the totals)."""
    case = ls.load(sid)
    for row in ls.solve(case, "si")["rows"]:
        if np.isfinite(row["_bound_fraction_B"]):
            assert abs(row["_bound_fraction_B"] - 0.5) < 0.01
