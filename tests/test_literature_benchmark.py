"""Literature benchmark (Roelens/Vacca 2009, HypNMR): external validation and regression.

Two references per case (docs/LITERATURE_BENCHMARK.md):

* the published HypNMR results, with the criteria pre-registered in §4 (classification
  "reproduce" = levels 1-3); cases that do not reproduce or are not supported are
  ``xfail(strict=True)`` with their cause and report section, never skipped;
* HM fit's own baseline (``baseline.json``), with a narrow tolerance that catches any
  change while allowing a new optimiser to reach the same minimum.

Every fit goes through ``hmfit_core.api.run_nmr_fit`` (the NMR tab's worker target)
via ``validation.literature_benchmark``.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation import literature_benchmark as lb

FIT_CASES = lb.case_ids("fit")

_W = "docs/LITERATURE_BENCHMARK.md §6.1"
EXPECTED_FAIL = {
    "r09a-r1-5sp": f"parcial: L2 fails for R2 (dlogb -0.016). HM fit's NMR fit weights each signal by "
                   f"1/sigma_j^2 (IRLS) while HypNMR uses unit weights; with unit weights HM fit's model "
                   f"reproduces the published values ({_W})",
    "r09b-r-qx": f"parcial: same data and cause as r09a-r1-5sp ({_W})",
    "r09a-r1-11-3n": f"parcial: IRLS weights make the fit follow Me (weight x1000 over NH): dlogb +0.19; "
                     f"unit-weight check reproduces ({_W})",
    "r09a-p2-058": f"parcial: L2 (PX, P2QX) and L3 (SE -22..-28 %) fail because of the IRLS per-signal "
                   f"weights; unit-weight check reproduces ({_W})",
    "r09a-p2-106": f"no reproduce: RMS +10 % (unweighted) because of the IRLS per-signal weights; "
                   f"unit-weight check reproduces ({_W})",
    "r09a-r1-3sp": f"no reproduce: RMS +28 % (unweighted) because of the IRLS per-signal weights in a "
                   f"deliberately inadequate model; unit-weight check reproduces ({_W})",
    "r09a-qx-dil": ("no reproduce: the published RMS is not reachable with the printed data and the 1:1 "
                    "model under unit weights (independent calculation: +7.2 %); the original fit used "
                    "data or weights other than the printed ones (docs/LITERATURE_BENCHMARK.md §6.2)"),
    "r09b-d-qx": ("no soportado: six DQ limiting shifts are fixed in the published fit and HM fit cannot "
                  "fix limiting shifts (docs/LITERATURE_BENCHMARK.md §2, §6.3)"),
}
EXPECTED_L0_FAIL = {
    "r09a-qx-dil": EXPECTED_FAIL["r09a-qx-dil"],
}

_cache: dict[str, dict] = {}


def _fit(cid):
    if cid not in _cache:
        case = lb.load_case(cid)
        _cache[cid] = {"case": case, "fit": lb.fit_case(case)}
    return _cache[cid]


def _params(ids, expected):
    return [pytest.param(c, marks=pytest.mark.xfail(strict=True, reason=expected[c])) if c in expected
            else c for c in ids]


def test_case_inventory_is_complete():
    assert set(FIT_CASES) == {
        "r09a-r1-11-me", "r09a-r1-11-3n", "r09a-r1-3sp", "r09a-r1-5sp", "r09a-qx-dil", "r09a-p2-058",
        "r09a-p2-106", "r09b-r-qx", "r09b-b-qx", "r09b-d-qx"}
    assert set(EXPECTED_FAIL) <= set(FIT_CASES)
    for cid in FIT_CASES:
        src = lb.load_case(cid)["source"]
        assert src["doi"] and src["si_pages"] and src["notice"]


@pytest.mark.parametrize("cid", _params(FIT_CASES, EXPECTED_FAIL))
def test_literature_reproduces_published_fit(cid):
    case = lb.load_case(cid)
    ok, why = lb.supported(case)
    assert ok, why
    rec = _fit(cid)
    assert rec["fit"]["success"], rec["fit"].get("error")
    cmp = lb.compare(rec["case"], rec["fit"])
    fails = [f"{lvl}: {x['name']} {x.get('delta', x.get('rel')):+.4g} (tol {x['tol']:.4g})"
             for lvl in ("L2", "L3") for x in cmp["levels"][lvl]["items"] if not x["pass"]]
    assert cmp["levels"]["L1"]["pass"], f"L1: RMS {cmp['levels']['L1']['rel']:+.2%}"
    assert cmp["classification"] == "reproduce", "; ".join(fails)


@pytest.mark.parametrize("cid", _params([c for c in FIT_CASES if lb.supported(lb.load_case(c))[0]],
                                        EXPECTED_L0_FAIL))
def test_literature_forward_model_at_published_constants(cid):
    """Level 0: constants fixed at the published values, HM fit solves the limiting
    shifts; its RMS must match the published one (transcription and model check)."""
    case = lb.load_case(cid)
    fw = lb.forward_check(case)
    assert fw["success"], fw.get("error")
    tol = (case.get("_tolerances") or {}).get("L1_rms_rel", lb.CRITERIA["L1_rms_rel"])
    assert abs(fw["rel"]) <= tol, f"RMS at the published constants differs by {fw['rel']:+.2%}"


@pytest.mark.parametrize("cid", [c for c in FIT_CASES if lb.supported(lb.load_case(c))[0]])
def test_literature_baseline_regression(cid):
    """Internal regression: |dlogb| <= 1e-4 and SE relative difference <= 1e-3 against
    baseline.json (updated only by an explicit 'Update literature baseline' commit)."""
    rec = _fit(cid)
    bl = rec["case"]["_baseline"]
    assert bl is not None, "missing baseline.json"
    fit = rec["fit"]
    assert fit["success"]
    for name, val, se, fx in zip(fit["names"], fit["log_beta"], fit["se"], fit["fixed"]):
        assert abs(val - bl["log_beta"][name]) <= 1e-4, f"{name}: log beta {val} vs baseline {bl['log_beta'][name]}"
        if not fx:
            ref_se = bl["se"][name]
            assert abs(se / ref_se - 1.0) <= 1e-3, f"{name}: SE {se} vs baseline {ref_se}"


def test_unsupported_case_is_detected_not_silently_fitted():
    ok, why = lb.supported(lb.load_case("r09b-d-qx"))
    assert not ok and "fixed limiting shifts" in why


def test_degrees_of_freedom_identity_holds_for_every_case():
    """The HypNMR identity (sigma/RMS)^2 = m/(m-p) confirms the number of refined parameters."""
    for cid in FIT_CASES:
        dof = lb.load_case(cid)["dof_identity"]
        assert abs(dof["rel_diff"]) < 1e-6, cid


def test_rounding_perturbation_respects_printed_precision():
    case = lb.load_case("r09b-b-qx")
    c0, s0 = lb.numeric_table(case)
    c1, s1 = lb.numeric_table(case, {"rng": np.random.default_rng(1)})
    assert np.all(np.abs(c1[:, 1:] - c0[:, 1:]) <= 0.5e-7 + 1e-18)      # 1.850E-04 -> half unit 5e-8
    assert np.all(c1[:, 1] == c1[:, 2])                                   # one draw for the salt
    assert np.all(c1[c0[:, 0] == 0, 0] == 0)                              # zeros stay zero
    m = np.isfinite(s0)
    assert np.all(np.abs(s1[m] - s0[m]) <= 0.5e-4 + 1e-12)
