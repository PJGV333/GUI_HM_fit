"""Coverage campaign (Phase 2, stage 2.2): CI smoke subset and slow full tiers.

The smoke cells run every technique through the GUI entry points with tiny N and B;
they check plumbing, determinism and resumability, not calibration.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from validation import coverage_campaign as cc  # noqa: E402


@pytest.mark.campaign_smoke
def test_smoke_campaign_runs_every_technique_and_resumes(tmp_path):
    grid = cc.smoke_grid()
    cc.run(grid, tmp_path, workers=1, progress=False)
    files = sorted(p for p in tmp_path.glob("*.jsonl") if p.name != "manifest.jsonl")
    assert len(files) == len(grid)
    techniques = set()
    for cell, path in zip(sorted(grid, key=lambda c: c["id"]), files):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["rep"] for r in sorted(rows, key=lambda r: r["rep"])] == list(range(cell["N"]))
        for row in rows:
            assert row["fit_ok"], row.get("fit_error")
            assert set(row["methods"]) == set(cell["methods"])
            for rec in row["methods"].values():
                assert "error" not in rec, rec.get("error")
                assert len(rec["se"]) == len(row["truth"])
        techniques.add(cell["technique"])
    assert techniques == {"uvvis", "nmr", "ab_nmr", "ab_spec", "pot", "kin"}

    before = {p.name: p.read_text() for p in files}
    cc.run(grid, tmp_path, workers=1, progress=False)  # resume: nothing to recompute
    assert {p.name: p.read_text() for p in files} == before

    summary = cc.summarize(tmp_path)
    assert summary and all(r["n_rep"] == 3 for r in summary)
    finite = [r for r in summary if not r.get("fixed") and r.get("n_used")]
    assert finite and all(np.isfinite(r["se_mean"]) for r in finite)


@pytest.mark.campaign_smoke
def test_replicates_are_deterministic():
    cell = cc.smoke_grid()[0]
    a = cc.run_replicate(cell, 1)
    b = cc.run_replicate(cell, 1)
    for m in cell["methods"]:
        assert a["methods"][m]["est"] == b["methods"][m]["est"]
        assert a["methods"][m]["se"] == b["methods"][m]["se"]


def test_wilson_and_classification_thresholds():
    lo, hi = cc.wilson(475, 500)
    assert lo == pytest.approx(0.9271, abs=1e-3) and hi == pytest.approx(0.9662, abs=1e-3)
    assert cc.classify(0.95, 1.0) == "calibrated"
    assert cc.classify(0.92, 1.0) == "anticonservative"
    assert cc.classify(0.95, 0.80) == "anticonservative"
    assert cc.classify(0.99, 1.0) == "conservative"
    assert cc.classify(0.95, 1.2) == "conservative"


def test_indeterminate_when_wilson_interval_contains_a_threshold():
    """Amendment 1: a cell is "indeterminate" when its Wilson interval contains 0.93 or 0.97."""
    lo, hi = cc.wilson(190, 200)  # 0.95 at N = 200: [0.910, 0.973] contains both thresholds
    assert cc.classify_ci(lo, hi, "calibrated") == "indeterminate"
    lo, hi = cc.wilson(475, 500)  # 0.95 at N = 500: [0.927, 0.966] still contains 0.93
    assert cc.classify_ci(lo, hi, "calibrated") == "indeterminate"
    lo, hi = cc.wilson(170, 200)  # 0.85: [0.794, 0.893], clearly below
    assert cc.classify_ci(lo, hi, "anticonservative") == "anticonservative"
    lo, hi = cc.wilson(499, 500)
    assert cc.classify_ci(lo, hi, "conservative") == "conservative"
    assert cc.classify_ci(float("nan"), float("nan"), "undefined") == "undefined"


def test_inverted_acid_base_diagnostic_cell_runs():
    cell = next(c for c in cc.diagnostic_grid() if c["id"] == "diag-ab_nmr-inverted-25-iid")
    assert cc.AB_PKA["inverted"] == [4.0, 8.0] and cell["tier"] == 9
    row = cc.run_replicate({**cell, "B": 20, "B_refit": 4}, 0, ["analytic", "lin_row"])
    assert row["truth"] == [4.0, 8.0] and "analytic" in row["methods"]


@pytest.mark.slow
@pytest.mark.parametrize("tier", [1, 2, 3, 4, 9])
def test_full_campaign_tier(tier, tmp_path):
    """Full pre-registered tier (hours). Run with: pytest -m slow -k "tier and 1"."""
    out = os.environ.get("HMFIT_CAMPAIGN_OUT", str(tmp_path))
    grid = [c for c in cc.primary_grid() + cc.diagnostic_grid() if c["tier"] == tier]
    cc.run(grid, cc.Path(out), workers=int(os.environ.get("HMFIT_CAMPAIGN_WORKERS", os.cpu_count() or 1)))
    assert cc.summarize(cc.Path(out))
