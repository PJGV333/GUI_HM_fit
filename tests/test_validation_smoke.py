"""Smoke tests for deterministic and resumable validation campaigns."""
import csv
import json

import pytest

from validation.run_campaign import run_campaign


def test_analysis_reports_coverage_and_failures_with_wilson_interval():
    from scipy.stats import binomtest
    from validation.analyze import summarize

    rows = [
        {"success": "True", "truth": "3", "estimate": "3.1", "se": ".2", "ci_low": "2.7", "ci_high": "3.5"},
        {"success": "True", "truth": "3", "estimate": "3.4", "se": ".1", "ci_low": "3.2", "ci_high": "3.6"},
        {"success": "False"},
    ]
    out = summarize(rows)
    interval = binomtest(1, 2).proportion_ci(confidence_level=0.95, method="wilson")
    assert out["attempted"] == 3 and out["converged"] == 2 and out["covered"] == 1
    assert out["coverage95"] == 0.5 and out["failure_fraction"] == pytest.approx(1/3)
    assert out["coverage_wilson_low"] == pytest.approx(interval.low)
    assert out["coverage_wilson_high"] == pytest.approx(interval.high)


def test_campaign_refuses_to_checkpoint_after_source_changes(tmp_path, monkeypatch):
    import validation.run_campaign as campaign
    config = {"scenario": "uvvis_11_iid", "replicates": 1, "seed": 701,
              "points": 8, "noise": 0.0002, "method": "analytic", "workers": 1}
    fingerprints = iter(["before", "after"])
    monkeypatch.setattr(campaign, "_source_digest", lambda: next(fingerprints))
    with pytest.raises(ValueError, match="source"):
        campaign.run_campaign(config, tmp_path)
    assert not list((tmp_path / "checkpoints").glob("*.json"))


def test_campaign_detects_corrupt_checkpoint(tmp_path):
    config = {"scenario": "uvvis_11_iid", "replicates": 1, "seed": 701,
              "points": 8, "noise": 0.0002, "method": "analytic", "workers": 1}
    run_campaign(config, tmp_path)
    path = tmp_path / "checkpoints" / "00000000.json"
    record = json.loads(path.read_text())
    record["row"]["seed"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="corrupt"):
        run_campaign(config, tmp_path)


@pytest.mark.slow_validation_smoke
def test_serial_and_parallel_campaigns_preserve_seeded_results(tmp_path):
    config = {"scenario": "uvvis_11_iid", "replicates": 2, "seed": 701,
              "points": 8, "noise": 0.0002, "method": "analytic", "workers": 1}
    run_campaign(config, tmp_path / "serial")
    run_campaign({**config, "workers": 2}, tmp_path / "parallel")
    def read(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    serial, parallel = read(tmp_path / "serial/results.csv"), read(tmp_path / "parallel/results.csv")
    for row in serial + parallel:
        row.pop("seconds")
    assert serial == parallel


@pytest.mark.slow_validation_smoke
def test_real_uvvis_campaign_can_resume_without_duplicate_rows(tmp_path):
    config = {"scenario": "uvvis_11_iid", "replicates": 2, "seed": 701,
              "points": 8, "noise": 0.0002, "method": "analytic", "workers": 1}
    run_campaign(config, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    with (tmp_path / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["replicate"] for row in rows} == {"0", "1"}
    assert all(row["method"] == "analytic" for row in rows)
    assert all(row["success"] == "True" for row in rows)
    assert all(float(row["ci_low"]) < float(row["ci_high"]) for row in rows)
    assert manifest["config"] == config
    before = (tmp_path / "results.csv").read_bytes()
    manifest["completed"] = 0  # simulate interruption after checkpoint write
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    run_campaign(config, tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["completed"] == 2
    assert (tmp_path / "results.csv").read_bytes() == before
    with pytest.raises(ValueError, match="manifest"):
        run_campaign({**config, "seed": 702}, tmp_path)
