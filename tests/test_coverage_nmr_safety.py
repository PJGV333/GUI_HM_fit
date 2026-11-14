"""Legacy NMR coverage must not be reused as evidence for IRLS."""
import json

import pytest

from validation import coverage_campaign as cc


def _row(cell_id):
    return {"cell": cell_id, "rep": 0, "truth": [3.0], "fit_ok": True,
            "methods": {"analytic": {"est": [3.0], "se": [0.1], "lo": [2.8], "hi": [3.2],
                                      "fixed": [False], "n_fail": 0, "seconds": 1.0}}}


def test_historical_nmr_coverage_is_not_summarized(tmp_path):
    cell = next(c for c in cc.smoke_grid() if c["technique"] == "nmr")
    path = tmp_path / f"{cell['id']}.jsonl"
    original = (json.dumps(_row(cell["id"])) + "\n").encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match="(?i)nmr.*(invalid|unavailable|unsafe)"):
        cc.summarize(tmp_path, tmp_path / "summary.csv")
    assert path.read_bytes() == original
    assert not (tmp_path / "summary.csv").exists()


def test_resume_rejects_legacy_nmr_before_any_write(tmp_path):
    cell = next(c for c in cc.smoke_grid() if c["technique"] == "nmr")
    path = tmp_path / f"{cell['id']}.jsonl"
    path.write_text(json.dumps(_row(cell["id"])) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"started": "original"}\n', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="(?i)nmr.*(invalid|unavailable|unsafe)"):
        cc.run([cell], tmp_path, workers=1, progress=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_new_nmr_campaign_is_rejected_before_creating_output(tmp_path):
    cell = next(c for c in cc.smoke_grid() if c["technique"] == "nmr")
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="(?i)nmr.*(invalid|unavailable|unsafe)"):
        cc.run([cell], out, workers=1, progress=False)
    assert not out.exists()


def test_secondary_analysis_rejects_historical_nmr(tmp_path):
    from validation.coverage_secondary import per_replicate

    cell = next(c for c in cc.primary_grid() if c["technique"] == "nmr")
    (tmp_path / f"{cell['id']}.jsonl").write_text(json.dumps(_row(cell["id"])) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)nmr.*(invalid|unavailable|unsafe)"):
        per_replicate(tmp_path)


def test_primary_report_rejects_historical_nmr(tmp_path):
    from validation.coverage_report import load

    cell = next(c for c in cc.primary_grid() if c["technique"] == "nmr")
    (tmp_path / f"{cell['id']}.jsonl").write_text(json.dumps(_row(cell["id"])) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)nmr.*invalid"):
        load(tmp_path)


def test_unknown_historical_cell_without_technique_is_not_summarized(tmp_path):
    cell_id = "custom-binding-example"
    (tmp_path / f"{cell_id}.jsonl").write_text(json.dumps(_row(cell_id)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(unknown|missing).*technique"):
        cc.summarize(tmp_path)


def test_custom_nmr_rows_cannot_evade_safeguard(tmp_path):
    cell_id = "custom-binding-example"
    row = {**_row(cell_id), "technique": "nmr"}
    (tmp_path / f"{cell_id}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)nmr.*invalid"):
        cc.summarize(tmp_path)


def test_custom_cell_requires_technique_in_each_replicate(tmp_path):
    cell_id = "custom-binding-example"
    rows = [{**_row(cell_id), "technique": "uvvis"}, {**_row(cell_id), "rep": 1}]
    (tmp_path / f"{cell_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(unknown|missing).*technique"):
        cc.summarize(tmp_path)


def test_registered_non_nmr_rows_are_still_summarized(tmp_path):
    cell = next(c for c in cc.smoke_grid() if c["technique"] == "uvvis")
    (tmp_path / f"{cell['id']}.jsonl").write_text(json.dumps(_row(cell["id"])) + "\n", encoding="utf-8")
    rows = cc.summarize(tmp_path)
    assert len(rows) == 1
    assert rows[0]["cell"] == cell["id"] and rows[0]["n_used"] == 1


def test_real_non_nmr_campaign_can_run_summarize_and_resume(tmp_path):
    cell = next(c for c in cc.smoke_grid() if c["technique"] == "uvvis")
    cc.run([cell], tmp_path, n_rep=1, methods=["analytic"], workers=1, progress=False)
    path = tmp_path / f"{cell['id']}.jsonl"
    before = path.read_bytes()
    row = json.loads(before)
    assert row["technique"] == "uvvis" and row["fit_ok"]
    assert "error" not in row["methods"]["analytic"]
    assert cc.summarize(tmp_path)[0]["n_used"] == 1
    cc.run([cell], tmp_path, n_rep=1, methods=["analytic"], workers=1, progress=False)
    assert path.read_bytes() == before
