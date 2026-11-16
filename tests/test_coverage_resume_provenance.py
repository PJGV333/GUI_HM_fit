"""Coverage campaigns must not mix incompatible provenance on resume."""
import json

import pytest

from validation import coverage_campaign as cc


@pytest.mark.parametrize('field', ['commit', 'dirty', 'source_diff_sha256', 'python', 'numpy', 'scipy',
                                   'numeric_build_sha256', 'dependencies', 'threads', 'base_seed'])
def test_changed_provenance_refuses_resume_without_writes(tmp_path, monkeypatch, field):
    cell = next(c for c in cc.smoke_grid() if c['technique'] == 'uvvis')
    original = cc._provenance()
    monkeypatch.setattr(cc, '_provenance', lambda: dict(original))
    cc.run([cell], tmp_path, n_rep=1, methods=['analytic'], workers=1, progress=False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    changed = dict(original)
    changed[field] = not changed[field] if field == 'dirty' else 'changed'
    monkeypatch.setattr(cc, '_provenance', lambda: changed)
    with pytest.raises(ValueError, match='(?i)(provenance|incompatible)'):
        cc.run([cell], tmp_path, n_rep=2, methods=['analytic'], workers=1, progress=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('change', ['methods', 'configuration'])
def test_changed_cell_contract_refuses_resume(tmp_path, change):
    cell = next(c for c in cc.smoke_grid() if c['technique'] == 'uvvis')
    cc.run([cell], tmp_path, n_rep=1, methods=['analytic'], workers=1, progress=False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    altered = {**cell, 'noise': 'hetero'} if change == 'configuration' else cell
    methods = ['lin_row'] if change == 'methods' else ['analytic']
    with pytest.raises(ValueError, match='(?i)(provenance|incompatible)'):
        cc.run([altered], tmp_path, n_rep=2, methods=methods, workers=1, progress=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_source_change_during_campaign_does_not_write_replicate(tmp_path, monkeypatch):
    cell = next(c for c in cc.smoke_grid() if c['technique'] == 'uvvis')
    original = cc._provenance()
    state = {'changed': False}
    monkeypatch.setattr(cc, '_provenance', lambda: {**original, 'commit': 'changed' if state['changed'] else original['commit']})
    worker = cc._worker
    def change_after_work(args):
        result = worker(args)
        state['changed'] = True
        return result
    monkeypatch.setattr(cc, '_worker', change_after_work)
    with pytest.raises(ValueError, match='(?i)provenance'):
        cc.run([cell], tmp_path, n_rep=1, methods=['analytic'], workers=1, progress=False)
    assert not (tmp_path / (cell['id'] + '.jsonl')).exists()


def test_historical_manifest_cannot_be_resumed(tmp_path):
    cell = next(c for c in cc.smoke_grid() if c['technique'] == 'uvvis')
    (tmp_path / 'manifest.jsonl').write_text(json.dumps({'commit': 'old', 'cells': [cell['id']]}) + '\n')
    path = tmp_path / (cell['id'] + '.jsonl')
    path.write_text(json.dumps({'cell': cell['id'], 'rep': 0}) + '\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='(?i)(provenance|incompatible)'):
        cc.run([cell], tmp_path, n_rep=2, workers=1, progress=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
