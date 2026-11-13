import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import phase1_cases as pc
from hmfit_core import run_spectroscopy


def _fit(tmp_path, bounds, optimizer='explicit_lm'):
    path = tmp_path / 's.xlsx'
    model = [[1, 0], [0, 1], [1, 1]]
    pc._spectro_dataset(path, [3.0], model)
    cfg = pc._spectro_cfg(path, model, [2.8], bounds)
    cfg.update(optimizer=optimizer, render_graphs=False)
    return pc._quiet(run_spectroscopy, cfg)


def test_spectral_explicit_lm_reproduces_synthetic_fit(tmp_path):
    out = _fit(tmp_path, [[1., 5.]])
    powell = _fit(tmp_path, [[1., 5.]], optimizer='powell')
    assert out['optimizer_result']['success'], out['optimizer_result']
    assert abs(out['export_data']['k'][0] - powell['export_data']['k'][0]) < 1e-4
    assert out['optimizer_result']['iterations'] > 0
    assert np.isfinite(out['optimizer_result']['scaled_gradient'])
    assert 'LM: success=' in out['results_text']


def test_spectral_explicit_lm_preserves_fixed_parameter_convention(tmp_path):
    out = _fit(tmp_path, [[3., 3.]])
    assert out['optimizer_result']['success']
    assert out['export_data']['SE_log10K'][0] == 0
    assert not any('límite' in note for note in out['export_data']['uncertainty_notes'])


def test_spectral_explicit_lm_reports_nonconvergence(tmp_path, monkeypatch):
    from hmfit_core.utils import explicit_lm as module
    original = module.explicit_lm
    def short_run(*args, **kwargs):
        return original(*args, maxiter=1, **kwargs)
    monkeypatch.setattr(module, 'explicit_lm', short_run)
    out = _fit(tmp_path, [[1., 5.]])
    assert not out['optimizer_result']['success']
    assert out['optimizer_result']['message'] == 'maxiter'
    assert '>>> FIT NOT RELIABLE' in out['results_text']


def test_spectral_explicit_lm_marks_bound_uncertainty(tmp_path):
    out = _fit(tmp_path, [[1., 2.9]])
    assert out['optimizer_result']['success'], out['optimizer_result']
    assert out['optimizer_result']['active_bounds'] == [True]
    # JSON sanitization represents undefined NaN as null.
    assert out['export_data']['SE_log10K'][0] is None
    assert 'límite' in out['results_text']
    assert any('límite' in note for note in out['export_data']['uncertainty_notes'])
