import numpy as np
import pandas as pd

from hmfit_core.processors import nmr_processor as nmr
from hmfit_core.solvers import NewtonRaphson


def _fit(tmp_path, monkeypatch, bounds, *, optimizer='explicit_lm', fixed=None, parallel=False):
    monkeypatch.setattr(nmr, 'generate_figure_base64', lambda *args, **kwargs: '')
    monkeypatch.setattr(nmr, 'generate_figure2_base64', lambda *args, **kwargs: '')
    n = 12
    H = np.full(n, 1e-3)
    G = np.linspace(0, 5e-3, n)
    model = [[1, 0], [0, 1], [1, 1]]
    solver = NewtonRaphson(np.column_stack([H, G]), np.array(model).T, nas=[1], model_sett='Free')
    _, Co = solver.concentraciones(np.array([3.]))
    fraction = Co[:, [0, 2]] / H[:, None]
    rng = np.random.default_rng(4)
    s1 = fraction @ [7., 7.6] + rng.normal(0, .002, n)
    s2 = fraction @ [8., 7.5] + rng.normal(0, .002, n)
    path = tmp_path / 'n.xlsx'
    pd.DataFrame({'H': H, 'G': G, 's1': s1, 's2': s2}).to_excel(path, index=False, sheet_name='S')
    return nmr.process_nmr_data(str(path), 'S', 'S', ['H', 'G'], ['s1', 's2'], 'H', 'G',
                                model, [2.8], bounds, 'Newton-Raphson', optimizer, 'Free', [1],
                                k_fixed=fixed, multi_start_runs=2 if parallel else 1,
                                multi_start_parallel=parallel, multi_start_seeds=[1, 2] if parallel else None)


def test_nmr_lm_with_irls_matches_powell(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 5.]])
    ref = _fit(tmp_path, monkeypatch, [[1., 5.]], optimizer='powell')
    assert out['success'], out
    assert out['optimizer_result']['success'], out['optimizer_result']
    assert abs(out['export_data']['k'][0] - ref['export_data']['k'][0]) < 1e-3
    assert ref['optimizer_result']['success']
    ed = out['export_data']
    diff = np.asarray(ed['Chemical_Shifts']) - np.asarray(ed['Calculated_Chemical_Shifts'])
    weights = np.asarray(ed['weights_used_for_fit'])
    np.testing.assert_allclose(ed['weighted_ssq_fit'], np.sum(diff**2 * weights[None, :]))
    assert ed['irls_cycles'] > 0
    assert 'LM: success=' in out['results_text']


def test_nmr_lm_marks_active_bound(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 2.9]])
    assert out['optimizer_result']['active_bounds'] == [True]
    assert out['export_data']['SE_log10K'][0] is None
    assert 'límite' in out['results_text']


def test_nmr_lm_reports_nonconvergence(tmp_path, monkeypatch):
    from hmfit_core.utils import explicit_lm as module
    original = module.explicit_lm
    def short_run(*args, **kwargs):
        return original(*args, maxiter=1, **kwargs)
    monkeypatch.setattr(module, 'explicit_lm', short_run)
    out = _fit(tmp_path, monkeypatch, [[1., 5.]])
    assert not out['optimizer_result']['success']
    assert out['optimizer_result']['message'] == 'maxiter'
    assert '>>> FIT NOT RELIABLE' in out['results_text']


def test_nmr_lm_fixed_parameter_unchanged(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], fixed=[True])
    assert out['success']
    assert out['export_data']['SE_log10K'][0] == 0


def test_nmr_lm_parallel_multistart(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], parallel=True)
    assert out['success'], out
    assert out['optimizer_result']['success'], out['optimizer_result']
