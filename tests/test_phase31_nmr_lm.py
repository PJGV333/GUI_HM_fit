import numpy as np
import pandas as pd

from hmfit_core.processors import nmr_processor as nmr
from hmfit_core.solvers import NewtonRaphson


def _fit(tmp_path, monkeypatch, bounds, *, optimizer='powell', fixed=None, parallel=False,
         weights_mode='irls'):
    monkeypatch.setattr(nmr, 'generate_figure_base64', lambda *args, **kwargs: '')
    monkeypatch.setattr(nmr, 'generate_figure2_base64', lambda *args, **kwargs: '')
    n = 12
    H = np.full(n, 1e-3)
    G = np.linspace(0, 5e-3, n)
    model = [[1, 0], [0, 1], [1, 1]]
    solver = NewtonRaphson(np.column_stack([H, G]), np.array(model).T, nas=[1], model_sett='Free')
    _, Co = solver.concentraciones(np.array([3.]))
    fraction = np.asarray(Co)[:, [0, 2]] / H[:, None]
    rng = np.random.default_rng(4)
    s1 = fraction @ [7., 7.6] + rng.normal(0, .002, n)
    s2 = fraction @ [8., 7.5] + rng.normal(0, .002, n)
    path = tmp_path / 'n.xlsx'
    pd.DataFrame({'H': H, 'G': G, 's1': s1, 's2': s2}).to_excel(path, index=False, sheet_name='S')
    return nmr.process_nmr_data(str(path), 'S', 'S', ['H', 'G'], ['s1', 's2'], 'H', 'G',
                                model, [2.8], bounds, 'Newton-Raphson', optimizer, 'Free', [1],
                                k_fixed=fixed, multi_start_runs=2 if parallel else 1,
                                multi_start_parallel=parallel, multi_start_seeds=[1, 2] if parallel else None,
                                weights_mode=weights_mode)




def test_nmr_irls_reports_and_uses_weights_of_the_last_fit(tmp_path, monkeypatch):
    original = nmr.compute_errors_nmr_varpro
    calls = []

    def capture(*args, **kwargs):
        calls.append(np.array(kwargs["weights"], copy=True))
        return original(*args, **kwargs)

    monkeypatch.setattr(nmr, "compute_errors_nmr_varpro", capture)
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], optimizer="powell")
    ed = out["export_data"]
    used = np.asarray(ed["weights_used_for_fit"])
    proposed = np.asarray(ed["irls_weights_proposed_next"])
    np.testing.assert_array_equal(ed["irls_weights_final"], used)
    assert not np.allclose(used, proposed, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(
        [out["final_weights_per_signal"][name] for name in ed["signal_names"]], used
    )
    np.testing.assert_allclose(calls[-1], used, rtol=0, atol=0)
    stats = dict(ed["stats_table"])
    np.testing.assert_allclose([stats["weight s1"], stats["weight s2"]], used, rtol=0, atol=0)
    assert any("conditional" in note for note in ed["uncertainty_notes"])


def test_unit_weights_run_one_cycle_and_are_exported(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], optimizer='powell', weights_mode='unit')
    ed = out['export_data']
    assert ed['weights_mode'] == 'unit'
    assert ed['irls_cycles'] == 1
    assert ed['irls_weights_final'] == [1., 1.]
    assert ed['weights_used_for_fit'] == [1., 1.]
    assert 'Weights: unit; cycles: 1' in out['results_text']
    assert dict(ed['stats_table'])['weight s1'] == 1.


def test_unknown_weight_mode_is_rejected(tmp_path, monkeypatch):
    import pytest
    with pytest.raises(ValueError, match='weights_mode'):
        _fit(tmp_path, monkeypatch, [[1., 5.]], weights_mode='bad')


def test_irls_stop_diagnostic_matches_reported_weight_change(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], optimizer='powell')
    ed = out['export_data']
    diag = ed['irls_diagnostics']
    assert diag['cycles'] == ed['irls_cycles']
    assert diag['stop_reason'] in {'rms_stable', 'weights_stable', 'max_cycles'}
    used = np.asarray(ed['weights_used_for_fit'])
    proposed = np.asarray(ed['irls_weights_proposed_next'])
    np.testing.assert_allclose(diag['max_relative_weight_change'],
                               np.max(np.abs(used - proposed) / (used + 1e-9)))
    assert diag['optimizer_success'] == out['optimizer_result']['success']
    assert 'IRLS Converged' not in out['results_text']


def test_failed_later_cycle_does_not_report_previous_weight_change(tmp_path, monkeypatch):
    original = nmr.optimize.minimize
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError('second cycle failed')
        return original(*args, **kwargs)

    monkeypatch.setattr(nmr.optimize, 'minimize', fail_after_first)
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], optimizer='powell')
    assert not out['success']
    diag = out['irls_diagnostics']
    assert diag['cycles'] == 2
    assert diag['stop_reason'] == 'optimization_failed'
    assert diag['optimizer_success'] is False
    assert diag['max_relative_weight_change'] is None


def test_unit_mode_does_not_claim_irls_convergence(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, [[1., 5.]], optimizer='powell', weights_mode='unit')
    diag = out['export_data']['irls_diagnostics']
    assert diag['stop_reason'] == 'unit_weights'
    assert diag['max_relative_weight_change'] is None
    assert diag['cycles'] == 1
