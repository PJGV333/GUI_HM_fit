import numpy as np
import pytest


@pytest.mark.gui
@pytest.mark.parametrize('technique', ['spectro', 'nmr'])
def test_retired_lm_full_refit_is_rejected_without_fallback(technique):
    from tests.test_phase2_invariants_matrix import _fake_tab, _nmr_case, _spectro_case
    if technique == 'spectro':
        from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab
        ctx, _ = _spectro_case('mixed')
        data = np.asarray(ctx['Y'])
        cls = SpectroscopyTab
    else:
        from hmfit_gui_qt.tabs.nmr_tab import NMRTab
        ctx, _ = _nmr_case('mixed')
        data = np.asarray(ctx['dq'])
        cls = NMRTab
    ctx['optimizer'] = 'explicit_lm'
    tab = _fake_tab(cls, ctx)
    estimate, ok, info = tab._refit_from_data(data, np.asarray(ctx['k_hat']), max_iter=150, tol=1e-8)
    assert not ok
    assert info['failure_reason'] == 'unsupported_optimizer'
    assert 'explicit_lm' in info['error'] and 'powell' in info['error']
    assert info['optimizer_requested'] == 'explicit_lm'
    assert info['optimizer_used'] is None
    assert not info['fallback_used']
    np.testing.assert_array_equal(estimate, np.asarray(ctx['k_hat']))


@pytest.mark.gui
@pytest.mark.parametrize('technique', ['spectro', 'nmr'])
def test_error_tab_marks_bound_se_undefined(technique):
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget
    from tests.test_phase2_invariants_matrix import _nmr_case, _spectro_case
    app = QApplication.instance() or QApplication([])
    ctx, _ = (_spectro_case('mixed') if technique == 'spectro' else _nmr_case('mixed'))
    ctx['active_bound_mask'] = [True, False]
    widget = ModelOptPlotsWidget()
    try:
        out = widget._compute_errors_from_context(ctx, options={'method': 'analytic'})
        assert np.isnan(out['se_log10'][0])
        assert out['se_log10'][1] == 0  # fixed parameters keep phase 1 convention
        assert out['param_status'][0] == 'bound'
        assert 'límite' in out['summary']
    finally:
        widget.deleteLater()
        app.processEvents()
