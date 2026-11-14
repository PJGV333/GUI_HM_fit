import pytest


@pytest.mark.gui
def test_new_and_reset_nmr_projects_default_to_unit_weights():
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab

    app = QApplication.instance() or QApplication([])
    tab = NMRTab()
    try:
        assert tab.combo_weights.currentData() == 'unit'
        tab.combo_weights.setCurrentIndex(tab.combo_weights.findData('irls'))
        tab.reset_tab()
        assert tab.combo_weights.currentData() == 'unit'
    finally:
        tab.close()
        app.processEvents()


@pytest.mark.gui
def test_nmr_weight_selection_roundtrips_and_legacy_defaults_to_irls(tmp_path):
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab
    from validation import literature_benchmark as lit

    app = QApplication.instance() or QApplication([])
    case = lit.load_case('r09a-r1-11-me')
    conc, shifts = lit.numeric_table(case)
    cfg = lit.gui_config(case, str(tmp_path / 'case.xlsx'), conc, shifts)
    tab = NMRTab()
    try:
        tab.load_config(dict(cfg, weights_mode='unit', optimizer='powell'))
        assert tab.combo_weights.currentData() == 'unit'
        assert tab._collect_config()['weights_mode'] == 'unit'
        assert tab._collect_config()['weights_schema_version'] == 1
        assert tab._collect_config()['optimizer'] == 'powell'
        saved = tab._collect_config()
        tab.load_config(saved)
        assert tab._collect_config()['weights_mode'] == 'unit'
        tab.load_config(cfg)  # old saved configuration has no weight setting
        assert tab._collect_config()['weights_mode'] == 'irls'
        assert tab._collect_config()['weights_schema_version'] == 1
        assert tab._collect_config()['optimizer'] == 'powell'
    finally:
        tab.close()
        app.processEvents()


@pytest.mark.gui
@pytest.mark.parametrize('extra', [
    {'weights_mode': 'unknown'},
    {'weights_mode': None},
    {'weights_schema_version': 1},
    {'weights_mode': 'unit', 'weights_schema_version': 2},
])
def test_unsupported_weight_configuration_does_not_modify_tab(extra):
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab

    app = QApplication.instance() or QApplication([])
    tab = NMRTab()
    try:
        before = (tab.combo_weights.currentData(), tab._file_path)
        with pytest.raises(ValueError, match='(?i)weight'):
            tab.load_config(dict(extra, file_path='does-not-exist.xlsx'))
        assert (tab.combo_weights.currentData(), tab._file_path) == before
    finally:
        tab.close()
        app.processEvents()


def test_versioned_and_legacy_nmr_runs_keep_selected_estimator(tmp_path):
    import numpy as np
    from hmfit_core.api import run_nmr_fit
    from validation import literature_benchmark as lit

    case = lit.load_case('r09a-r1-11-me')
    conc, shifts = lit.numeric_table(case)
    cfg = lit.gui_config(case, str(tmp_path / 'case.xlsx'), conc, shifts)
    legacy = run_nmr_fit(cfg, progress_cb=lambda _message: None)
    irls = run_nmr_fit({**cfg, 'weights_mode': 'irls', 'weights_schema_version': 1},
                       progress_cb=lambda _message: None)
    unit = run_nmr_fit({**cfg, 'weights_mode': 'unit', 'weights_schema_version': 1},
                       progress_cb=lambda _message: None)
    for result in (legacy, irls, unit):
        assert result['success']
    a, b, c = [result['export_data'] for result in (legacy, irls, unit)]
    assert a['weights_mode'] == b['weights_mode'] == 'irls'
    assert c['weights_mode'] == 'unit'
    for result, data in zip((legacy, irls, unit), (a, b, c)):
        np.testing.assert_array_equal(
            [result['final_weights_per_signal'][name] for name in data['signal_names']],
            data['weights_used_for_fit'],
        )
    np.testing.assert_array_equal(a['k'], b['k'])
    np.testing.assert_array_equal(a['weights_used_for_fit'], b['weights_used_for_fit'])
    np.testing.assert_array_equal(c['weights_used_for_fit'], np.ones(len(c['signal_names'])))
