import pytest


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
        tab.load_config(dict(cfg, weights_mode='unit', optimizer='explicit_lm'))
        assert tab.combo_weights.currentData() == 'unit'
        assert tab._collect_config()['weights_mode'] == 'unit'
        assert tab._collect_config()['optimizer'] == 'explicit_lm'
        tab.load_config(cfg)  # old saved configuration has no weight setting
        assert tab._collect_config()['weights_mode'] == 'irls'
        assert tab._collect_config()['optimizer'] == 'powell'
    finally:
        tab.close()
        app.processEvents()
