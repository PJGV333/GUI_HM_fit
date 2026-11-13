"""Retired optimizer must not be silently replaced when loading projects."""
import pytest


@pytest.mark.gui
@pytest.mark.parametrize("technique", ["nmr", "spectroscopy"])
def test_explicit_lm_not_selectable_or_silently_loaded(technique):
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab
    from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab

    app = QApplication.instance() or QApplication([])
    tab = (NMRTab if technique == "nmr" else SpectroscopyTab)()
    try:
        combo = tab.model_opt_plots.combo_optimizer
        before = combo.currentText()
        assert combo.findText("explicit_lm") == -1
        assert combo.findText("powell") >= 0
        with pytest.raises(ValueError, match="(?i)explicit_lm.*powell"):
            tab.load_config({"optimizer": "explicit_lm", "file_path": "does-not-exist.xlsx"})
        assert combo.currentText() == before
    finally:
        tab.close()
        app.processEvents()
