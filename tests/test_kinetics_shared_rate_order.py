"""A shared rate parameter must have the same physical units in every reaction."""
import pytest

pytest.importorskip("PySide6.QtWidgets")

from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab
from PySide6.QtWidgets import QApplication


def test_legacy_config_with_incompatible_shared_rate_opens_but_cannot_fit():
    app = QApplication.instance() or QApplication([])
    text = "species: A, B, C\nA -> B ; k\nA + B -> C ; k\n"
    tab = KineticsTab()
    tab._apply_config({"mechanism_text": text, "params": {"k": {"value": 2.0}}})
    assert tab.model_opt_plots.mechanism_text() == text
    assert tab.model_opt_plots.get_params_state().values == [2.0]
    error = tab.model_opt_plots.mechanism_summary.toPlainText()
    assert all(s in error for s in ("k", "A -> B", "A + B -> C", "s⁻¹", "M^-1 s⁻¹"))
    assert tab._model is None
    assert not tab.model_opt_plots.btn_process.isEnabled()
    tab.deleteLater()
    assert app is not None


def test_same_order_shared_rate_is_valid():
    app = QApplication.instance() or QApplication([])
    tab = KineticsTab()
    tab.model_opt_plots.set_mechanism_text(
        "species: A, B, C, P\nA + B -> P ; k\nA + C -> P ; k\n"
    )
    tab._on_validate_mechanism()
    assert tab._model is not None
    tab.deleteLater()
    assert app is not None
