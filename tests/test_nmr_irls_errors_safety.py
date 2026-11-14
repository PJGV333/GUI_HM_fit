"""Do not present unit-weight NMR errors as IRLS uncertainty."""
import types
from typing import Any, cast

import pytest


@pytest.mark.parametrize("method", ["analytic", "bootstrap_linear", "bootstrap_onestep", "bootstrap_full_refit_audit"])
@pytest.mark.parametrize("mode", ["irls", None])
def test_irls_and_unknown_weight_errors_are_unavailable(method, mode):
    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    from PySide6.QtWidgets import QApplication
    from tests.test_phase2_invariants_matrix import _nmr_case

    app = QApplication.instance() or QApplication([])
    ctx, _ = _nmr_case("mixed")
    ctx["weights_mode"] = mode
    widget = ModelOptPlotsWidget()
    try:
        with pytest.raises(ValueError, match="(?i)(irls|weight).*(unavailable|not available)"):
            widget._compute_errors_from_context(ctx, options={"method": method, "B": 3})
    finally:
        widget.deleteLater()
        app.processEvents()


def test_nmr_result_context_preserves_fitted_weight_mode():
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab

    captured = {}
    fake = types.SimpleNamespace(_last_config={}, _last_fit_context=None,
        _refit_from_data=lambda *args: None,
        model_opt_plots=types.SimpleNamespace(set_errors_context=lambda context, **kwargs: captured.update(context)))
    result = {"success": True, "export_data": {"k": [3.0], "weights_mode": "irls",
                                                "weights_used_for_fit": [2.0, 3.0]}}
    NMRTab._update_errors_context_from_result(cast(Any, fake), result)
    assert captured["weights_mode"] == "irls"
    assert captured["weights_used_for_fit"] == [2.0, 3.0]


@pytest.mark.parametrize("mode", ["irls", None])
def test_direct_refit_rejects_irls_and_unknown_weights(mode):
    import numpy as np
    from hmfit_gui_qt.tabs.nmr_tab import NMRTab

    tab = types.SimpleNamespace(_last_fit_context={"weights_mode": mode})
    initial = np.array([3.0])
    estimate, ok, details = NMRTab._refit_from_data(cast(Any, tab), [], initial)
    np.testing.assert_array_equal(estimate, initial)
    assert not ok
    assert details["failure_reason"] == "unsupported_weights"


def test_irls_context_disables_automatic_errors_and_explains_why():
    from PySide6.QtWidgets import QApplication
    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    app = QApplication.instance() or QApplication([])
    widget = ModelOptPlotsWidget()
    try:
        widget.set_errors_context({"technique": "nmr", "weights_mode": "irls"}, auto_compute=True)
        panel = widget.errors_panel
        assert panel is not None
        assert panel._errors_worker is None
        assert not panel.btn_error_compute.isEnabled()
        assert "unavailable" in panel.text_error_summary.toPlainText()
        widget.set_errors_context({"technique": "nmr", "weights_mode": "unit"}, auto_compute=False)
        assert panel.btn_error_compute.isEnabled()
    finally:
        widget.deleteLater()
        app.processEvents()
