"""Phase 2, stage 2.5: honesty fixes in the UI (offscreen GUI tests, one block per item)."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QCheckBox, QTableWidgetItem  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------------- 2.5.1
def test_kinetics_algorithm_combo_offers_only_what_runs(qapp):
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab

    tab = KineticsTab()
    combo = tab.model_opt_plots.combo_algorithm
    assert combo.count() == 1
    assert combo.itemData(0) == "trust-region"
    assert "multistart" not in combo.itemText(0).lower()
    assert hasattr(tab.model_opt_plots, "chk_multistart")  # multi-start lives only here


def test_kinetics_legacy_algorithm_configs_load_and_keep_behaviour(qapp):
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab

    tab = KineticsTab()
    tab._apply_config({"algorithm": "multistart", "multi_start_runs": 1})
    assert tab.model_opt_plots.combo_algorithm.currentData() == "trust-region"
    assert tab.model_opt_plots.get_multi_start()[0] == 10  # legacy "multistart" forced 10 starts
    tab._apply_config({"algorithm": "least_squares", "multi_start_runs": 1})
    assert tab.model_opt_plots.combo_algorithm.currentData() == "trust-region"
    assert tab.model_opt_plots.get_multi_start()[0] == 1


# ----------------------------------------------------------------------------- 2.5.2
def _error_widget():
    from hmfit_gui_qt.widgets.error_analysis_widget import ErrorAnalysisWidget

    return ErrorAnalysisWidget(lambda payload, **_k: {})


def _headers(table):
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


@pytest.mark.parametrize(
    "names, label",
    [(["pKa1", "pKa2"], "pKa"), (["log_beta1", "log_beta2"], "log β"), (["K1", "K2"], "log10K")],
)
def test_errors_header_names_the_real_parameter(qapp, names, label):
    w = _error_widget()
    w.set_errors_context({"parameter_names": names, "param_names": names})
    headers = _headers(w.table_error_results)
    assert f"Estimate ({label})" in headers and f"SE ({label})" in headers
    assert "Estimate log10K" not in headers or label == "log10K"
    w._apply_errors_output({
        "param_names": names, "results_columns": ["Parameter", "Estimate", "SE", "CI 2.5", "CI 97.5", "Fixed"],
        "results_rows": [[n, 1.0, 0.1, 0.8, 1.2, ""] for n in names], "corr": np.eye(2), "summary": "",
        "param_status": ["", ""],
    })
    headers = _headers(w.table_error_results)
    assert headers[1] == f"Estimate ({label})" and headers[2] == f"SE ({label})"


def test_acid_base_header_no_longer_says_log10k(qapp):
    w = _error_widget()
    w.set_errors_context({"parameter_names": ["pKa1"]})
    assert not any("log10K" in h for h in _headers(w.table_error_results))


# ----------------------------------------------------------------------------- 2.5.3
def _fill_binding_table(widget, rows):
    t = widget.params_table
    t.setRowCount(len(rows))
    for r, (name, val, lo, hi, fixed) in enumerate(rows):
        for c, text in enumerate((name, val, lo, hi)):
            t.setItem(r, c, QTableWidgetItem(str(text)))
        item = QTableWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked)
        t.setItem(r, 4, item)


def test_binding_inverted_bounds_message_names_parameter_and_fix(qapp):
    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    w = ModelOptPlotsWidget()
    w.params_table.blockSignals(True)
    _fill_binding_table(w, [("K1", "3", "0", "8", False), ("K2", "0.3", "0.5", "0.1", False)])
    with pytest.raises(ValueError) as exc:
        w.validate_parameter_table()
    msg = str(exc.value)
    assert "K2: the lower bound (0.5) is greater than the upper bound (0.1)." in msg
    assert "Swap them, or set both equal to fix the parameter." in msg
    assert "K1" not in msg


def test_binding_non_numeric_value_is_rejected_not_replaced(qapp):
    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    w = ModelOptPlotsWidget()
    w.params_table.blockSignals(True)
    _fill_binding_table(w, [("K1", "abc", "", "nan", False)])
    with pytest.raises(ValueError) as exc:
        w.validate_parameter_table()
    msg = str(exc.value)
    assert "K1: the initial value ('abc') is not a number" in msg
    assert "K1: the upper bound is NaN" in msg


def test_kinetics_initial_value_outside_bounds_message(qapp):
    from hmfit_gui_qt.widgets.kinetics_model_opt_plots import KineticsModelOptPlotsWidget

    w = KineticsModelOptPlotsWidget()
    w.set_params(["k1", "k2"])
    for row, (val, lo, hi) in enumerate((("1.0", "0", "10"), ("5.0", "0.1", "2"))):
        for col, text in zip((1, 2, 3), (val, lo, hi)):
            w.params_table.setItem(row, col, QTableWidgetItem(text))
    issues = w.validate_parameter_table()
    assert issues == ["k2: the initial value (5) is outside its bounds [0.1, 2]. "
                      "Move it inside the interval or widen the bounds."]


def test_acid_base_pka_table_message(qapp, monkeypatch):
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    tab = AcidBaseTab()
    t = tab.table_basic_pka
    t.blockSignals(True)
    t.setRowCount(1)
    for col, text in enumerate(("pKa1", "x7", "3", "1")):
        t.setItem(0, col, QTableWidgetItem(text))
    t.setCellWidget(0, 4, QCheckBox())
    t.cellWidget(0, 4).setChecked(True)
    with pytest.raises(ValueError) as exc:
        tab._validate_pka_table()
    msg = str(exc.value)
    assert "pKa1: the initial value ('x7') is not a number" in msg
    assert "pKa1: the lower bound (3) is greater than the upper bound (1)." in msg


# ----------------------------------------------------------------------------- 2.5.4
def test_undefined_values_show_marker_with_reason_and_export_keeps_nan(qapp):
    from hmfit_gui_qt.widgets.error_analysis_widget import UNDEFINED_MARK

    w = _error_widget()
    out = {
        "param_names": ["K1", "K2", "K3"], "k_hat": [3.0, 2.0, 1.0], "median": [3.0, 2.0, 1.0],
        "ci_2p5": [2.9, 2.0, np.nan], "ci_97p5": [3.1, 2.0, np.nan], "ci_16": None, "ci_84": None,
        "se_log10": [0.05, 0.0, np.nan], "perc_err": [11.0, 0.0, np.nan],
        "corr": np.array([[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        "summary": "", "param_status": ["", "fixed", "nonidentifiable"],
    }
    w.chk_error_show_corr.setChecked(True)
    w._apply_errors_output(out)
    t = w.table_error_results
    se_cell = t.item(2, 7)
    assert se_cell.text() == UNDEFINED_MARK and "nan" not in se_cell.text().lower()
    assert "not identifiable" in se_cell.toolTip()
    assert "Fixed parameter" in t.item(1, 0).toolTip() and t.item(1, 7).text() == "0"
    c = w.table_error_corr
    assert c.item(0, 1).text() == UNDEFINED_MARK and "fixed or non-identifiable" in c.item(0, 1).toolTip()
    notes = w._export_notes(out, 3)
    assert notes[0] == "" and "fixed" in notes[1] and "not identifiable" in notes[2]


@pytest.mark.parametrize("code, text", [("dof", "degrees of freedom"), ("failed", "could not compute")])
def test_every_undefined_cause_has_a_reason(qapp, code, text):
    from hmfit_gui_qt.widgets.error_analysis_widget import status_reason_text

    assert text in status_reason_text(lambda s: s, code)


def test_kinetics_table_marks_undefined(qapp):
    from hmfit_gui_qt.widgets.error_analysis_widget import UNDEFINED_MARK
    from hmfit_gui_qt.widgets.kinetics_model_opt_plots import KineticsModelOptPlotsWidget

    w = KineticsModelOptPlotsWidget()
    w._apply_errors_output({
        "param_names": ["k1", "k2"], "k_hat": [1.0, 0.5], "median": [1.0, 0.5], "ci_2p5": [0.9, np.nan],
        "ci_97p5": [1.1, np.nan], "ci_16": None, "ci_84": None, "se": [0.05, np.nan], "perc_err": [5.0, np.nan],
        "corr": np.array([[1.0, np.nan], [np.nan, np.nan]]), "summary": "", "param_status": ["", "dof"],
    })
    assert w.table_error_results.item(1, 7).text() == UNDEFINED_MARK
    assert "degrees of freedom" in w.table_error_results.item(1, 7).toolTip()


def test_acid_base_export_has_note_column_and_numeric_nan(tmp_path):
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from tests.test_phase2_invariants_matrix import _acid_base_case

    ctx, _ = _acid_base_case("nmr", "nonidentifiable", str(tmp_path))
    out = compute_errors_acid_base_from_context(ctx, {"method": "analytic"})
    params = out["export_frames"]["Parameters"]
    assert "Note" in params.columns
    assert np.isnan(float(params.loc[1, "SE"]))
    assert "not identifiable" in params.loc[1, "Note"] and params.loc[0, "Note"] == ""
