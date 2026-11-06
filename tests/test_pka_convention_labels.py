"""pKa convention made visible (author decision, Phase 2): HM fit keeps pKa1 = log beta1,
i.e. pKa_n belongs to H_nL <=> H_(n-1)L + H+ and pKa1 is the most basic site.

(a) every pKa label names its equilibrium, (b) an inverted order (pKa_(n+1) > pKa_n) is
warned about and never silently reordered, (c) exports carry the equilibrium and the
convention note.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hmfit_core.acid_base import (  # noqa: E402
    distribution_fractions_from_pH,
    inverted_pka_pairs,
    inverted_pka_warning,
    make_simple_acid_base_system,
    pka_equilibria,
    pka_to_log_beta,
)
from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context  # noqa: E402
from hmfit_core.run_acid_base import run_acid_base  # noqa: E402


def test_equilibrium_of_each_pka():
    system = make_simple_acid_base_system(pka=[9.0, 6.0, 3.0], base_charge=-3)
    assert pka_equilibria(system) == ["HL ⇌ L + H⁺", "H2L ⇌ HL + H⁺", "H3L ⇌ H2L + H⁺"]


def test_inverted_order_detection():
    assert inverted_pka_pairs([8.0, 4.0]) == []
    assert inverted_pka_pairs([4.0, 8.0]) == [(1, 2)]
    assert inverted_pka_pairs([9.0, 4.0, 6.0]) == [(2, 3)]
    msg = inverted_pka_warning([4.0, 8.0], ["HL ⇌ L + H⁺", "H2L ⇌ HL + H⁺"])
    assert "pKa2 (H2L ⇌ HL + H⁺) = 8 > pKa1 (HL ⇌ L + H⁺) = 4" in msg
    assert "not reordered" in msg
    assert inverted_pka_warning([8.0, 4.0]) == ""


def _nmr_fit(tmp_path, pka_true, pka0):
    pH = np.linspace(2.0, 11.0, 25)
    y = distribution_fractions_from_pH(pH, pka_to_log_beta(pka_true)) @ np.array([7.9, 7.5, 7.0])
    y = y + np.random.default_rng(0).normal(0.0, 2e-3, pH.size)
    path = tmp_path / "ab.csv"
    pd.DataFrame({"pH": pH, "H1": y}).to_csv(path, index=False)
    log: list[str] = []
    res = run_acid_base({"file_path": str(path), "data_type": "nmr", "pka_initial": ",".join(map(str, pka0)),
                         "analyte_concentration": 1e-3, "base_charge": -2}, progress_cb=log.append)
    return res, log


def test_fit_results_and_exports_name_the_equilibrium(tmp_path):
    res, log = _nmr_fit(tmp_path, [8.0, 4.0], [7.7, 4.3])
    assert not any("Inverted" in m for m in log)
    text = res["results_text"]
    assert "pKa1 (HL ⇌ L + H⁺):" in text and "pKa2 (H2L ⇌ HL + H⁺):" in text
    assert "pKa1 is the most basic site" in text and "Warning" not in text
    rows = res["export_data"]["pka_table"]
    assert [r["equilibrium"] for r in rows] == ["HL ⇌ L + H⁺", "H2L ⇌ HL + H⁺"]
    assert any("pKa convention" in line for line in res["export_data"]["pka_convention"])
    out = compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    frame = out["export_frames"]["pKa"]
    assert list(frame["Parameter"]) == ["pKa1", "pKa2"]
    assert list(frame["Equilibrium"]) == ["HL ⇌ L + H⁺", "H2L ⇌ HL + H⁺"]


def test_inverted_initial_values_warn_and_are_not_reordered(tmp_path):
    res, log = _nmr_fit(tmp_path, [8.0, 4.0], [4.3, 7.7])
    assert any(m.startswith("Warning (initial values): Inverted protonation constants") for m in log)
    # the entered order is used as is: no silent swap of the initial vector
    assert res["errors_context"]["parameter_names"] == ["pKa1", "pKa2"]


def test_xlsx_export_has_equilibrium_and_convention_sheet(tmp_path):
    from hmfit_core.exports import write_results_xlsx

    res, _ = _nmr_fit(tmp_path, [8.0, 4.0], [7.7, 4.3])
    out = tmp_path / "res.xlsx"
    write_results_xlsx(str(out), constants=res["constants"], statistics=res["statistics"],
                       results_text=res["results_text"], export_data=res["export_data"])
    sheets = pd.read_excel(out, sheet_name=None)
    assert "pKa_convention" in sheets
    assert "most basic site" in " ".join(sheets["pKa_convention"].astype(str).values.ravel())
    pka_sheet = sheets["pKa"].astype(str)
    assert "HL ⇌ L + H⁺" in " ".join(pka_sheet.values.ravel())


# ------------------------------------------------------------------ GUI (offscreen)
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_gui_pka_table_labels_warning_and_defaults(qapp):
    from PySide6.QtWidgets import QTableWidgetItem

    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    tab = AcidBaseTab()
    tab._generate_basic_species(2)
    t = tab.table_basic_pka
    assert t.rowCount() == 2
    assert t.item(0, 0).text() == "pKa1 (HL ⇌ L + H⁺)"
    assert t.item(1, 0).text() == "pKa2 (H2L ⇌ HL + H⁺)"
    assert [tab._basic_pka_key(r) for r in range(2)] == ["pKa1", "pKa2"]
    defaults = [float(t.item(r, 1).text()) for r in range(2)]
    assert defaults[0] > defaults[1]  # default guesses follow the convention
    assert tab._update_pka_order_warning() == ""
    t.setItem(0, 1, QTableWidgetItem("3.0"))  # now pKa1 = 3 < pKa2
    msg = tab._update_pka_order_warning()
    assert "Inverted protonation constants" in msg and not tab.lbl_pka_order_warning.isHidden()
    assert float(t.item(0, 1).text()) == 3.0  # not reordered
    assert [row["parameter"] for row in tab._basic_pka_rows()] == ["pKa1", "pKa2"]


def test_gui_equation_examples_follow_the_convention(qapp):
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    tab = AcidBaseTab()
    placeholder = tab.equation_editor.placeholderText()
    assert "L + H <=> HL ; pKa=8.90" in placeholder and "HL + H <=> H2L ; pKa=4.50" in placeholder


def test_errors_table_shows_equilibrium(qapp):
    from hmfit_gui_qt.widgets.error_analysis_widget import ErrorAnalysisWidget

    w = ErrorAnalysisWidget(lambda payload, **_k: {})
    w.set_errors_context({"parameter_names": ["pKa1", "pKa2"], "pka_equilibria": ["HL ⇌ L + H⁺", "H2L ⇌ HL + H⁺"]})
    w._apply_errors_output({
        "param_names": ["pKa1", "pKa2"], "results_columns": ["Parameter", "Estimate", "SE", "CI 2.5", "CI 97.5", "Fixed"],
        "results_rows": [["pKa1", 8.0, 0.1, 7.8, 8.2, ""], ["pKa2", 4.0, 0.1, 3.8, 4.2, ""]],
        "corr": np.eye(2), "summary": "", "param_status": ["", ""],
    })
    assert w.table_error_results.item(0, 0).text() == "pKa1 (HL ⇌ L + H⁺)"
    assert w.table_error_results.item(1, 0).text() == "pKa2 (H2L ⇌ HL + H⁺)"
