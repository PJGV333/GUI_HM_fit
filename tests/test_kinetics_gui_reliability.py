"""Kinetics and acid-base GUI defects found in the November 2026 review (Qt offscreen)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6.QtWidgets")

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset  # noqa: E402
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism  # noqa: E402
from hmfit.kinetics.model.kinetics_model import KineticsModel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _config(model, datasets, names, p0, **extra):
    cfg = {"model": model, "datasets": datasets, "params0": list(p0), "bounds": (0.0, np.inf),
           "param_names": list(names), "log_params": [], "fixed_mask": [False] * len(names),
           "method": "trust-region", "backend": "scipy", "nnls": False}
    cfg.update(extra)
    return cfg


def _abc(sigma=2e-3, nch=6, seed=3):
    mech = "species: A, B, C\nA -> B ; k1\nB -> C ; k2\n"
    t = np.linspace(0.0, 600.0, 40)
    y0 = {"A": 1e-3, "B": 0.0, "C": 0.0}
    model = KineticsModel(parse_mechanism(mech))
    C = model.solve_concentrations(t, y0, {"k1": 0.05, "k2": 0.004})
    rng = np.random.default_rng(seed)
    D = C @ ((rng.random((3, nch)) + 0.2) * 1000.0) + rng.normal(0.0, sigma, (40, nch))
    return model, t, y0, D


def test_multistart_with_default_table_bounds_runs(qapp):
    """min = 0 / no max (the table defaults): random starts were negative and the fit aborted."""
    from hmfit_gui_qt.tabs.kinetics_tab import _run_kinetics_fit

    model, t, y0, D = _abc()
    cfg = _config(model, [KineticsFitDataset(t=t, y0=y0, D=D)], ["k1", "k2"], [0.03, 0.006],
                  bounds=(np.zeros(2), np.full(2, np.inf)), multi_start_runs=5,
                  multi_start_seeds=[1, 2, 3, 4, 5])
    res = _run_kinetics_fit(cfg, progress_cb=lambda *_a, **_k: None)
    # equivalent minima keep the solution reached from the user's own start
    assert res["params"]["k1"] == pytest.approx(0.05, rel=0.05)
    assert res["params"]["k2"] == pytest.approx(0.004, rel=0.05)


def test_fit_report_carries_reliability_warnings(qapp):
    from hmfit_gui_qt.tabs.kinetics_tab import _run_kinetics_fit

    model, t, y0, D = _abc()
    ds = [KineticsFitDataset(t=t, y0=y0, D=D, name="run1")]
    good = _run_kinetics_fit(_config(model, ds, ["k1", "k2"], [0.03, 0.006]), progress_cb=lambda *_a, **_k: None)
    assert good["results_text"].startswith("Fit complete.")
    assert "Two equivalent solutions" in good["results_text"]          # slow-fast ambiguity
    assert good["diagnostics"]["equivalent_solutions"][0]["a"] == "k1"
    assert "run1: y0: A=0.001" in good["results_text"]
    assert good["diagnostics"]["residual_serial_correlation"]["level"] == "ok"

    stuck = _run_kinetics_fit(_config(model, ds, ["k1", "k2"], [1e4, 1e4]), progress_cb=lambda *_a, **_k: None)
    assert stuck["results_text"].startswith(">>> FIT NOT RELIABLE")
    assert stuck["diagnostics"]["convergence_warnings"]

    # wrong mechanism (one step for two-step data): structured residuals are reported
    one_step = KineticsModel(parse_mechanism("species: A, B\nA -> B ; k1\n"))
    ds1 = [KineticsFitDataset(t=t, y0={"A": 1e-3, "B": 0.0}, D=D)]
    wrong = _run_kinetics_fit(_config(one_step, ds1, ["k1"], [0.01]), progress_cb=lambda *_a, **_k: None)
    assert wrong["diagnostics"]["residual_serial_correlation"]["level"] in {"warn", "strong"}
    assert any("correlated along time" in w for w in wrong["diagnostics"]["warnings"])


def test_global_fit_keeps_per_dataset_concentrations(qapp):
    """
    Pressing Process copied the concentrations of the displayed dataset to every dataset:
    a global fit of A + B -> P at [B]0 = 1, 2 and 4 mM used 2 mM for all three and gave
    k = 41.9 instead of 50, without any message.
    """
    from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab

    mech = "species: A, B, P\nA + B -> P ; k\n"
    model = KineticsModel(parse_mechanism(mech))
    t = np.linspace(0.0, 200.0, 40)
    tab = KineticsTab()
    for i, b0 in enumerate((1e-3, 2e-3, 4e-3)):
        y0 = {"A": 1e-4, "B": b0, "P": 0.0}
        C = model.solve_concentrations(t, y0, {"k": 50.0})
        D = C @ np.array([[5000.0], [0.0], [12000.0]]) + np.random.default_rng(i).normal(0, 1e-3, (40, 1))
        tab._datasets.append(KineticsFitDataset(t=t, y0=dict(y0), D=D, name=f"B0={b0:g}"))
    tab._refresh_dataset_list()
    tab.model_opt_plots.set_mechanism_text(mech)
    tab._on_validate_mechanism()
    tab._dataset_list.setCurrentRow(1)
    tab._sync_selected_metadata_from_tables()          # what Process does first
    assert [ds.y0["B"] for ds in tab._datasets] == [1e-3, 2e-3, 4e-3]
    # editing the table changes the selected dataset only
    tab.model_opt_plots.y0_table.cellWidget(1, 1).setValue(2.5e-3)
    assert [ds.y0["B"] for ds in tab._datasets] == [1e-3, 2.5e-3, 4e-3]
    # a dataset without concentrations is still filled from the table (one edit for all)
    tab._datasets[2].y0 = None
    tab._sync_selected_metadata_from_tables()
    assert tab._datasets[2].y0["B"] == 2.5e-3 and tab._datasets[0].y0["B"] == 1e-3
    tab.deleteLater()


def test_concentration_fields_keep_small_values(qapp):
    """A 6-decimal spin box turned 2.5e-7 M into 0 and 1.5e-6 M into 2e-6 M."""
    from PySide6.QtWidgets import QDoubleSpinBox
    from hmfit.kinetics.gui.dataset_editor import _make_spinbox
    from hmfit_gui_qt.widgets.kinetics_model_opt_plots import KineticsModelOptPlotsWidget

    widget = KineticsModelOptPlotsWidget()
    widget.set_species_tables(["A", "B"], ["S"])
    values = {"A": 2.5e-7, "B": 1.5e-6}
    widget.set_y0_values(values)
    widget.set_fixed_conc_values({"S": 4.4e-9})
    assert widget.get_y0_values() == values
    assert widget.get_fixed_conc_values() == {"S": 4.4e-9}
    spin = widget.y0_table.cellWidget(0, 1)
    assert isinstance(spin, QDoubleSpinBox) and spin.text() == "2.5e-07" and spin.minimum() == 0.0
    editor_spin = _make_spinbox(widget)
    editor_spin.setValue(3.3e-8)
    assert editor_spin.value() == 3.3e-8
    assert editor_spin.validate("2.5e-", 5)[0].name == "Intermediate"
    assert editor_spin.validate("abc", 3)[0].name == "Invalid"
    assert editor_spin.validate("2.5e-7", 6)[0].name == "Acceptable"
    widget.deleteLater()


def test_mechanism_examples_parse_and_unfinished_buttons_are_hidden(qapp):
    from hmfit_gui_qt.widgets.kinetics_model_opt_plots import MECHANISM_TEMPLATES, KineticsModelOptPlotsWidget

    for _label, text in MECHANISM_TEMPLATES:
        KineticsModel(parse_mechanism(text))
    widget = KineticsModelOptPlotsWidget()
    widget._on_mechanism_template_chosen(2)
    assert widget.mechanism_text() == MECHANISM_TEMPLATES[1][1]
    assert widget.combo_mechanism_template.currentIndex() == 0
    assert widget.btn_export_png.isHidden() and widget.btn_export_csv.isHidden()
    widget.deleteLater()


def test_acid_base_data_type_is_reachable_and_guessed(qapp, tmp_path):
    """The Data type selector was hidden: spectroscopy/NMR vs pH could not be chosen in the GUI."""
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab, _guess_data_type

    assert _guess_data_type(["volume_mL", "pH"]) == "potentiometry"
    assert _guess_data_type(["V (uL)", "E_mV"]) == "potentiometry"
    assert _guess_data_type(["pH", "signal"]) == "spectroscopy"
    assert _guess_data_type(["pH", "250", "260", "270"]) == "spectroscopy"
    assert _guess_data_type(["pH", "H1", "H2", "H3a"]) == "nmr"
    assert _guess_data_type(["pH"]) is None and _guess_data_type(["x", "y"]) is None

    tab = AcidBaseTab()
    assert not tab.combo_data_type.isHidden() and not tab.lbl_data_type.isHidden()
    cases = {"spec.csv": ({"pH": [4.0, 5.0, 6.0], "signal": [0.2, 0.5, 0.8]}, "spectroscopy"),
             "nmr.csv": ({"pH": [4.0, 5.0, 6.0], "H1": [7.1, 7.3, 7.6]}, "nmr"),
             "pot.csv": ({"volume_mL": [0.0, 0.5, 1.0], "pH": [3.1, 3.4, 3.9]}, "potentiometry")}
    for name, (data, expected) in cases.items():
        path = tmp_path / name
        pd.DataFrame(data).to_csv(path, index=False)
        tab._set_file_path(str(path))
        assert tab.combo_data_type.currentData() == expected
    tab.deleteLater()
