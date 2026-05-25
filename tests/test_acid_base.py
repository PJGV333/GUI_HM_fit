from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmfit_core.acid_base import (
    NMRAcidBaseDataset,
    SpectroscopicAcidBaseDataset,
    distribution_fractions_from_pH,
    fit_nmr_acid_base,
    fit_spectroscopy_acid_base,
    log_beta_to_pka,
    make_simple_acid_base_system,
    pka_to_log_beta,
)
from hmfit_core.acid_base_model_utils import (
    acid_base_model_from_equations,
    build_acid_base_template,
    canonicalize_acid_base_model,
)
from hmfit_core.potentiometry import (
    PotentiometryExperiment,
    fit_potentiometry,
    simulate_pH_titration,
)
from hmfit_core.run_acid_base import _build_system, run_acid_base


def test_pka_log_beta_roundtrip():
    pka = [4.0, 7.0]
    log_beta = pka_to_log_beta(pka)
    pka_back = log_beta_to_pka(log_beta)
    assert np.allclose(pka_back, pka, atol=1e-12, rtol=0.0)


def test_distribution_fractions_monoprotic():
    fractions = distribution_fractions_from_pH(np.asarray([2.0, 5.0, 8.0]), [5.0])
    assert np.allclose(fractions.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    assert np.allclose(fractions[1], [0.5, 0.5], atol=1e-12, rtol=0.0)
    assert fractions[0, 1] > 0.999
    assert fractions[2, 0] > 0.999


def test_synthetic_potentiometry_recovers_pka():
    rng = np.random.default_rng(1234)
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[5.0],
        base_charge=-1,
    )
    volumes = np.linspace(0.0, 18.0, 45)
    experiment_template = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment_template, true_system)
    measured = pH + rng.normal(0.0, 1.0e-3, size=pH.shape)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        measured_pH=measured,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    start_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[4.4],
        base_charge=-1,
    )
    result = fit_potentiometry(experiment, start_system, initial_pka=[4.4])
    assert result.success
    assert abs(result.fitted_pka[0] - 5.0) < 0.03


def test_synthetic_potentiometry_keeps_fixed_pkw_out_of_fit():
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[7.0],
        base_charge=-1,
        kw=1.0e-14,
    )
    volumes = np.linspace(0.0, 18.0, 45)
    experiment_template = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment_template, true_system)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        measured_pH=pH,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    start_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[6.5],
        base_charge=-1,
        kw=1.0e-14,
    )
    result = fit_potentiometry(experiment, start_system, initial_pka=[6.5])
    assert result.success
    assert "pKw" not in result.parameter_names


def test_synthetic_potentiometry_can_fit_apparent_pkw():
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[7.0],
        base_charge=-1,
        kw=1.0e-18,
    )
    volumes = np.linspace(0.0, 28.0, 70)
    experiment_template = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    fit_options = {"pH_bounds": (-2.0, 24.0)}
    pH = simulate_pH_titration(experiment_template, true_system, fit_options)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        measured_pH=pH,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    start_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[6.4],
        base_charge=-1,
        kw=1.0e-14,
    )
    result = fit_potentiometry(
        experiment,
        start_system,
        fit_options={
            "parameter_names": ["pKa1", "pKw"],
            "initial_params": [6.4, 14.0],
            "bounds": ([0.0, 10.0], [14.0, 30.0]),
            "pH_bounds": (-2.0, 24.0),
        },
    )
    pkw_idx = result.parameter_names.index("pKw")
    assert result.success
    assert abs(result.fitted_pka[0] - 7.0) < 1.0e-5
    assert abs(float(result.theta_hat[pkw_idx]) - 18.0) < 1.0e-5


def test_potentiometry_runner_reports_fitted_apparent_pkw():
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[7.0],
        base_charge=-1,
        kw=1.0e-18,
    )
    volumes = np.linspace(0.0, 28.0, 70)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment, true_system, {"pH_bounds": (-2.0, 24.0)})
    result = run_acid_base(
        {
            "data_type": "potentiometry",
            "signal_type": "pH",
            "titrant_volume": volumes,
            "observed_signal": pH,
            "pka_initial": "6.4",
            "analyte_concentration": 1.0e-2,
            "titrant_concentration": 1.0e-2,
            "initial_volume": 10.0,
            "titrant_type": "base",
            "pkw": 14.0,
            "fit_pkw": True,
            "pkw_bounds": [10.0, 30.0],
            "pH_bounds": [-2.0, 24.0],
        },
        progress_cb=lambda _msg: None,
    )
    pkw_rows = [row for row in result["constants"] if row["parameter"] == "pKw_app"]
    assert result["success"] is True
    assert pkw_rows
    assert abs(float(pkw_rows[0]["value"]) - 18.0) < 1.0e-5
    assert "pKw was fitted as an apparent medium parameter." in result["export_data"]["potentiometry_note"]


def test_synthetic_spectroscopy_recovers_pka():
    rng = np.random.default_rng(5678)
    pH = np.linspace(2.0, 8.0, 41)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    signal = fractions @ np.asarray([0.15, 1.25])
    signal = signal + rng.normal(0.0, 1.0e-4, size=signal.shape)
    dataset = SpectroscopicAcidBaseDataset(pH=pH, signal=signal)
    start_system = make_simple_acid_base_system(pka=[4.2])
    result = fit_spectroscopy_acid_base(dataset, start_system, initial_pka=[4.2])
    assert result.success
    assert abs(result.fitted_pka[0] - 5.0) < 0.02


def test_synthetic_nmr_recovers_pka():
    rng = np.random.default_rng(9012)
    pH = np.linspace(2.0, 8.0, 41)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    limiting_shifts = np.asarray(
        [
            [7.10, 8.20, 6.95],
            [8.80, 6.40, 7.55],
        ]
    )
    shifts = fractions @ limiting_shifts
    shifts = shifts + rng.normal(0.0, 1.0e-4, size=shifts.shape)
    dataset = NMRAcidBaseDataset(pH=pH, shifts=shifts, nuclei_labels=["H1", "H2", "H3"])
    start_system = make_simple_acid_base_system(pka=[5.8])
    result = fit_nmr_acid_base(dataset, start_system, initial_pka=[5.8])
    assert result.success
    assert abs(result.fitted_pka[0] - 5.0) < 0.02


def test_xlsx_potentiometry_import_recovers_pka(tmp_path):
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[5.0],
        base_charge=-1,
    )
    volumes = np.linspace(0.0, 18.0, 30)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment, true_system)
    path = tmp_path / "acid_base.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"volume_mL": volumes, "pH": pH}).to_excel(
            writer,
            sheet_name="Pot",
            index=False,
        )

    messages: list[str] = []
    result = run_acid_base(
        {
            "file_path": str(path),
            "sheet_name": "Pot",
            "data_type": "potentiometry",
            "pka_initial": "4.2",
            "analyte_concentration": 1.0e-2,
            "titrant_concentration": 1.0e-2,
            "initial_volume": 10.0,
            "titrant_type": "base",
        },
        progress_cb=messages.append,
    )
    assert result["success"] is True
    assert abs(float(result["constants"][0]["value"]) - 5.0) < 1.0e-6
    assert any("pKw=14.0000" in message and "Kw=1e-14" in message for message in messages)


def test_kw_config_changes_potentiometric_simulated_ph():
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=np.asarray([0.0], dtype=float),
        analyte_concentration=1.0e-12,
        titrant_concentration=0.0,
        titrant_type="base",
    )
    base_cfg = {
        "component_name": "L",
        "pka_initial": "5.0",
        "analyte_concentration": 1.0e-12,
        "base_charge": -1,
    }
    pH_default_kw = simulate_pH_titration(
        experiment,
        _build_system({**base_cfg, "kw": 1.0e-14}),
    )[0]
    pH_high_kw = simulate_pH_titration(
        experiment,
        _build_system({**base_cfg, "kw": 1.0e-10}),
    )[0]
    assert abs(float(pH_default_kw) - float(pH_high_kw)) > 1.0
    assert abs(float(pH_default_kw) - 7.0) < 1.0e-3
    assert abs(float(pH_high_kw) - 5.0) < 1.0e-3


def test_explicit_acid_base_model_config_builds_diprotic_system():
    system = _build_system(
        {
            "kw": 1.0e-14,
            "acid_base_model": {
                "model_type": "polyprotic",
                "components": [
                    {
                        "name": "L",
                        "analytical_concentration": 1.0e-3,
                        "base_charge": -2,
                        "pka": [4.5, 8.9],
                        "use_log_beta": False,
                        "role": "analyte",
                    }
                ],
                "species": [
                    {"name": "L", "component": "L", "h_count": 0, "charge": -2, "log_beta": 0.0},
                    {"name": "HL", "component": "L", "h_count": 1, "charge": -1, "log_beta": 4.5},
                    {"name": "H2L", "component": "L", "h_count": 2, "charge": 0, "log_beta": 13.4},
                ],
            },
        }
    )
    comp = system.components[0]
    assert comp.name == "L"
    assert comp.species[0].charge == -2
    assert comp.species[1].charge == -1
    assert comp.species[2].charge == 0
    assert [sp.log_beta for sp in comp.species[1:]] == pytest.approx([4.5, 13.4])


def test_matrix_generated_monoprotic_model_template():
    model = build_acid_base_template("simple_monoprotic")
    assert model["component_names"] == ["L", "H"]
    assert model["species_names"] == ["L", "HL"]
    assert model["pka"] == pytest.approx([5.20])
    assert model["log_beta"] == pytest.approx([5.20])


def test_polyprotic_template_generates_ladder_species_and_matrix():
    model = build_acid_base_template("polyprotic_acid_base", n_pka=4, pka=[3.0, 5.0, 7.0, 9.0])
    assert model["component_names"] == ["L", "H"]
    assert model["species_names"] == ["L", "HL", "H2L", "H3L", "H4L"]
    assert [row["charge"] for row in model["species"]] == [-4, -3, -2, -1, 0]
    assert model["stoichiometric_matrix"] == [[1, 1, 1, 1, 1], [0, 1, 2, 3, 4]]
    assert model["pka"] == pytest.approx([3.0, 5.0, 7.0, 9.0])


def test_matrix_generated_diprotic_model_template():
    model = build_acid_base_template("diprotic_ligand")
    assert model["component_names"] == ["L", "H"]
    assert model["species_names"] == ["L", "HL", "H2L"]
    assert model["pka"] == pytest.approx([4.50, 8.90])
    assert model["log_beta"] == pytest.approx([4.50, 13.40])


def test_equation_parser_monoprotic_model():
    model = acid_base_model_from_equations("L + H <=> HL ; pKa=5.20")
    assert model["species_names"] == ["L", "HL"]
    assert [row["h_count"] for row in model["species"]] == [0, 1]
    assert model["log_beta"] == pytest.approx([5.20])


def test_equation_parser_diprotic_stepwise_model():
    model = acid_base_model_from_equations(
        "L + H <=> HL ; pKa=4.50\n"
        "HL + H <=> H2L ; pKa=8.90"
    )
    assert model["pka"] == pytest.approx([4.50, 8.90])
    assert model["log_beta"] == pytest.approx([4.50, 13.40])


def test_equation_parser_diprotic_cumulative_model():
    model = acid_base_model_from_equations(
        "L + H <=> HL ; logB=4.50\n"
        "L + 2H <=> H2L ; logB=13.40"
    )
    assert model["log_beta"] == pytest.approx([4.50, 13.40])
    assert model["pka"] == pytest.approx([4.50, 8.90])


def test_diprotic_species_fractions_sum_to_one():
    pH = np.linspace(0.0, 14.0, 71)
    fractions = distribution_fractions_from_pH(pH, [4.5, 13.4])
    assert fractions.shape == (71, 3)
    assert np.allclose(fractions.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_imposed_ph_spectroscopy_fit_ignores_kw(tmp_path):
    pH = np.linspace(2.0, 8.0, 31)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    signal = fractions @ np.asarray([0.1, 1.2])
    path = tmp_path / "spectro.csv"
    pd.DataFrame({"pH": pH, "signal": signal}).to_csv(path, index=False)
    base_cfg = {
        "file_path": str(path),
        "data_type": "spectroscopy",
        "pka_initial": "4.4",
        "analyte_concentration": 1.0e-3,
        "base_charge": -1,
    }
    result_default_kw = run_acid_base({**base_cfg, "kw": 1.0e-14}, progress_cb=lambda _msg: None)
    result_high_kw = run_acid_base({**base_cfg, "kw": 1.0e-10}, progress_cb=lambda _msg: None)
    assert result_default_kw["success"] is True
    assert result_high_kw["success"] is True
    assert float(result_default_kw["constants"][0]["value"]) == pytest.approx(
        float(result_high_kw["constants"][0]["value"]),
        abs=1e-8,
    )


def test_gui_config_generates_diprotic_species_and_pkw(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "pot.csv"
    pd.DataFrame({"volume_mL": [0.0, 0.1], "pH": [6.5, 6.6]}).to_csv(path, index=False)

    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    tab._load_template("diprotic_ligand")
    tab._refresh_parameter_table()
    tab._set_parameter_value("pKw", "13.78")

    cfg = tab._collect_config()
    species = cfg["acid_base_model"]["species"]
    assert [row["name"] for row in species[:3]] == ["L", "HL", "H2L"]
    assert [row["charge"] for row in species[:3]] == [-2, -1, 0]
    assert [row["log_beta"] for row in species[:3]] == pytest.approx([0.0, 4.5, 13.4])
    assert cfg["kw"] == pytest.approx(10.0**-13.78)
    assert cfg["fit_pkw"] is False


def test_gui_config_default_pkw_is_14(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "pot.csv"
    pd.DataFrame({"volume_mL": [0.0], "pH": [7.0]}).to_csv(path, index=False)
    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    cfg = tab._collect_config()
    assert cfg["pkw"] == pytest.approx(14.0)
    assert cfg["pkw_bounds"] == pytest.approx([0.0, 30.0])
    assert cfg["fit_pkw"] is False
    assert cfg["kw"] == pytest.approx(1.0e-14)


def test_gui_config_can_mark_pkw_as_fitted(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "pot.csv"
    pd.DataFrame({"volume_mL": [0.0, 0.1], "pH": [7.0, 7.2]}).to_csv(path, index=False)
    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    tab.model_advanced_group.setChecked(True)
    tab.spin_pkw.setValue(18.0)
    tab.spin_pkw_min.setValue(10.0)
    tab.spin_pkw_max.setValue(30.0)
    tab.chk_fit_pkw.setChecked(True)

    cfg = tab._collect_config()
    assert cfg["pkw"] == pytest.approx(18.0)
    assert cfg["pkw_bounds"] == pytest.approx([10.0, 30.0])
    assert cfg["fit_pkw"] is True


def test_gui_detects_ul_volume_column(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "pot_ul.csv"
    pd.DataFrame({"vol µL": [0.0, 50.0, 100.0], "pH": [7.0, 7.1, 7.2]}).to_csv(path, index=False)
    tab = AcidBaseTab()
    tab._set_file_path(str(path))

    assert tab.combo_volume_column.currentData() == "vol µL"
    assert tab.combo_volume_unit.currentData() == "µL"
    payload = tab._build_potentiometry_import_payload()
    assert payload["titrant_volume"] == pytest.approx([0.0, 0.05, 0.1])


def test_gui_detects_emf_signal_column(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "pot_emf.csv"
    pd.DataFrame({"Vadd": [0.0, 0.1, 0.2], "E_mV": [10.0, 11.0, 12.0]}).to_csv(path, index=False)
    tab = AcidBaseTab()
    tab._set_file_path(str(path))

    assert tab.combo_volume_column.currentData() == "Vadd"
    assert tab.combo_signal_column.currentData() == "E_mV"
    assert tab.combo_signal_type.currentData() == "mV"


def test_basic_mode_does_not_populate_advanced_tables_until_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    tab = AcidBaseTab()
    before = tab._advanced_table_rebuild_count
    tab._generate_basic_species(3)

    assert tab.chk_advanced_mode.isChecked() is False
    assert tab._advanced_table_rebuild_count == before


def test_basic_fit_checkbox_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    tab = AcidBaseTab()
    tab._refresh_parameter_table()
    tab._set_parameter_fit("pKa1", True)
    assert tab._parameter_fixed("pKa1") is False
    tab._set_parameter_fit("pKa1", False)
    assert tab._parameter_fixed("pKa1") is True


def test_basic_species_generates_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    tab = AcidBaseTab()
    tab._generate_basic_species(2)
    model = tab._basic_model_from_species_table()

    assert model["stoichiometric_matrix"] == [[1, 1, 1], [0, 1, 2]]


def test_signal_type_ph_hides_electrode_options(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    tab = AcidBaseTab()
    tab.combo_signal_type.setCurrentIndex(tab.combo_signal_type.findData("pH"))
    tab._on_signal_type_changed()

    assert tab.chk_ideal_nernst.isVisible() is False
    assert tab.chk_fit_electrode_basic.isVisible() is False
    assert tab.electrode_group.isVisible() is False


def test_signal_type_emf_shows_electrode_options(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    tab = AcidBaseTab()
    tab.combo_signal_type.setCurrentIndex(tab.combo_signal_type.findData("mV"))
    tab._on_signal_type_changed()

    assert tab.chk_ideal_nernst.isHidden() is False
    assert tab.chk_fit_electrode_basic.isHidden() is False


def test_mixed_solvent_pkw_suggestion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "high_ph.csv"
    pd.DataFrame({"volume_mL": [0.0, 0.1, 0.2], "pH*": [13.8, 14.3, 15.1]}).to_csv(path, index=False)
    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    tab._suggest_setup_from_data()

    assert tab.combo_medium.currentData() == "mixed"
    assert tab.spin_pkw.value() == pytest.approx(18.0)
    assert "apparent pKw" in tab.lbl_pot_validation.text()


def test_pkw_fitting_still_reaches_core(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "pot.csv"
    pd.DataFrame({"volume_mL": [0.0, 0.1, 0.2], "pH": [7.0, 7.2, 7.4]}).to_csv(path, index=False)
    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    tab.combo_medium.setCurrentIndex(tab.combo_medium.findData("mixed"))
    tab.chk_fit_pkw.setChecked(True)
    cfg = tab._collect_config()

    assert cfg["fit_pkw"] is True
    assert cfg["pkw"] == pytest.approx(18.0)
    assert cfg["pkw_bounds"] == pytest.approx([10.0, 30.0])


def test_advanced_mode_exposes_full_model(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    tab = AcidBaseTab()
    tab.chk_advanced_mode.setChecked(True)

    assert tab.components_table.isHidden() is False
    assert tab.species_table.isHidden() is False
    assert tab.stoich_table.isHidden() is False
    assert tab.titration_group.isHidden() is False
    assert tab.combo_titrant_type.findData("custom") >= 0


def test_gui_equation_editor_generates_canonical_model(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    path = tmp_path / "spec.csv"
    pd.DataFrame({"pH": [4.0, 5.0, 6.0], "signal": [0.2, 0.5, 0.8]}).to_csv(path, index=False)

    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    tab.radio_model_equations.setChecked(True)
    tab.equation_editor.setPlainText("L + H <=> HL ; pKa=5.20")
    tab._apply_equations_model()

    cfg = tab._collect_config()
    model = canonicalize_acid_base_model(cfg["acid_base_model"])
    assert model["definition_mode"] == "equations"
    assert model["species_names"] == ["L", "HL"]
    assert model["pka"] == pytest.approx([5.20])


def test_old_simple_config_still_builds_system():
    system = _build_system(
        {
            "component_name": "Lig",
            "pka_initial": "4.5, 8.9",
            "analyte_concentration": 1.0e-3,
            "base_charge": -2,
            "kw": 1.0e-14,
        }
    )
    assert system.components[0].name == "Lig"
    assert [sp.charge for sp in system.components[0].species] == [-2, -1, 0]


def test_potentiometry_runner_uses_processed_arrays_and_mask():
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[5.0],
        base_charge=-1,
    )
    volumes_mL = np.linspace(0.0, 18.0, 30)
    template = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes_mL,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(template, true_system)
    include_mask = np.ones(volumes_mL.shape, dtype=bool)
    include_mask[[3, 7, 18]] = False

    result = run_acid_base(
        {
            "file_path": __file__,
            "data_type": "potentiometry",
            "pka_initial": "4.4",
            "analyte_concentration": 1.0e-2,
            "titrant_concentration": 1.0e-2,
            "initial_volume": 10.0,
            "titrant_type": "base",
            "titrant_volume": volumes_mL.tolist(),
            "observed_signal": pH.tolist(),
            "included_mask": include_mask.tolist(),
            "signal_type": "pH",
            "volume_unit": "mL",
        },
        progress_cb=lambda _msg: None,
    )

    assert result["success"] is True
    assert abs(float(result["constants"][0]["value"]) - 5.0) < 1.0e-6
    exported = result["export_data"]["experimental_vs_calculated"]
    assert len(exported["volume_mL"]) == int(include_mask.sum())
    assert exported["signal_type"] == ["pH"] * int(include_mask.sum())
    assert exported["source_volume_unit"] == ["mL"] * int(include_mask.sum())


def test_gui_potentiometry_import_payload_converts_units_and_preserves_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    qt_core = pytest.importorskip("PySide6.QtCore")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app

    path = tmp_path / "pot_payload.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "added_uL": [0.0, 250.0, 500.0, 900.0],
                "glass_electrode": [6.20, 6.55, 6.90, 7.15],
                "note": ["a", "b", "c", "d"],
            }
        ).to_excel(writer, sheet_name="Titration", index=False)

    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    tab.combo_data_type.setCurrentIndex(tab.combo_data_type.findData("potentiometry"))
    tab.combo_sheet.setCurrentIndex(tab.combo_sheet.findData("Titration"))
    tab.combo_volume_column.setCurrentIndex(tab.combo_volume_column.findData("added_uL"))
    tab.combo_signal_column.setCurrentIndex(tab.combo_signal_column.findData("glass_electrode"))
    tab.combo_volume_unit.setCurrentIndex(tab.combo_volume_unit.findData("µL"))
    tab.combo_signal_type.setCurrentIndex(tab.combo_signal_type.findData("pH"))

    item = tab.table_pot_preview.item(1, 0)
    assert item is not None
    item.setCheckState(qt_core.Qt.CheckState.Unchecked)
    payload = tab._build_potentiometry_import_payload()

    assert payload["volume_column"] == "added_uL"
    assert payload["signal_column"] == "glass_electrode"
    assert payload["volume_unit"] == "µL"
    assert payload["signal_type"] == "pH"
    assert payload["included_mask"] == [True, False, True, True]
    assert payload["titrant_volume"] == pytest.approx([0.0, 0.25, 0.5, 0.9])
    assert payload["observed_signal"] == pytest.approx([6.2, 6.55, 6.9, 7.15])


def test_gui_potentiometry_validation_warns_for_operational_ph(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app

    path = tmp_path / "pot_warn.csv"
    pd.DataFrame({"volume_mL": [0.0, 0.2, 0.4], "pH": [13.9, 14.2, 14.5]}).to_csv(path, index=False)

    tab = AcidBaseTab()
    tab._set_file_path(str(path))
    payload = tab._build_potentiometry_import_payload()

    assert payload["signal_type"] == "pH"
    assert any("operational pH" in warning for warning in payload["potentiometry_warnings"])
    assert "Initial volume is zero." in payload["potentiometry_warnings"]
