from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from hmfit_core.acid_base import distribution_fractions_from_pH, make_simple_acid_base_system
from hmfit_core.acid_base_errors import (
    compute_errors_acid_base_from_context,
    propagate_log_beta_to_pka,
    propagate_pka_to_log_beta,
)
from hmfit_core.exports import write_results_xlsx
from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
from hmfit_core.run_acid_base import run_acid_base


def test_pka_log_beta_covariance_propagation():
    pka = np.asarray([4.5, 8.9], dtype=float)
    cov_pka = np.asarray([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    log_beta, cov_log_beta, transform = propagate_pka_to_log_beta(pka, cov_pka)

    expected_transform = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=float)
    assert np.allclose(transform, expected_transform, atol=1e-12, rtol=0.0)
    assert np.allclose(log_beta, [4.5, 13.4], atol=1e-12, rtol=0.0)
    assert np.allclose(cov_log_beta, expected_transform @ cov_pka @ expected_transform.T, atol=1e-12, rtol=0.0)


def test_log_beta_pka_covariance_propagation():
    log_beta = np.asarray([4.5, 13.4], dtype=float)
    cov_log_beta = np.asarray([[0.04, 0.03], [0.03, 0.13]], dtype=float)
    pka, cov_pka, transform = propagate_log_beta_to_pka(log_beta, cov_log_beta)

    expected_transform = np.asarray([[1.0, 0.0], [-1.0, 1.0]], dtype=float)
    assert np.allclose(transform, expected_transform, atol=1e-12, rtol=0.0)
    assert np.allclose(pka, [4.5, 8.9], atol=1e-12, rtol=0.0)
    assert np.allclose(cov_pka, expected_transform @ cov_log_beta @ expected_transform.T, atol=1e-12, rtol=0.0)


def test_analytical_covariance_spectroscopy_pka(tmp_path):
    rng = np.random.default_rng(321)
    pH = np.linspace(2.0, 8.0, 41)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    signal = fractions @ np.asarray([0.2, 1.35])
    signal = signal + rng.normal(0.0, 1.0e-4, size=signal.shape)
    path = tmp_path / "acid_base_spec.csv"
    pd.DataFrame({"pH": pH, "signal": signal}).to_csv(path, index=False)

    result = run_acid_base(
        {
            "file_path": str(path),
            "data_type": "spectroscopy",
            "pka_initial": "4.3",
            "analyte_concentration": 1.0e-3,
            "base_charge": -1,
            "baseline": False,
        },
        progress_cb=lambda _msg: None,
    )
    assert result["success"] is True
    ctx = result["errors_context"]
    err = compute_errors_acid_base_from_context(ctx, {"method": "analytic", "include_16_84": True})

    params = err["export_frames"]["Parameters"]
    assert not params.empty
    assert float(params.loc[0, "SE"]) > 0.0
    assert np.asarray(err["corr"], dtype=float).shape == (1, 1)


def test_potentiometry_analytical_errors(tmp_path):
    true_system = make_simple_acid_base_system(
        analytical_concentration=1.0e-2,
        pka=[5.0],
        base_charge=-1,
    )
    volumes = np.linspace(0.0, 18.0, 35)
    experiment = PotentiometryExperiment(
        initial_volume=10.0,
        titrant_volumes=volumes,
        analyte_concentration=1.0e-2,
        titrant_concentration=1.0e-2,
        titrant_type="base",
    )
    pH = simulate_pH_titration(experiment, true_system)
    path = tmp_path / "acid_base_pot.csv"
    pd.DataFrame({"volume_mL": volumes, "pH": pH}).to_csv(path, index=False)

    result = run_acid_base(
        {
            "file_path": str(path),
            "data_type": "potentiometry",
            "pka_initial": "4.2",
            "analyte_concentration": 1.0e-2,
            "titrant_concentration": 1.0e-2,
            "initial_volume": 10.0,
            "titrant_type": "base",
        },
        progress_cb=lambda _msg: None,
    )
    assert result["success"] is True
    err = compute_errors_acid_base_from_context(result["errors_context"], {"method": "analytic"})

    params = err["export_frames"]["Parameters"]
    assert not params.empty
    assert np.isfinite(float(params.loc[0, "SE"]))
    assert np.asarray(err["corr"], dtype=float).shape == (1, 1)


def test_nmr_analytical_errors(tmp_path):
    rng = np.random.default_rng(987)
    pH = np.linspace(2.0, 8.0, 41)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    limiting_shifts = np.asarray(
        [
            [7.10, 8.20, 6.95],
            [8.85, 6.35, 7.60],
        ]
    )
    shifts = fractions @ limiting_shifts
    shifts = shifts + rng.normal(0.0, 1.0e-4, size=shifts.shape)
    path = tmp_path / "acid_base_nmr.csv"
    pd.DataFrame({"pH": pH, "H1": shifts[:, 0], "H2": shifts[:, 1], "H3": shifts[:, 2]}).to_csv(path, index=False)

    result = run_acid_base(
        {
            "file_path": str(path),
            "data_type": "nmr",
            "pka_initial": "5.7",
            "analyte_concentration": 1.0e-3,
            "base_charge": -1,
        },
        progress_cb=lambda _msg: None,
    )
    assert result["success"] is True
    err = compute_errors_acid_base_from_context(result["errors_context"], {"method": "analytic"})

    params = err["export_frames"]["Parameters"]
    assert not params.empty
    assert float(params.loc[0, "SE"]) > 0.0
    assert np.asarray(err["corr"], dtype=float).shape == (1, 1)


def test_gui_result_context_enables_errors_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.acid_base_tab import AcidBaseTab

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app

    pH = np.linspace(2.0, 8.0, 31)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    signal = fractions @ np.asarray([0.15, 1.2])
    path = tmp_path / "gui_acid_base_spec.csv"
    pd.DataFrame({"pH": pH, "signal": signal}).to_csv(path, index=False)

    result = run_acid_base(
        {
            "file_path": str(path),
            "data_type": "spectroscopy",
            "pka_initial": "4.4",
            "analyte_concentration": 1.0e-3,
            "base_charge": -1,
        },
        progress_cb=lambda _msg: None,
    )
    assert result["success"] is True

    tab = AcidBaseTab()
    tab._last_result = result
    tab._update_errors_context_from_result(result, auto_compute=False)

    assert tab.errors_panel is not None
    assert tab.errors_panel.has_errors_context() is True
    output = tab.errors_panel.compute_now()

    assert isinstance(output, dict)
    assert tab.errors_panel.table_error_results.rowCount() >= 1
    assert tab.errors_panel.table_error_corr.rowCount() >= 1


def test_acid_base_export_includes_error_sheets(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    pH = np.linspace(2.0, 8.0, 31)
    fractions = distribution_fractions_from_pH(pH, [5.0])
    signal = fractions @ np.asarray([0.12, 1.10])
    data_path = tmp_path / "export_spec.csv"
    out_path = tmp_path / "acid_base_results.xlsx"
    pd.DataFrame({"pH": pH, "signal": signal}).to_csv(data_path, index=False)

    result = run_acid_base(
        {
            "file_path": str(data_path),
            "data_type": "spectroscopy",
            "pka_initial": "4.4",
            "analyte_concentration": 1.0e-3,
            "base_charge": -1,
        },
        progress_cb=lambda _msg: None,
    )
    assert result["success"] is True

    write_results_xlsx(
        out_path,
        constants=result.get("constants") or [],
        statistics=result.get("statistics") or {},
        results_text=str(result.get("results_text") or ""),
        export_data=result.get("export_data") or {},
    )

    workbook = openpyxl.load_workbook(out_path, read_only=True)
    sheet_names = set(workbook.sheetnames)
    assert "Parameters" in sheet_names
    assert "pKa" in sheet_names
    assert "log_beta" in sheet_names
    assert "Derived_constants" in sheet_names
    assert "Error_diagnostics" in sheet_names
