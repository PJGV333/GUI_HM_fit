"""Tests for Kinetics XLSX/GUI reporting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.gui.reporting import (
    NOT_AVAILABLE_MESSAGE,
    build_absorptivity_df,
    build_absorbance_df,
    build_fit_df,
    build_gui_parameter_rows,
    build_model_df,
    build_parameters_df,
    build_raw_df,
    build_summary_df,
    write_kinetics_results_xlsx,
)


def _build_result(*, include_fit_profiles: bool = True) -> dict:
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    D = np.array(
        [
            [0.20, 0.10],
            [0.30, 0.20],
            [0.40, 0.30],
        ],
        dtype=float,
    )
    D_hat = D * 0.95 if include_fit_profiles else None
    residuals = (D - D_hat) if D_hat is not None else None
    fit_C = (
        np.array(
            [
                [1.00, 0.00],
                [0.65, 0.35],
                [0.30, 0.70],
            ],
            dtype=float,
        )
        if include_fit_profiles
        else None
    )
    fit_A = (
        np.array(
            [
                [1.20, 0.80],
                [0.20, 1.10],
            ],
            dtype=float,
        )
        if include_fit_profiles
        else None
    )
    ds = KineticsFitDataset(
        t=t,
        y0={"A": 1.0, "B": 0.0},
        fixed_conc={},
        D=D,
        x=np.array([450.0, 500.0], dtype=float),
        channel_labels=["450", "500"],
        technique="spec_full",
        time_unit="s",
        x_unit="nm",
        name="dataset-1",
        fit_C=fit_C,
        fit_A=fit_A,
        fit_D_hat=D_hat,
        fit_residuals=residuals,
    )
    return {
        "success": True,
        "completed_at": "2026-02-28T12:00:00",
        "optimizer_method": "trf",
        "optimizer_requested": "least_squares",
        "optimizer_backend": "scipy",
        "params": {"k1": 1.23, "k2": 0.45},
        "param_names": ["k1", "k2"],
        "log_params": [],
        "dynamic_species": ["A", "B"],
        "parameter_errors": {
            "parameter": ["k1", "k2"],
            "estimate": [1.23, 0.45],
            "se": [0.012, 0.020],
            "perc_err": [0.98, 4.44],
            "ci_low": [1.20, 0.41],
            "ci_high": [1.26, 0.49],
            "fixed": [False, False],
            "scale": ["linear", "linear"],
        },
        "statistics": {"RMS": 0.005},
        "ssq": 0.0012,
        "cost": 0.0006,
        "nfev": 42,
        "njev": 21,
        "nfev_total": 42,
        "njev_total": 21,
        "multi_start_runs": 1,
        "condition_number": 3456.0,
        "diagnostics": {
            "warnings": [],
            "stability_status": "excellent",
            "diag_summary": "Stable",
            "condition_number": 3456.0,
            "method_requested": "least_squares",
            "method_used": "trf",
            "backend": "scipy",
            "nnls": False,
            "weighted_fit": False,
            "optimizer_success": True,
        },
        "datasets": [ds],
    }


def test_write_kinetics_results_xlsx_contains_expected_sheets(tmp_path: Path) -> None:
    result = _build_result(include_fit_profiles=True)
    out_path = tmp_path / "kinetics_report.xlsx"
    write_kinetics_results_xlsx(str(out_path), result=result, config={"method": "least_squares"})

    xls = pd.ExcelFile(out_path)
    expected = {
        "Summary",
        "Parameters",
        "Model",
        "RawData",
        "FitData",
        "Concentrations_Ct",
        "Absorptivity_Profiles",
        "Absorbance_Profiles",
        "Diagnostics",
    }
    assert expected.issubset(set(xls.sheet_names))

    summary_df = pd.read_excel(out_path, sheet_name="Summary")
    params_df = pd.read_excel(out_path, sheet_name="Parameters")
    raw_df = pd.read_excel(out_path, sheet_name="RawData")
    fit_df = pd.read_excel(out_path, sheet_name="FitData")
    conc_df = pd.read_excel(out_path, sheet_name="Concentrations_Ct")
    diag_df = pd.read_excel(out_path, sheet_name="Diagnostics")

    assert {"metric", "value"}.issubset(summary_df.columns)
    assert {"parameter", "estimate", "SE", "%err", "CI_low", "CI_high", "fixed", "scale"}.issubset(
        params_df.columns
    )
    assert {"time", "obs_450", "obs_500"}.issubset(raw_df.columns)
    assert {"time", "obs_450", "fit_450", "res_450", "obs_500", "fit_500", "res_500"}.issubset(fit_df.columns)
    assert {"time", "A", "B"}.issubset(conc_df.columns)
    assert "format" not in raw_df.columns
    assert "format" not in fit_df.columns
    assert "format" not in conc_df.columns
    assert {"type", "key", "value"}.issubset(diag_df.columns)


def test_parameters_export_includes_se_and_percent_error() -> None:
    result = _build_result(include_fit_profiles=True)
    df = build_parameters_df(result)
    row = df[df["parameter"] == "k1"].iloc[0]
    assert np.isclose(float(row["SE"]), 0.012)
    assert np.isclose(float(row["%err"]), 0.98)


def test_not_available_sheet_message_when_section_missing() -> None:
    result = _build_result(include_fit_profiles=False)
    df = build_absorptivity_df(result)
    assert list(df.columns) == ["message"]
    assert str(df.iloc[0, 0]) == NOT_AVAILABLE_MESSAGE


def test_gui_summary_rows_allow_missing_ci() -> None:
    result = _build_result(include_fit_profiles=True)
    err = dict(result["parameter_errors"])
    err.pop("ci_low", None)
    err.pop("ci_high", None)
    result["parameter_errors"] = err

    rows = build_gui_parameter_rows(result)
    assert len(rows) == 2
    assert np.isclose(float(rows[0]["Estimate"]), 1.23)
    assert np.isnan(float(rows[0]["CI 2.5"]))
    assert np.isnan(float(rows[0]["CI 97.5"]))


def test_summary_uses_run_timestamp_when_available() -> None:
    result = _build_result(include_fit_profiles=True)
    df = build_summary_df(result)
    ts = df.loc[df["metric"] == "timestamp", "value"].iloc[0]
    assert str(ts) == "2026-02-28T12:00:00"


def test_model_df_includes_error_config_flags() -> None:
    result = _build_result(include_fit_profiles=True)
    cfg = {"auto_parameter_errors": False, "error_condition_threshold": 1.0e10}
    df = build_model_df(result, config=cfg)
    assert "auto_parameter_errors" in set(df["key"].astype(str))
    assert "error_condition_threshold" in set(df["key"].astype(str))


def test_wide_profiles_and_absorbance_layout() -> None:
    result = _build_result(include_fit_profiles=True)
    raw_df = build_raw_df(result)
    fit_df = build_fit_df(result)
    eps_df = build_absorptivity_df(result)
    abs_df = build_absorbance_df(result)

    assert list(raw_df.columns) == ["time", "obs_450", "obs_500"]
    assert list(fit_df.columns) == [
        "time",
        "obs_450",
        "fit_450",
        "res_450",
        "obs_500",
        "fit_500",
        "res_500",
    ]
    assert list(eps_df.columns) == ["wavelength", "A", "B"]
    assert list(abs_df.columns) == [
        "time",
        "D_450",
        "D_hat_450",
        "residual_450",
        "D_500",
        "D_hat_500",
        "residual_500",
    ]


def test_nmr_like_kinetics_wide_layout() -> None:
    t = np.array([0.0, 10.0], dtype=float)
    D = np.array([[1.0, 0.5], [0.8, 0.7]], dtype=float)
    D_hat = np.array([[0.95, 0.55], [0.78, 0.72]], dtype=float)
    ds = KineticsFitDataset(
        t=t,
        y0={"A": 1.0},
        fixed_conc={},
        D=D,
        channel_labels=["H1", "H2"],
        technique="nmr_integrals",
        time_unit="s",
        name="nmr_dataset",
        fit_D_hat=D_hat,
        fit_residuals=D - D_hat,
    )
    result = {
        "param_names": ["k1"],
        "params": {"k1": 1.0},
        "dynamic_species": ["A"],
        "datasets": [ds],
    }

    raw_df = build_raw_df(result)
    fit_df = build_fit_df(result)
    abs_df = build_absorbance_df(result)

    assert list(raw_df.columns) == ["time", "obs_H1", "obs_H2"]
    assert list(fit_df.columns) == ["time", "obs_H1", "fit_H1", "res_H1", "obs_H2", "fit_H2", "res_H2"]
    assert list(abs_df.columns) == [
        "time",
        "D_H1",
        "D_hat_H1",
        "residual_H1",
        "D_H2",
        "D_hat_H2",
        "residual_H2",
    ]
