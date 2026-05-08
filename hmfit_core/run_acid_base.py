# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import base64
import io

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from hmfit_core.acid_base import (
    AcidBaseFitResult,
    NMRAcidBaseDataset,
    SpectroscopicAcidBaseDataset,
    clone_system_with_pka,
    fit_nmr_acid_base,
    fit_spectroscopy_acid_base,
    make_simple_acid_base_system,
    predict_nmr_acid_base,
    predict_spectroscopy_acid_base,
    simulate_species_vs_pH,
)
from hmfit_core.potentiometry import (
    PotentiometryExperiment,
    electrode_pH_from_emf,
    fit_potentiometry,
    observed_pH,
    simulate_pH_titration,
    simulate_species_vs_volume,
)

ProgressCallback = Callable[[str], None] | None


def _normalize_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dict-like mapping or a dataclass instance")


def _parse_float_list(raw: Any, *, default: Sequence[float] | None = None) -> list[float]:
    if raw is None:
        return list(default or [])
    if isinstance(raw, np.ndarray):
        return [float(v) for v in raw.reshape(-1)]
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    text = str(raw).replace(";", ",").strip()
    if not text:
        return list(default or [])
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _csv_column(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lookup = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _read_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError("Acid-base v1 imports CSV/TXT files.")
    return pd.read_csv(path)


def _build_system(cfg: Mapping[str, Any]):
    pka = _parse_float_list(cfg.get("pka_initial") or cfg.get("initial_pka"), default=[5.0])
    concentration = float(cfg.get("analyte_concentration", cfg.get("concentration", 1.0e-3)) or 1.0e-3)
    base_charge = int(cfg.get("base_charge", -1) or -1)
    return make_simple_acid_base_system(
        name=str(cfg.get("component_name") or "L"),
        analytical_concentration=concentration,
        pka=pka,
        base_charge=base_charge,
        temperature=float(cfg.get("temperature", 298.15) or 298.15),
        kw=float(cfg.get("kw", 1e-14) or 1e-14),
    )


def _figure_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_series(
    x: Sequence[float],
    y: Any,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: Sequence[str] | None = None,
    marker: str = "-",
) -> str:
    fig = Figure(figsize=(7.0, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float)
    labels = list(labels or [])
    if y_arr.ndim == 1:
        ax.plot(x_arr, y_arr, marker, label=labels[0] if labels else None)
    else:
        if y_arr.shape[0] != x_arr.size and y_arr.shape[1] == x_arr.size:
            y_arr = y_arr.T
        n = y_arr.shape[1]
        for idx in range(n):
            ax.plot(x_arr, y_arr[:, idx], marker, label=labels[idx] if idx < len(labels) else None)
    if labels:
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _plot_overlay(
    x: Sequence[float],
    observed: Any,
    calculated: Any,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    max_columns: int = 10,
) -> str:
    fig = Figure(figsize=(7.0, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    obs = np.asarray(observed, dtype=float)
    calc = np.asarray(calculated, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if calc.ndim == 1:
        calc = calc.reshape(-1, 1)
    n = min(obs.shape[1], calc.shape[1], int(max_columns))
    for idx in range(n):
        show_label = idx == 0
        ax.plot(x_arr, obs[:, idx], "o", ms=4, label="observed" if show_label else None)
        ax.plot(x_arr, calc[:, idx], "-", label="calculated" if show_label else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _fit_result_tables(result: AcidBaseFitResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pka_table = [
        {"parameter": f"pKa{i+1}", "value": float(value)}
        for i, value in enumerate(result.fitted_pka)
    ]
    log_beta_table = [
        {"parameter": f"log_beta{i+1}", "value": float(value)}
        for i, value in enumerate(result.fitted_log_beta)
    ]
    return pka_table, log_beta_table


def _base_result_payload(
    result: AcidBaseFitResult,
    *,
    analysis_kind: str,
    graphs: dict[str, str],
    export_data: dict[str, Any],
) -> dict[str, Any]:
    pka_table, log_beta_table = _fit_result_tables(result)
    constants = pka_table + log_beta_table
    stats = {
        "chi_square": float(result.chi_square),
        "reduced_chi_square": result.reduced_chi_square,
        "aic": result.aic,
        "bic": result.bic,
        "analysis_kind": analysis_kind,
    }
    lines = ["Acid-base fit results", ""]
    for row in pka_table:
        lines.append(f"{row['parameter']}: {row['value']:.8g}")
    for row in log_beta_table:
        lines.append(f"{row['parameter']}: {row['value']:.8g}")
    lines.append("")
    lines.append(f"chi_square: {result.chi_square:.8g}")
    if result.reduced_chi_square is not None:
        lines.append(f"reduced_chi_square: {result.reduced_chi_square:.8g}")
    export_data = dict(export_data)
    export_data["analysis_type"] = "acid_base"
    export_data["analysis_kind"] = analysis_kind
    export_data["pka_table"] = pka_table
    export_data["log_beta_table"] = log_beta_table
    export_data["parameter_table"] = result.parameter_table.to_dict(orient="records")
    export_data["covariance_matrix"] = (
        None if result.covariance_matrix is None else np.asarray(result.covariance_matrix).tolist()
    )
    export_data["correlation_matrix"] = (
        None if result.correlation_matrix is None else np.asarray(result.correlation_matrix).tolist()
    )
    export_data["residuals"] = np.asarray(result.residuals, dtype=float).tolist()
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "constants": constants,
        "statistics": stats,
        "results_text": "\n".join(lines),
        "graphs": graphs,
        "legacy_graphs": graphs,
        "availablePlots": [
            {"id": key, "title": title, "kind": "image"}
            for key, title in {
                "fit": "Observed/calculated fit",
                "residuals": "Residuals",
                "species_pH": "Species vs pH",
                "species_volume": "Species vs volume",
            }.items()
            if graphs.get(key)
        ],
        "export_data": export_data,
    }


def _run_potentiometry(cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    df = _read_csv(cfg["file_path"])
    volume_col = _csv_column(df, ["volume_mL", "volume", "v", "v_ml"])
    if volume_col is None:
        raise ValueError("Potentiometry CSV needs a volume_mL column.")
    ph_col = _csv_column(df, ["pH", "ph"])
    emf_col = _csv_column(df, ["E_mV", "emf", "emf_mV", "E"])
    if ph_col is None and emf_col is None:
        raise ValueError("Potentiometry CSV needs pH or E_mV.")

    volumes = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=float)
    measured_pH = None
    measured_emf = None
    if ph_col is not None:
        measured_pH = pd.to_numeric(df[ph_col], errors="coerce").to_numpy(dtype=float)
    if emf_col is not None:
        measured_emf = pd.to_numeric(df[emf_col], errors="coerce").to_numpy(dtype=float)

    system = _build_system(cfg)
    initial_pka = _parse_float_list(cfg.get("pka_initial") or cfg.get("initial_pka"), default=[5.0])
    experiment = PotentiometryExperiment(
        initial_volume=float(cfg.get("initial_volume", 10.0) or 10.0),
        titrant_volumes=volumes,
        measured_pH=measured_pH,
        measured_emf=measured_emf,
        analyte_concentration=float(cfg.get("analyte_concentration", 1.0e-3) or 1.0e-3),
        titrant_concentration=float(cfg.get("titrant_concentration", 1.0e-3) or 1.0e-3),
        titrant_type=str(cfg.get("titrant_type") or "base"),
        temperature=float(cfg.get("temperature", 298.15) or 298.15),
        electrode_e0=(None if cfg.get("electrode_e0") in (None, "") else float(cfg.get("electrode_e0"))),
        electrode_slope=(None if cfg.get("electrode_slope") in (None, "") else float(cfg.get("electrode_slope"))),
    )
    fit_options: dict[str, Any] = {
        "sigma_pH": float(cfg.get("sigma_pH", 1.0) or 1.0),
        "sigma_E": float(cfg.get("sigma_E", cfg.get("sigma_emf", 1.0)) or 1.0),
        "fit_signal": str(cfg.get("fit_signal") or "auto"),
    }
    if bool(cfg.get("fit_electrode", False)) and measured_emf is not None:
        fit_options["parameter_names"] = ["pKa1", "electrode_e0", "electrode_slope"]
        fit_options["initial_params"] = [
            float(initial_pka[0]),
            0.0 if experiment.electrode_e0 is None else float(experiment.electrode_e0),
            -59.16 if experiment.electrode_slope is None else float(experiment.electrode_slope),
        ]
        fit_options["bounds"] = ([-5.0, -1.0e5, -120.0], [25.0, 1.0e5, 120.0])
        fit_options["fit_signal"] = "emf"
    log("Fitting potentiometric acid-base data...")
    result = fit_potentiometry(experiment, system, initial_pka=initial_pka, fit_options=fit_options)
    fitted_system = clone_system_with_pka(system, result.fitted_pka)
    calc_pH = simulate_pH_titration(experiment, fitted_system, fit_options)
    obs_ph = observed_pH(experiment)
    obs_for_plot = measured_pH if measured_pH is not None else obs_ph
    residuals = np.asarray(obs_for_plot, dtype=float).reshape(-1) - calc_pH
    dist_ph = simulate_species_vs_pH(fitted_system)
    dist_vol = simulate_species_vs_volume(experiment, fitted_system, fit_options)
    graphs = {
        "fit": _plot_overlay(volumes, obs_for_plot, calc_pH, title="pH vs titrant volume", xlabel="volume_mL", ylabel="pH"),
        "residuals": _plot_series(volumes, residuals, title="Residuals", xlabel="volume_mL", ylabel="observed - calculated", marker="o"),
        "species_pH": _plot_series(dist_ph.x, dist_ph.fractions, title="Species distribution vs pH", xlabel="pH", ylabel="fraction", labels=dist_ph.species_names),
        "species_volume": _plot_series(dist_vol.x, dist_vol.fractions, title="Species distribution vs volume", xlabel="volume_mL", ylabel="fraction", labels=dist_vol.species_names),
    }
    export_data = {
        "experimental_vs_calculated": pd.DataFrame(
            {
                "volume_mL": volumes,
                "pH_observed": obs_for_plot,
                "pH_calculated": calc_pH,
                "residual_pH": residuals,
            }
        ).to_dict(orient="list"),
        "species_vs_pH": pd.DataFrame(
            np.column_stack([dist_ph.x, dist_ph.fractions]),
            columns=["pH"] + dist_ph.species_names,
        ).to_dict(orient="list"),
        "species_vs_volume": pd.DataFrame(
            np.column_stack([dist_vol.x, dist_vol.pH, dist_vol.fractions]),
            columns=["volume_mL", "pH"] + dist_vol.species_names,
        ).to_dict(orient="list"),
    }
    return _base_result_payload(result, analysis_kind="potentiometry", graphs=graphs, export_data=export_data)


def _run_spectroscopy(cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    df = _read_csv(cfg["file_path"])
    ph_col = _csv_column(df, ["pH", "ph"])
    if ph_col is None:
        raise ValueError("Spectroscopy CSV needs a pH column.")
    pH = pd.to_numeric(df[ph_col], errors="coerce").to_numpy(dtype=float)
    value_cols = [col for col in df.columns if str(col) != ph_col]
    if not value_cols:
        raise ValueError("Spectroscopy CSV needs signal columns.")
    signal = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if signal.shape[1] == 1:
        signal = signal[:, 0]
    wavelengths = None
    try:
        wavelengths = np.asarray([float(c) for c in value_cols], dtype=float)
    except Exception:
        wavelengths = None
    dataset = SpectroscopicAcidBaseDataset(pH=pH, signal=signal, wavelengths=wavelengths)
    system = _build_system(cfg)
    initial_pka = _parse_float_list(cfg.get("pka_initial") or cfg.get("initial_pka"), default=[5.0])
    log("Fitting spectroscopic acid-base data...")
    result = fit_spectroscopy_acid_base(
        dataset,
        system,
        initial_pka=initial_pka,
        fit_options={"baseline": bool(cfg.get("baseline", False))},
    )
    fitted_system = clone_system_with_pka(system, result.fitted_pka)
    pred = predict_spectroscopy_acid_base(dataset, fitted_system, {"baseline": bool(cfg.get("baseline", False))})
    residuals = np.asarray(dataset.signal, dtype=float) - np.asarray(pred["calculated"], dtype=float)
    dist_ph = simulate_species_vs_pH(fitted_system)
    graphs = {
        "fit": _plot_overlay(pH, dataset.signal, pred["calculated"], title="Spectroscopic signal vs pH", xlabel="pH", ylabel="signal"),
        "residuals": _plot_series(pH, residuals, title="Spectroscopy residuals", xlabel="pH", ylabel="observed - calculated", marker="o"),
        "species_pH": _plot_series(dist_ph.x, dist_ph.fractions, title="Species distribution vs pH", xlabel="pH", ylabel="fraction", labels=dist_ph.species_names),
        "species_volume": "",
    }
    export_data = {
        "experimental_vs_calculated": pd.DataFrame(
            {
                "pH": pH,
                "signal_observed": np.asarray(dataset.signal).reshape(pH.size, -1)[:, 0],
                "signal_calculated": np.asarray(pred["calculated"]).reshape(pH.size, -1)[:, 0],
            }
        ).to_dict(orient="list"),
        "species_vs_pH": pd.DataFrame(
            np.column_stack([dist_ph.x, dist_ph.fractions]),
            columns=["pH"] + dist_ph.species_names,
        ).to_dict(orient="list"),
        "pure_signals": np.asarray(pred["coefficients"], dtype=float).tolist(),
    }
    return _base_result_payload(result, analysis_kind="spectroscopy", graphs=graphs, export_data=export_data)


def _run_nmr(cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    df = _read_csv(cfg["file_path"])
    ph_col = _csv_column(df, ["pH", "ph"])
    if ph_col is None:
        raise ValueError("NMR CSV needs a pH column.")
    pH = pd.to_numeric(df[ph_col], errors="coerce").to_numpy(dtype=float)
    shift_cols = [col for col in df.columns if str(col) != ph_col]
    if not shift_cols:
        raise ValueError("NMR CSV needs one or more shift columns.")
    shifts = df[shift_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    dataset = NMRAcidBaseDataset(pH=pH, shifts=shifts, nuclei_labels=[str(c) for c in shift_cols])
    system = _build_system(cfg)
    initial_pka = _parse_float_list(cfg.get("pka_initial") or cfg.get("initial_pka"), default=[5.0])
    log("Fitting fast-exchange 1H NMR acid-base data...")
    result = fit_nmr_acid_base(dataset, system, initial_pka=initial_pka)
    fitted_system = clone_system_with_pka(system, result.fitted_pka)
    pred = predict_nmr_acid_base(dataset, fitted_system)
    residuals = np.asarray(dataset.shifts, dtype=float) - np.asarray(pred["calculated"], dtype=float)
    dist_ph = simulate_species_vs_pH(fitted_system)
    graphs = {
        "fit": _plot_overlay(pH, dataset.shifts, pred["calculated"], title="NMR shifts vs pH", xlabel="pH", ylabel="shift"),
        "residuals": _plot_series(pH, residuals, title="NMR residuals", xlabel="pH", ylabel="observed - calculated", marker="o"),
        "species_pH": _plot_series(dist_ph.x, dist_ph.fractions, title="Species distribution vs pH", xlabel="pH", ylabel="fraction", labels=dist_ph.species_names),
        "species_volume": "",
    }
    export_data = {
        "experimental_vs_calculated": pd.DataFrame(
            {
                "pH": pH,
                "shift_observed": np.asarray(dataset.shifts).reshape(pH.size, -1)[:, 0],
                "shift_calculated": np.asarray(pred["calculated"]).reshape(pH.size, -1)[:, 0],
            }
        ).to_dict(orient="list"),
        "species_vs_pH": pd.DataFrame(
            np.column_stack([dist_ph.x, dist_ph.fractions]),
            columns=["pH"] + dist_ph.species_names,
        ).to_dict(orient="list"),
        "limiting_shifts": np.asarray(pred["coefficients"], dtype=float).tolist(),
    }
    return _base_result_payload(result, analysis_kind="nmr", graphs=graphs, export_data=export_data)


def run_acid_base(config: Mapping[str, Any] | Any, progress_cb: ProgressCallback = None) -> dict[str, Any]:
    cfg = _normalize_config(config)

    def log(message: str) -> None:
        if progress_cb is not None:
            progress_cb(str(message))
        else:
            print(str(message))

    kind = str(cfg.get("data_type") or cfg.get("analysis_kind") or "potentiometry").strip().lower()
    if kind in {"pot", "ph", "emf", "potentiometric"}:
        kind = "potentiometry"
    if kind in {"uvvis", "fluorescence", "spectro"}:
        kind = "spectroscopy"
    if kind in {"1h_nmr", "proton_nmr"}:
        kind = "nmr"

    if kind == "potentiometry":
        return _run_potentiometry(cfg, log)
    if kind == "spectroscopy":
        return _run_spectroscopy(cfg, log)
    if kind == "nmr":
        return _run_nmr(cfg, log)
    raise ValueError(f"Unknown acid-base analysis kind: {kind}")


__all__ = ["run_acid_base"]
