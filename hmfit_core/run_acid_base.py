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
    AcidBaseComponent,
    AcidBaseFitResult,
    AcidBaseSpecies,
    AcidBaseSystem,
    NMRAcidBaseDataset,
    SpectroscopicAcidBaseDataset,
    clone_system_with_pka,
    component_log_beta,
    fit_nmr_acid_base,
    fit_spectroscopy_acid_base,
    make_simple_acid_base_system,
    pka_to_log_beta,
    predict_nmr_acid_base,
    predict_spectroscopy_acid_base,
    simulate_species_vs_pH,
    system_pka_values,
)
from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
from hmfit_core.acid_base_model_utils import (
    acid_base_model_from_simple_config,
    acid_base_system_from_model,
    serializable_model_definition_from_system,
)
from hmfit_core.potentiometry import (
    PotentiometryExperiment,
    electrode_emf_from_pH,
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


def _read_table(file_path: str | Path, sheet_name: Any = None) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".xlsx":
        sheet = 0 if sheet_name in (None, "") else sheet_name
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError("Acid-base v1 imports CSV/TXT/XLSX files.")


def _configured_sheet(cfg: Mapping[str, Any]) -> Any:
    return cfg.get("sheet_name") or cfg.get("data_sheet") or cfg.get("excel_sheet")


def _kw_from_config(cfg: Mapping[str, Any]) -> float:
    if cfg.get("kw") not in (None, ""):
        kw = float(cfg.get("kw"))
    elif cfg.get("pkw") not in (None, ""):
        kw = 10.0 ** (-float(cfg.get("pkw")))
    else:
        kw = 1e-14
    if not np.isfinite(kw) or kw <= 0.0:
        raise ValueError("Kw must be a positive finite value.")
    return float(kw)


def _pkw_from_kw(kw: float) -> float:
    return float(-np.log10(float(kw)))


def _pkw_from_config(cfg: Mapping[str, Any], *, default_kw: float = 1.0e-14) -> float:
    if cfg.get("pkw") not in (None, ""):
        pkw = float(cfg.get("pkw"))
    elif cfg.get("kw") not in (None, ""):
        pkw = _pkw_from_kw(float(cfg.get("kw")))
    else:
        pkw = _pkw_from_kw(default_kw)
    if not np.isfinite(pkw):
        raise ValueError("pKw must be a finite numeric value.")
    return float(pkw)


def _pkw_bounds_from_config(cfg: Mapping[str, Any]) -> tuple[float, float]:
    raw = cfg.get("pkw_bounds")
    if isinstance(raw, Mapping):
        lo = raw.get("min", raw.get("lower", 0.0))
        hi = raw.get("max", raw.get("upper", 30.0))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        lo, hi = raw[0], raw[1]
    else:
        lo = cfg.get("pkw_min", 0.0)
        hi = cfg.get("pkw_max", 30.0)
    lo_f = float(lo)
    hi_f = float(hi)
    if not (np.isfinite(lo_f) and np.isfinite(hi_f)):
        raise ValueError("pKw bounds must be finite numeric values.")
    if lo_f >= hi_f:
        raise ValueError("pKw min must be lower than pKw max.")
    return lo_f, hi_f


def _build_system(cfg: Mapping[str, Any]):
    model_def = cfg.get("acid_base_model")
    if isinstance(model_def, Mapping) and model_def.get("components"):
        return _build_system_from_model(model_def, cfg)

    pka = _parse_float_list(cfg.get("pka_initial") or cfg.get("initial_pka"), default=[5.0])
    concentration = float(cfg.get("analyte_concentration", cfg.get("concentration", 1.0e-3)) or 1.0e-3)
    base_charge = int(cfg.get("base_charge", -1) or -1)
    return make_simple_acid_base_system(
        name=str(cfg.get("component_name") or "L"),
        analytical_concentration=concentration,
        pka=pka,
        base_charge=base_charge,
        temperature=float(cfg.get("temperature", 298.15) or 298.15),
        kw=_kw_from_config(cfg),
    )


def _build_system_from_model(model_def: Mapping[str, Any], cfg: Mapping[str, Any]) -> AcidBaseSystem:
    return acid_base_system_from_model(
        model_def,
        temperature=float(cfg.get("temperature", 298.15) or 298.15),
        ionic_strength=(
            None
            if cfg.get("ionic_strength") in (None, "")
            else float(cfg.get("ionic_strength"))
        ),
        kw=_kw_from_config(cfg),
    )


def _initial_pka_from_config(cfg: Mapping[str, Any], system: AcidBaseSystem) -> list[float]:
    raw = cfg.get("pka_initial") or cfg.get("initial_pka")
    if raw not in (None, ""):
        return _parse_float_list(raw, default=system_pka_values(system))
    try:
        return system_pka_values(system)
    except Exception:
        return [5.0]


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


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _serializable_model_definition(cfg: Mapping[str, Any], system: AcidBaseSystem) -> dict[str, Any]:
    model_def = cfg.get("acid_base_model")
    if isinstance(model_def, Mapping) and model_def.get("components"):
        return _to_serializable(serializable_model_definition_from_system(model_def, system=system))
    return _to_serializable(
        serializable_model_definition_from_system(
            acid_base_model_from_simple_config(
                component_name=str(cfg.get("component_name") or "L"),
                pka=_parse_float_list(cfg.get("pka_initial") or cfg.get("initial_pka"), default=[5.0]),
                analytical_concentration=float(cfg.get("analyte_concentration", cfg.get("concentration", 1.0e-3)) or 1.0e-3),
                base_charge=int(cfg.get("base_charge", -1) or -1),
            ),
            system=system,
        )
    )


def _serializable_bounds(bounds: Any, n_params: int) -> list[list[float | None]]:
    out: list[list[float | None]] = [[None, None] for _ in range(max(0, int(n_params)))]
    if bounds is None:
        return out
    if isinstance(bounds, tuple) and len(bounds) == 2:
        left = np.asarray(bounds[0], dtype=float).reshape(-1)
        right = np.asarray(bounds[1], dtype=float).reshape(-1)
        if left.size == 1:
            left = np.full(n_params, float(left[0]), dtype=float)
        if right.size == 1:
            right = np.full(n_params, float(right[0]), dtype=float)
        for idx in range(min(n_params, left.size, right.size)):
            out[idx] = [float(left[idx]), float(right[idx])]
        return out
    for idx, row in enumerate(list(bounds or [])[:n_params]):
        if isinstance(row, Mapping):
            lo = row.get("min")
            hi = row.get("max")
        else:
            seq = list(row or [])
            lo = seq[0] if len(seq) >= 1 else None
            hi = seq[1] if len(seq) >= 2 else None
        out[idx] = [
            None if lo in (None, "") else float(lo),
            None if hi in (None, "") else float(hi),
        ]
    return out


def _build_errors_context(
    *,
    cfg: Mapping[str, Any],
    system: AcidBaseSystem,
    result: AcidBaseFitResult,
    analysis_kind: str,
    dataset: dict[str, Any],
    fit_options: Mapping[str, Any],
) -> dict[str, Any]:
    theta_hat = np.asarray(result.theta_hat, dtype=float).reshape(-1)
    return {
        "analysis_type": "acid_base",
        "analysis_kind": str(analysis_kind),
        "theta_hat": theta_hat.tolist(),
        "parameter_names": list(result.parameter_names or [row["parameter"] for row in _fit_result_tables(result)[0]]),
        "fixed_mask": list(result.fixed_mask or [False for _ in range(theta_hat.size)]),
        "bounds": _serializable_bounds(result.bounds, theta_hat.size),
        "jacobian": (
            None
            if result.jacobian is None
            else np.asarray(result.jacobian, dtype=float).tolist()
        ),
        "residuals": np.asarray(result.residuals, dtype=float).tolist(),
        "primary_constant_space": str(result.parameter_space or "pka"),
        "constant_parameter_indices": list(range(len(result.fitted_pka))),
        "n_acid_base_constants": len(result.fitted_pka),
        "dataset": _to_serializable(dataset),
        "model": {
            "acid_base_model": _serializable_model_definition(cfg, system),
            "temperature": float(system.temperature),
            "ionic_strength": (
                None if system.ionic_strength is None else float(system.ionic_strength)
            ),
            "kw": float(system.kw),
        },
        "fit_options": _to_serializable(dict(fit_options or {})),
    }


def _merge_error_output_into_export(export_data: dict[str, Any], error_output: Mapping[str, Any]) -> None:
    frames = dict(error_output.get("export_frames") or {})
    if frames:
        if "Parameters" in frames:
            export_data["error_parameters"] = frames["Parameters"]
        if "pKa" in frames:
            export_data["error_pka_table"] = frames["pKa"]
        if "log_beta" in frames:
            export_data["error_log_beta_table"] = frames["log_beta"]
        if "Derived constants" in frames:
            export_data["derived_constants"] = frames["Derived constants"]
        if "Covariance" in frames:
            export_data["error_covariance_matrix"] = frames["Covariance"]
        if "Correlation" in frames:
            export_data["error_correlation_matrix"] = frames["Correlation"]
        if "Error diagnostics" in frames:
            export_data["error_diagnostics"] = frames["Error diagnostics"]
        if "Bootstrap samples" in frames:
            export_data["bootstrap_samples"] = frames["Bootstrap samples"]

    core_metrics = dict(error_output.get("core_metrics") or {})
    stability_diag = dict(core_metrics.get("stability_diag") or {})
    export_data["diagnostics_summary"] = str(error_output.get("summary") or "")
    export_data["diagnostics_full"] = str(stability_diag.get("diag_full") or "")
    export_data["stability_status"] = stability_diag.get("status")
    export_data["condition_number"] = stability_diag.get("cond_jjt")
    export_data["stability_indicator"] = stability_diag.get("stability_indicator")


def _base_result_payload(
    result: AcidBaseFitResult,
    *,
    analysis_kind: str,
    graphs: dict[str, str],
    export_data: dict[str, Any],
    errors_context: dict[str, Any] | None = None,
    error_output: Mapping[str, Any] | None = None,
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
    if error_output is not None:
        core_metrics = dict(error_output.get("core_metrics") or {})
        stability_diag = dict(core_metrics.get("stability_diag") or {})
        stats["stability_status"] = stability_diag.get("status")
        stats["condition_number"] = stability_diag.get("cond_jjt")
        stats["max_abs_correlation"] = stability_diag.get("max_abs_corr")
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
    if errors_context is not None:
        export_data["errors_context"] = _to_serializable(errors_context)
    if error_output is not None:
        _merge_error_output_into_export(export_data, error_output)
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
        "errors_context": _to_serializable(errors_context) if errors_context is not None else None,
    }


def _run_potentiometry(cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    signal_type = str(cfg.get("signal_type") or "").strip()
    volume_unit = str(cfg.get("volume_unit") or "mL").strip()
    if cfg.get("titrant_volume") is not None and cfg.get("observed_signal") is not None:
        volumes_all = np.asarray(cfg.get("titrant_volume"), dtype=float).reshape(-1)
        observed_all = np.asarray(cfg.get("observed_signal"), dtype=float).reshape(-1)
        include_raw = cfg.get("included_mask")
        if include_raw is None:
            include_mask = np.ones(volumes_all.shape, dtype=bool)
        else:
            include_mask = np.asarray(include_raw, dtype=bool).reshape(-1)
        if volumes_all.size != observed_all.size or volumes_all.size != include_mask.size:
            raise ValueError("Potentiometry titrant_volume, observed_signal, and included_mask must have the same length.")
        volumes = volumes_all[include_mask]
        observed = observed_all[include_mask]
    else:
        df = _read_table(cfg["file_path"], _configured_sheet(cfg))
        volume_col = _csv_column(df, ["volume_mL", "volume", "v", "v_ml"])
        if volume_col is None:
            raise ValueError("Potentiometry data needs a volume_mL column.")
        ph_col = _csv_column(df, ["pH", "ph"])
        emf_col = _csv_column(df, ["E_mV", "emf", "emf_mV", "E"])
        if ph_col is None and emf_col is None:
            raise ValueError("Potentiometry data needs pH or E_mV.")
        volumes = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=float)
        measured_pH = None
        measured_emf = None
        if ph_col is not None:
            measured_pH = pd.to_numeric(df[ph_col], errors="coerce").to_numpy(dtype=float)
            signal_type = signal_type or "pH"
        if emf_col is not None:
            measured_emf = pd.to_numeric(df[emf_col], errors="coerce").to_numpy(dtype=float)
            if measured_pH is None:
                signal_type = signal_type or "mV"
        observed = measured_emf if measured_pH is None else measured_pH
        include_mask = np.ones(volumes.shape, dtype=bool)

    if volumes.size == 0:
        raise ValueError("Potentiometry dataset has no included rows.")
    measured_pH = None
    measured_emf = None
    if signal_type.lower() in {"ph", "ph*"}:
        measured_pH = observed
    elif signal_type.lower() == "mv":
        measured_emf = observed
    else:
        raise ValueError("Potentiometry signal_type must be pH, pH*, or mV.")
    for warning in list(cfg.get("potentiometry_warnings") or []):
        log(f"Warning: {warning}")

    system = _build_system(cfg)
    initial_pka = _initial_pka_from_config(cfg, system)
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
    if cfg.get("volume_offset") not in (None, ""):
        fit_options["volume_offset"] = float(cfg.get("volume_offset"))
    if cfg.get("pH_bounds") not in (None, ""):
        bounds = list(cfg.get("pH_bounds") or [])
        if len(bounds) >= 2:
            fit_options["pH_bounds"] = (float(bounds[0]), float(bounds[1]))
    if cfg.get("initial_strong_charge") not in (None, ""):
        fit_options["initial_strong_charge"] = float(cfg.get("initial_strong_charge"))
    if cfg.get("titrant_strong_charge") not in (None, ""):
        fit_options["titrant_strong_charge"] = float(cfg.get("titrant_strong_charge"))
    if isinstance(cfg.get("initial_concentrations"), Mapping):
        fit_options["initial_concentrations"] = dict(cfg.get("initial_concentrations") or {})
    if isinstance(cfg.get("titrant_concentrations"), Mapping):
        fit_options["titrant_concentrations"] = dict(cfg.get("titrant_concentrations") or {})
    pkw_initial = _pkw_from_config(cfg, default_kw=system.kw)
    pkw_min, pkw_max = _pkw_bounds_from_config(cfg)
    if not pkw_min <= pkw_initial <= pkw_max:
        raise ValueError("Initial pKw must be within the configured pKw bounds.")

    pka_fit_mask_raw = cfg.get("pka_fit_mask")
    if pka_fit_mask_raw is None:
        pka_fit_mask = [True] * len(initial_pka)
    else:
        pka_fit_mask = [bool(v) for v in list(pka_fit_mask_raw)[: len(initial_pka)]]
        while len(pka_fit_mask) < len(initial_pka):
            pka_fit_mask.append(True)
    pka_bounds_raw = list(cfg.get("pka_bounds") or [])
    parameter_names = []
    initial_params = []
    lower_bounds = []
    upper_bounds = []
    for idx, value in enumerate(initial_pka):
        if not pka_fit_mask[idx]:
            continue
        lo, hi = -5.0, 25.0
        if idx < len(pka_bounds_raw):
            raw_bounds = pka_bounds_raw[idx]
            if isinstance(raw_bounds, Mapping):
                lo = float(raw_bounds.get("min", lo))
                hi = float(raw_bounds.get("max", hi))
            else:
                seq = list(raw_bounds or [])
                if len(seq) >= 2:
                    lo, hi = float(seq[0]), float(seq[1])
        if lo >= hi:
            raise ValueError(f"pKa{idx + 1} min must be lower than max.")
        parameter_names.append(f"pKa{idx + 1}")
        initial_params.append(float(value))
        lower_bounds.append(float(lo))
        upper_bounds.append(float(hi))
    fit_pkw = bool(cfg.get("fit_pkw", cfg.get("fit_pKw", False)))
    if bool(cfg.get("fit_analyte_concentration", False)):
        parameter_names.append("analyte_concentration")
        initial_params.append(float(experiment.analyte_concentration or 0.0))
        lower_bounds.append(1.0e-12)
        upper_bounds.append(1.0e6)
    if bool(cfg.get("fit_titrant_concentration", False)):
        parameter_names.append("titrant_concentration")
        initial_params.append(float(experiment.titrant_concentration or 0.0))
        lower_bounds.append(1.0e-12)
        upper_bounds.append(1.0e6)
    if bool(cfg.get("fit_volume_offset", False)):
        parameter_names.append("volume_offset")
        initial_params.append(float(fit_options.get("volume_offset", experiment.volume_offset or 0.0) or 0.0))
        lower_bounds.append(-1.0e9)
        upper_bounds.append(1.0e9)
    if bool(cfg.get("fit_electrode", False)) and measured_emf is not None:
        parameter_names.extend(["electrode_e0", "electrode_slope"])
        initial_params.extend(
            [
                0.0 if experiment.electrode_e0 is None else float(experiment.electrode_e0),
                -59.16 if experiment.electrode_slope is None else float(experiment.electrode_slope),
            ]
        )
        lower_bounds.extend([-1.0e5, -120.0])
        upper_bounds.extend([1.0e5, 120.0])
        fit_options["fit_signal"] = "emf"
    if fit_pkw:
        parameter_names.append("pKw")
        initial_params.append(float(pkw_initial))
        lower_bounds.append(float(pkw_min))
        upper_bounds.append(float(pkw_max))
        correlated = [
            "pKa",
            "analyte concentration" if bool(cfg.get("fit_analyte_concentration", False)) else "",
            "titrant concentration" if bool(cfg.get("fit_titrant_concentration", False)) else "",
            "volume offset" if bool(cfg.get("fit_volume_offset", False)) else "",
            "electrode parameters" if bool(cfg.get("fit_electrode", False)) and measured_emf is not None else "",
        ]
        correlated = [item for item in correlated if item]
        if len(correlated) >= 2:
            log(
                "Warning: Fitting pKw together with concentration, electrode and volume offset "
                "parameters may lead to strong parameter correlation. For reliable results, fit "
                "pKw only when the titration data contain enough information and the solvent "
                "system justifies it."
            )
    default_all_pka = [f"pKa{idx + 1}" for idx in range(len(initial_pka))]
    if not parameter_names:
        raise ValueError("Select at least one parameter to fit.")
    if parameter_names != default_all_pka or pka_bounds_raw:
        fit_options["parameter_names"] = parameter_names
        fit_options["initial_params"] = initial_params
        fit_options["bounds"] = (lower_bounds, upper_bounds)
    log(f"Water autoionization: pKw={_pkw_from_kw(system.kw):.4f}, Kw={system.kw:.6g}")
    log("Fitting potentiometric acid-base data...")
    result = fit_potentiometry(experiment, system, initial_pka=initial_pka, fit_options=fit_options)
    fitted_params = {
        str(name).strip().lower().replace(" ", "_").replace("-", "_"): float(value)
        for name, value in zip(result.parameter_names or [], np.asarray(result.theta_hat, dtype=float).reshape(-1))
    }
    fitted_pkw = float(fitted_params.get("pkw", _pkw_from_kw(system.kw)))
    fitted_system = clone_system_with_pka(system, result.fitted_pka)
    fitted_system.kw = 10.0 ** (-fitted_pkw)
    calc_pH = simulate_pH_titration(experiment, fitted_system, fit_options)
    obs_ph = observed_pH(experiment)
    obs_for_plot = measured_pH if measured_pH is not None else obs_ph
    residuals = np.asarray(obs_for_plot, dtype=float).reshape(-1) - calc_pH
    calc_emf = electrode_emf_from_pH(
        calc_pH,
        electrode_e0=fitted_params.get("electrode_e0", experiment.electrode_e0),
        electrode_slope=fitted_params.get("electrode_slope", experiment.electrode_slope),
    )
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
                "E_mV_observed": measured_emf,
                "E_mV_calculated": calc_emf if measured_emf is not None else np.nan,
                "residual_pH": residuals,
                "signal_type": [signal_type] * volumes.size,
                "source_volume_unit": [volume_unit] * volumes.size,
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
    dataset_context = {
        "volume_mL": volumes.tolist(),
        "pH_observed": None if measured_pH is None else np.asarray(measured_pH, dtype=float).tolist(),
        "E_mV_observed": None if measured_emf is None else np.asarray(measured_emf, dtype=float).tolist(),
        "pH_calculated": np.asarray(calc_pH, dtype=float).tolist(),
        "E_mV_calculated": np.asarray(calc_emf, dtype=float).tolist(),
        "observed_kind": "emf" if measured_pH is None and measured_emf is not None else "ph",
        "signal_type": signal_type,
        "source_volume_unit": volume_unit,
        "initial_volume": float(experiment.initial_volume),
        "analyte_concentration": (
            None if experiment.analyte_concentration is None else float(experiment.analyte_concentration)
        ),
        "titrant_concentration": (
            None if experiment.titrant_concentration is None else float(experiment.titrant_concentration)
        ),
        "titrant_type": str(experiment.titrant_type),
        "temperature": float(experiment.temperature),
        "electrode_e0": (
            None if fitted_params.get("electrode_e0") is None else float(fitted_params["electrode_e0"])
        ),
        "electrode_slope": (
            None if fitted_params.get("electrode_slope") is None else float(fitted_params["electrode_slope"])
        ),
        "volume_offset": float(fit_options.get("volume_offset", experiment.volume_offset or 0.0) or 0.0),
        "pkw": float(fitted_pkw),
        "kw": float(fitted_system.kw),
        "pkw_fitted": bool(fit_pkw),
    }
    errors_context = _build_errors_context(
        cfg=cfg,
        system=fitted_system,
        result=result,
        analysis_kind="potentiometry",
        dataset=dataset_context,
        fit_options=fit_options,
    )
    error_output = None
    try:
        error_output = compute_errors_acid_base_from_context(
            errors_context,
            {"method": "analytic", "include_16_84": True},
        )
    except Exception as exc:
        log(f"Analytical acid-base errors unavailable: {exc}")
    payload = _base_result_payload(
        result,
        analysis_kind="potentiometry",
        graphs=graphs,
        export_data=export_data,
        errors_context=errors_context,
        error_output=error_output,
    )
    pkw_row = {
        "parameter": "pKw_app" if fit_pkw else "apparent pKw",
        "value": float(fitted_pkw),
        "fixed": not bool(fit_pkw),
        "description": (
            "pKw was fitted as an apparent medium parameter."
            if fit_pkw
            else "Fixed apparent medium autoprotolysis parameter."
        ),
    }
    payload["constants"].append({"parameter": pkw_row["parameter"], "value": pkw_row["value"]})
    payload["results_text"] = (
        str(payload.get("results_text") or "")
        + f"\n{pkw_row['parameter']}: {pkw_row['value']:.8g} "
        + ("(fitted apparent medium parameter)" if fit_pkw else "(fixed)")
    )
    payload["export_data"]["medium_parameters"] = [pkw_row]
    if fit_pkw:
        payload["export_data"]["potentiometry_note"] = "pKw was fitted as an apparent medium parameter."
    species_names = []
    try:
        species_names = [str(sp.name) for comp in fitted_system.components for sp in comp.species]
    except Exception:
        species_names = []
    summary_lines = [
        "Acid-base fit summary",
        "",
        "Model:",
        f"- Species: {', '.join(species_names) if species_names else 'n/a'}",
        f"- Titrant: {str(experiment.titrant_type)}",
        f"- Initial volume: {float(experiment.initial_volume):.8g}",
        f"- Analyte concentration: {float(experiment.analyte_concentration or 0.0):.8g}",
        f"- Titrant concentration: {float(experiment.titrant_concentration or 0.0):.8g}",
        f"- Points included: {int(volumes.size)}",
        "",
        "Fitted parameters:",
    ]
    for row in payload.get("constants") or []:
        name = str(row.get("parameter") or "")
        if name.startswith("pKa") or name == "pKw_app":
            summary_lines.append(f"- {name} = {float(row.get('value')):.8g}")
    summary_lines.extend(
        [
            "",
            "Statistics:",
            f"- chi_square = {float(result.chi_square):.8g}",
        ]
    )
    if result.reduced_chi_square is not None:
        summary_lines.append(f"- reduced_chi_square = {float(result.reduced_chi_square):.8g}")
    if result.aic is not None:
        summary_lines.append(f"- AIC = {float(result.aic):.8g}")
    if result.bic is not None:
        summary_lines.append(f"- BIC = {float(result.bic):.8g}")
    warnings_out: list[str] = []
    if fit_pkw:
        warnings_out.append("pKw_app fitted as an apparent medium parameter.")
    if warnings_out:
        summary_lines.append("")
        summary_lines.append("Warnings:")
        summary_lines.extend(f"- {warning}" for warning in warnings_out)
    payload["results_text"] = "\n".join(summary_lines) + "\n\n" + str(payload.get("results_text") or "")
    payload["export_data"]["fit_summary"] = {"line": summary_lines}
    return payload


def _run_spectroscopy(cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    df = _read_table(cfg["file_path"], _configured_sheet(cfg))
    ph_col = _csv_column(df, ["pH", "ph"])
    if ph_col is None:
        raise ValueError("Spectroscopy data needs a pH column.")
    pH = pd.to_numeric(df[ph_col], errors="coerce").to_numpy(dtype=float)
    value_cols = [col for col in df.columns if str(col) != ph_col]
    if not value_cols:
        raise ValueError("Spectroscopy data needs signal columns.")
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
    initial_pka = _initial_pka_from_config(cfg, system)
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
    errors_context = _build_errors_context(
        cfg=cfg,
        system=system,
        result=result,
        analysis_kind="spectroscopy",
        dataset={
            "pH": np.asarray(dataset.pH, dtype=float).tolist(),
            "signal_observed": np.asarray(dataset.signal, dtype=float).tolist(),
            "signal_calculated": np.asarray(pred["calculated"], dtype=float).tolist(),
            "wavelengths": None if dataset.wavelengths is None else np.asarray(dataset.wavelengths, dtype=float).tolist(),
        },
        fit_options={"baseline": bool(cfg.get("baseline", False))},
    )
    error_output = None
    try:
        error_output = compute_errors_acid_base_from_context(
            errors_context,
            {"method": "analytic", "include_16_84": True},
        )
    except Exception as exc:
        log(f"Analytical acid-base errors unavailable: {exc}")
    return _base_result_payload(
        result,
        analysis_kind="spectroscopy",
        graphs=graphs,
        export_data=export_data,
        errors_context=errors_context,
        error_output=error_output,
    )


def _run_nmr(cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    df = _read_table(cfg["file_path"], _configured_sheet(cfg))
    ph_col = _csv_column(df, ["pH", "ph"])
    if ph_col is None:
        raise ValueError("NMR data needs a pH column.")
    pH = pd.to_numeric(df[ph_col], errors="coerce").to_numpy(dtype=float)
    shift_cols = [col for col in df.columns if str(col) != ph_col]
    if not shift_cols:
        raise ValueError("NMR data needs one or more shift columns.")
    shifts = df[shift_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    dataset = NMRAcidBaseDataset(pH=pH, shifts=shifts, nuclei_labels=[str(c) for c in shift_cols])
    system = _build_system(cfg)
    initial_pka = _initial_pka_from_config(cfg, system)
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
    errors_context = _build_errors_context(
        cfg=cfg,
        system=system,
        result=result,
        analysis_kind="nmr",
        dataset={
            "pH": np.asarray(dataset.pH, dtype=float).tolist(),
            "shifts_observed": np.asarray(dataset.shifts, dtype=float).tolist(),
            "shifts_calculated": np.asarray(pred["calculated"], dtype=float).tolist(),
            "nuclei_labels": [str(value) for value in dataset.nuclei_labels],
        },
        fit_options={},
    )
    error_output = None
    try:
        error_output = compute_errors_acid_base_from_context(
            errors_context,
            {"method": "analytic", "include_16_84": True},
        )
    except Exception as exc:
        log(f"Analytical acid-base errors unavailable: {exc}")
    return _base_result_payload(
        result,
        analysis_kind="nmr",
        graphs=graphs,
        export_data=export_data,
        errors_context=errors_context,
        error_output=error_output,
    )


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
