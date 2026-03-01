"""Reporting helpers for Kinetics GUI/XLSX export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset


NOT_AVAILABLE_MESSAGE = "Not available for this kinetics run"


@dataclass(frozen=True)
class ParameterRows:
    names: list[str]
    estimate: np.ndarray
    se: np.ndarray
    perc_err: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    fixed: list[bool]
    scale: list[str]


def _to_float_array(values: Any, size: int, fill: float = np.nan) -> np.ndarray:
    if values is None:
        return np.full(size, fill, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == size:
        return arr
    out = np.full(size, fill, dtype=float)
    n = min(size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _not_available_df(message: str = NOT_AVAILABLE_MESSAGE) -> pd.DataFrame:
    return pd.DataFrame({"message": [str(message)]})


def _iter_fit_datasets(result: dict[str, Any]) -> list[KineticsFitDataset]:
    return [
        ds
        for ds in (result.get("datasets") or [])
        if isinstance(ds, KineticsFitDataset)
    ]


def _reaction_to_text(reaction: Any) -> str:
    reactants = getattr(reaction, "reactants", {}) or {}
    products = getattr(reaction, "products", {}) or {}
    k_f = getattr(reaction, "k_forward", None)
    k_r = getattr(reaction, "k_reverse", None)

    def _side_text(side: dict[str, Any]) -> str:
        parts: list[str] = []
        for species, coeff in side.items():
            try:
                coeff_i = int(coeff)
            except Exception:
                coeff_i = coeff
            if coeff_i == 1:
                parts.append(str(species))
            else:
                parts.append(f"{coeff_i} {species}")
        return " + ".join(parts) if parts else "0"

    lhs = _side_text(reactants)
    rhs = _side_text(products)
    if k_r:
        return f"{lhs} <=> {rhs} ; kf={k_f}, kr={k_r}"
    return f"{lhs} -> {rhs} ; k={k_f}"


def extract_parameter_rows(result: dict[str, Any]) -> ParameterRows:
    names = list(result.get("param_names") or [])
    n = len(names)
    params = result.get("params") or {}
    estimate = np.asarray([float(params.get(name, np.nan)) for name in names], dtype=float)

    err = result.get("parameter_errors") or {}
    se = _to_float_array(err.get("se"), n, fill=np.nan)
    perc_err = _to_float_array(err.get("perc_err"), n, fill=np.nan)
    ci_low = _to_float_array(err.get("ci_low"), n, fill=np.nan)
    ci_high = _to_float_array(err.get("ci_high"), n, fill=np.nan)
    fixed_raw = list(err.get("fixed") or [])
    fixed = [bool(fixed_raw[i]) if i < len(fixed_raw) else False for i in range(n)]
    scale_raw = list(err.get("scale") or [])
    scale = [str(scale_raw[i]) if i < len(scale_raw) else "linear" for i in range(n)]
    return ParameterRows(
        names=names,
        estimate=estimate,
        se=se,
        perc_err=perc_err,
        ci_low=ci_low,
        ci_high=ci_high,
        fixed=fixed,
        scale=scale,
    )


def build_summary_df(result: dict[str, Any]) -> pd.DataFrame:
    stats = result.get("statistics") or {}
    diagnostics = result.get("diagnostics") or {}
    stability = result.get("stability_indicator") or {}
    now = datetime.now().isoformat(timespec="seconds")
    timestamp = result.get("completed_at") or now
    row_data = [
        ("timestamp", timestamp),
        ("success", bool(result.get("success", True))),
        ("optimizer_method", result.get("optimizer_method")),
        ("optimizer_requested", result.get("optimizer_requested")),
        ("optimizer_backend", result.get("optimizer_backend")),
        ("nfev", result.get("nfev")),
        ("njev", result.get("njev")),
        ("nfev_total", result.get("nfev_total")),
        ("njev_total", result.get("njev_total")),
        ("multi_start_runs", result.get("multi_start_runs")),
        ("RMS", stats.get("RMS")),
        ("SSQ", result.get("ssq")),
        ("cost", result.get("cost")),
        ("condition_number", result.get("condition_number")),
        ("stability_status", diagnostics.get("stability_status")),
        ("stability_label", stability.get("label")),
        ("warnings_count", len(diagnostics.get("warnings") or [])),
    ]
    params = extract_parameter_rows(result)
    for idx, name in enumerate(params.names):
        row_data.append((f"param_{name}_estimate", params.estimate[idx]))
        row_data.append((f"param_{name}_SE", params.se[idx]))
        row_data.append((f"param_{name}_%err", params.perc_err[idx]))
    return pd.DataFrame(row_data, columns=["metric", "value"])


def build_parameters_df(result: dict[str, Any]) -> pd.DataFrame:
    rows = extract_parameter_rows(result)
    if not rows.names:
        return _not_available_df()
    return pd.DataFrame(
        {
            "parameter": rows.names,
            "estimate": rows.estimate,
            "SE": rows.se,
            "%err": rows.perc_err,
            "CI_low": rows.ci_low,
            "CI_high": rows.ci_high,
            "fixed": rows.fixed,
            "scale": rows.scale,
        }
    )


def build_model_df(result: dict[str, Any], *, config: dict[str, Any] | None = None) -> pd.DataFrame:
    model = result.get("model")
    mechanism = getattr(model, "mechanism", None)
    row_data: list[tuple[str, Any]] = []
    if mechanism is not None:
        species = list(getattr(mechanism, "species", []) or [])
        fixed = sorted(getattr(mechanism, "fixed", set()) or [])
        reactions = list(getattr(mechanism, "reactions", []) or [])
        temp_models = getattr(mechanism, "temp_models", {}) or {}
        row_data.append(("species", ", ".join(species)))
        row_data.append(("fixed_species", ", ".join(fixed) if fixed else "none"))
        row_data.append(("reactions_count", len(reactions)))
        for idx, reaction in enumerate(reactions, start=1):
            row_data.append((f"reaction_{idx}", _reaction_to_text(reaction)))
        if temp_models:
            row_data.append(("temperature_models", json.dumps(temp_models, default=str)))
    else:
        row_data.append(("model", "Not available"))

    row_data.append(("dynamic_species", ", ".join(result.get("dynamic_species") or [])))
    row_data.append(("param_names", ", ".join(result.get("param_names") or [])))
    row_data.append(("log_params", ", ".join(result.get("log_params") or [])))

    if config:
        row_data.append(("method", config.get("method")))
        row_data.append(("backend", config.get("backend")))
        row_data.append(("nnls", bool(config.get("nnls", False))))
        row_data.append(("multi_start_runs", config.get("multi_start_runs")))
        row_data.append(("auto_parameter_errors", bool(config.get("auto_parameter_errors", True))))
        row_data.append(("error_condition_threshold", config.get("error_condition_threshold", 1.0e12)))
        row_data.append(("show_stability_diagnostics", bool(config.get("show_stability_diagnostics", False))))
        bounds = config.get("bounds")
        if bounds is not None:
            try:
                lower = np.asarray(bounds[0], dtype=float).tolist()
                upper = np.asarray(bounds[1], dtype=float).tolist()
                row_data.append(("bounds_lower", json.dumps(lower)))
                row_data.append(("bounds_upper", json.dumps(upper)))
            except Exception:
                row_data.append(("bounds", str(bounds)))

    datasets = _iter_fit_datasets(result)
    for ds_idx, ds in enumerate(datasets, start=1):
        y0 = ds.y0 or {}
        fixed_conc = ds.fixed_conc or {}
        row_data.append((f"dataset_{ds_idx}_name", ds.name))
        row_data.append((f"dataset_{ds_idx}_technique", ds.technique))
        row_data.append((f"dataset_{ds_idx}_y0", json.dumps(y0, default=float)))
        row_data.append((f"dataset_{ds_idx}_fixed_conc", json.dumps(fixed_conc, default=float)))

    return pd.DataFrame(row_data, columns=["key", "value"])


def _as_2d_float(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return None


def _prefix_for_dataset(ds: KineticsFitDataset, idx: int, total: int) -> str:
    if total <= 1:
        return ""
    name = str(ds.name or f"dataset_{idx + 1}").strip() or f"dataset_{idx + 1}"
    return f"{name} | "


def _sanitize_label(raw: Any, fallback: str) -> str:
    label = str(raw).strip()
    if not label:
        return fallback
    clean = label.replace(" ", "_")
    clean = clean.replace("/", "_").replace("\\", "_")
    return clean


def _channel_labels(ds: KineticsFitDataset, n_channels: int) -> list[str]:
    labels = [str(v) for v in (ds.channel_labels or [])]
    if len(labels) < n_channels and ds.x is not None:
        x_vals = np.asarray(ds.x, dtype=float).ravel()
        labels.extend(f"{float(x):g}" for x in x_vals[len(labels) : n_channels])
    if len(labels) < n_channels:
        labels.extend(f"ch{idx + 1}" for idx in range(len(labels), n_channels))
    labels = labels[:n_channels]
    used: dict[str, int] = {}
    out: list[str] = []
    for idx, label in enumerate(labels):
        key = _sanitize_label(label, f"ch{idx + 1}")
        count = used.get(key, 0)
        used[key] = count + 1
        out.append(key if count == 0 else f"{key}_{count + 1}")
    return out


def _species_labels(result: dict[str, Any], n_species: int) -> list[str]:
    labels = [str(v) for v in (result.get("dynamic_species") or [])]
    if len(labels) < n_species:
        labels.extend(f"sp_{idx + 1}" for idx in range(len(labels), n_species))
    labels = labels[:n_species]
    used: dict[str, int] = {}
    out: list[str] = []
    for idx, label in enumerate(labels):
        key = _sanitize_label(label, f"sp_{idx + 1}")
        count = used.get(key, 0)
        used[key] = count + 1
        out.append(key if count == 0 else f"{key}_{count + 1}")
    return out


def _trim_time_matrix(t: np.ndarray, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows = min(t.size, mat.shape[0])
    if n_rows <= 0:
        return np.asarray([], dtype=float), mat[:0, :]
    return t[:n_rows], mat[:n_rows, :]


def _outer_merge_on_key(frames: list[pd.DataFrame], key: str) -> pd.DataFrame | None:
    if not frames:
        return None
    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on=key, how="outer", sort=False)
    try:
        merged = merged.sort_values(by=key, kind="stable")
    except Exception:
        pass
    return merged.reset_index(drop=True)


def build_raw_df(result: dict[str, Any]) -> pd.DataFrame:
    datasets = _iter_fit_datasets(result)
    frames: list[pd.DataFrame] = []
    for idx, ds in enumerate(datasets):
        D = _as_2d_float(ds.D)
        t = np.asarray(ds.t, dtype=float).ravel()
        if D is None or t.size == 0:
            continue
        t_fit, D_fit = _trim_time_matrix(t, D)
        if t_fit.size == 0:
            continue
        prefix = _prefix_for_dataset(ds, idx, len(datasets))
        labels = _channel_labels(ds, D_fit.shape[1])
        data: dict[str, Any] = {"time": t_fit}
        for i_ch, label in enumerate(labels):
            data[f"{prefix}obs_{label}"] = D_fit[:, i_ch]
        frames.append(pd.DataFrame(data))

    merged = _outer_merge_on_key(frames, "time")
    if merged is None or merged.empty:
        return _not_available_df()
    return merged


def build_fit_df(result: dict[str, Any]) -> pd.DataFrame:
    datasets = _iter_fit_datasets(result)
    frames: list[pd.DataFrame] = []
    has_fit = False
    for idx, ds in enumerate(datasets):
        D = _as_2d_float(ds.D)
        D_hat = _as_2d_float(ds.fit_D_hat)
        t = np.asarray(ds.t, dtype=float).ravel()
        if D is None or D_hat is None or t.size == 0:
            continue
        n_rows = min(t.size, D.shape[0], D_hat.shape[0])
        n_channels = min(D.shape[1], D_hat.shape[1])
        if n_rows <= 0 or n_channels <= 0:
            continue
        t_fit = t[:n_rows]
        D_fit = D[:n_rows, :n_channels]
        D_hat_fit = D_hat[:n_rows, :n_channels]
        resid = _as_2d_float(ds.fit_residuals)
        if resid is not None:
            resid_fit = resid[:n_rows, :n_channels]
        else:
            resid_fit = D_fit - D_hat_fit

        prefix = _prefix_for_dataset(ds, idx, len(datasets))
        labels = _channel_labels(ds, n_channels)
        data: dict[str, Any] = {"time": t_fit}
        for i_ch, label in enumerate(labels):
            data[f"{prefix}obs_{label}"] = D_fit[:, i_ch]
            data[f"{prefix}fit_{label}"] = D_hat_fit[:, i_ch]
            data[f"{prefix}res_{label}"] = resid_fit[:, i_ch]
        frames.append(pd.DataFrame(data))
        has_fit = True

    merged = _outer_merge_on_key(frames, "time")
    if merged is None or merged.empty or not has_fit:
        return _not_available_df()
    return merged


def build_concentrations_df(result: dict[str, Any]) -> pd.DataFrame:
    datasets = _iter_fit_datasets(result)
    frames: list[pd.DataFrame] = []
    for idx, ds in enumerate(datasets):
        C = _as_2d_float(ds.fit_C)
        t = np.asarray(ds.t, dtype=float).ravel()
        if C is None or t.size == 0:
            continue
        t_fit, C_fit = _trim_time_matrix(t, C)
        if t_fit.size == 0:
            continue
        prefix = _prefix_for_dataset(ds, idx, len(datasets))
        species = _species_labels(result, C_fit.shape[1])
        data: dict[str, Any] = {"time": t_fit}
        for i_sp, sp in enumerate(species):
            data[f"{prefix}{sp}"] = C_fit[:, i_sp]
        frames.append(pd.DataFrame(data))

    merged = _outer_merge_on_key(frames, "time")
    if merged is None or merged.empty:
        return _not_available_df()
    return merged


def build_absorptivity_df(result: dict[str, Any]) -> pd.DataFrame:
    datasets = _iter_fit_datasets(result)
    frames: list[pd.DataFrame] = []
    for idx, ds in enumerate(datasets):
        A = _as_2d_float(ds.fit_A)
        if A is None:
            continue
        x = np.asarray(ds.x, dtype=float).ravel() if ds.x is not None else np.arange(A.shape[1], dtype=float)
        if x.size == 0:
            continue
        n_axis = min(x.size, A.shape[1])
        if n_axis <= 0:
            continue
        axis_vals = x[:n_axis]
        prefix = _prefix_for_dataset(ds, idx, len(datasets))
        species = _species_labels(result, A.shape[0])
        data: dict[str, Any] = {"wavelength": axis_vals}
        for i_sp, sp in enumerate(species):
            data[f"{prefix}{sp}"] = A[i_sp, :n_axis]
        frames.append(pd.DataFrame(data))

    merged = _outer_merge_on_key(frames, "wavelength")
    if merged is None or merged.empty:
        return _not_available_df()
    return merged


def build_absorbance_df(result: dict[str, Any]) -> pd.DataFrame:
    datasets = _iter_fit_datasets(result)
    frames: list[pd.DataFrame] = []
    has_fit = False
    for idx, ds in enumerate(datasets):
        D = _as_2d_float(ds.D)
        D_hat = _as_2d_float(ds.fit_D_hat)
        t = np.asarray(ds.t, dtype=float).ravel()
        if D is None or D_hat is None or t.size == 0:
            continue
        n_rows = min(t.size, D.shape[0], D_hat.shape[0])
        n_channels = min(D.shape[1], D_hat.shape[1])
        if n_rows <= 0 or n_channels <= 0:
            continue
        t_fit = t[:n_rows]
        D_fit = D[:n_rows, :n_channels]
        D_hat_fit = D_hat[:n_rows, :n_channels]
        resid = _as_2d_float(ds.fit_residuals)
        if resid is not None:
            resid_fit = resid[:n_rows, :n_channels]
        else:
            resid_fit = D_fit - D_hat_fit

        prefix = _prefix_for_dataset(ds, idx, len(datasets))
        labels = _channel_labels(ds, n_channels)
        data: dict[str, Any] = {"time": t_fit}
        for i_ch, label in enumerate(labels):
            data[f"{prefix}D_{label}"] = D_fit[:, i_ch]
            data[f"{prefix}D_hat_{label}"] = D_hat_fit[:, i_ch]
            data[f"{prefix}residual_{label}"] = resid_fit[:, i_ch]
        frames.append(pd.DataFrame(data))
        has_fit = True

    merged = _outer_merge_on_key(frames, "time")
    if merged is None or merged.empty or not has_fit:
        return _not_available_df()
    return merged


def build_diagnostics_df(result: dict[str, Any]) -> pd.DataFrame:
    diagnostics = result.get("diagnostics") or {}
    warnings = list(diagnostics.get("warnings") or [])
    row_data: list[dict[str, Any]] = []
    row_data.append({"type": "convergence", "key": "success", "value": bool(result.get("success", True))})
    row_data.append({"type": "convergence", "key": "optimizer_method", "value": result.get("optimizer_method")})
    row_data.append({"type": "convergence", "key": "optimizer_backend", "value": result.get("optimizer_backend")})
    row_data.append({"type": "convergence", "key": "nfev", "value": result.get("nfev")})
    row_data.append({"type": "convergence", "key": "njev", "value": result.get("njev")})
    row_data.append({"type": "stability", "key": "status", "value": diagnostics.get("stability_status")})
    row_data.append({"type": "stability", "key": "summary", "value": diagnostics.get("diag_summary")})
    if diagnostics.get("diag_full"):
        row_data.append({"type": "stability", "key": "full", "value": diagnostics.get("diag_full")})
    if diagnostics.get("condition_number") is not None:
        row_data.append(
            {
                "type": "stability",
                "key": "condition_number",
                "value": diagnostics.get("condition_number"),
            }
        )
    extra_keys = [
        "method_requested",
        "method_used",
        "backend",
        "nnls",
        "weighted_fit",
        "optimizer_success",
        "optimizer_status",
        "optimizer_message",
    ]
    for key in extra_keys:
        if key in diagnostics:
            row_data.append({"type": "run", "key": key, "value": diagnostics.get(key)})
    if not warnings:
        row_data.append({"type": "warning", "key": "none", "value": "No warnings"})
    else:
        for idx, message in enumerate(warnings, start=1):
            row_data.append({"type": "warning", "key": f"warning_{idx}", "value": str(message)})
    return pd.DataFrame(row_data, columns=["type", "key", "value"])


def build_gui_parameter_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = extract_parameter_rows(result)
    out: list[dict[str, Any]] = []
    for i, name in enumerate(rows.names):
        out.append(
            {
                "Parameter": str(name),
                "Estimate": float(rows.estimate[i]) if i < rows.estimate.size else np.nan,
                "SE": float(rows.se[i]) if i < rows.se.size else np.nan,
                "%err": float(rows.perc_err[i]) if i < rows.perc_err.size else np.nan,
                "CI 2.5": float(rows.ci_low[i]) if i < rows.ci_low.size else np.nan,
                "CI 97.5": float(rows.ci_high[i]) if i < rows.ci_high.size else np.nan,
            }
        )
    return out


def write_kinetics_results_xlsx(
    path: str,
    *,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Summary", build_summary_df(result)),
        ("Parameters", build_parameters_df(result)),
        ("Model", build_model_df(result, config=config)),
        ("RawData", build_raw_df(result)),
        ("FitData", build_fit_df(result)),
        ("Concentrations_Ct", build_concentrations_df(result)),
        ("Absorptivity_Profiles", build_absorptivity_df(result)),
        ("Absorbance_Profiles", build_absorbance_df(result)),
        ("Diagnostics", build_diagnostics_df(result)),
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            out_df = df if isinstance(df, pd.DataFrame) else _not_available_df()
            if out_df.empty:
                out_df = _not_available_df()
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)
