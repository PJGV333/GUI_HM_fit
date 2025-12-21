from __future__ import annotations

import re
from typing import Any


def build_spectroscopy_plot_sources(result: dict[str, Any]) -> dict[str, Any]:
    spec_data = result.get("plotData", {}).get("spec") or {}
    if not isinstance(spec_data, dict):
        spec_data = {}

    plot_data = result.get("plot_data") or {}
    if not isinstance(plot_data, dict):
        plot_data = {}
    numerics = plot_data.get("numerics") or {}
    if not isinstance(numerics, dict):
        numerics = {}
    plot_meta = plot_data.get("plot_meta") or {}
    if not isinstance(plot_meta, dict):
        plot_meta = {}
    export_data = result.get("export_data") or {}
    if not isinstance(export_data, dict):
        export_data = {}
    dist = spec_data.get("spec_species_distribution") or {}

    axis_vectors = dist.get("axisVectors") or {}
    x_titrant = axis_vectors.get("titrant_total") or []
    x_label = "[X]"
    for opt in dist.get("axisOptions") or []:
        if str(opt.get("id") or "") == "titrant_total":
            x_label = str(opt.get("label") or x_label)
            break

    axes_meta = plot_meta.get("axes") or {}
    if not isinstance(axes_meta, dict):
        axes_meta = {}
    ct_axes = []
    for axis in axes_meta.values():
        if not isinstance(axis, dict):
            continue
        if str(axis.get("values_key") or "") != "Ct":
            continue
        col_idx = axis.get("column")
        if col_idx is None:
            continue
        ct_axes.append(axis)
    ct_axes.sort(key=lambda a: int(a.get("column") or 0))
    column_names = [str(a.get("label") or "").strip() for a in ct_axes if str(a.get("label") or "").strip()]
    if not column_names:
        column_names = [str(c) for c in (export_data.get("column_names") or []) if str(c).strip()]

    c_tot = numerics.get("Ct") or export_data.get("C_T") or []
    col_axis_vectors: dict[str, list[Any]] = {}
    if column_names and isinstance(c_tot, list) and c_tot and isinstance(c_tot[0], list):
        for idx, col in enumerate(column_names):
            col_axis_vectors[col] = [
                row[idx] if isinstance(row, list) and idx < len(row) else None for row in c_tot
            ]

    default_id = ""
    if x_label:
        match = re.search(r"\[([^\]]+)\]", x_label)
        if match:
            label = match.group(1)
            if label in column_names:
                default_id = label
    if not default_id and column_names:
        default_id = column_names[0]

    col_axis_options: list[dict[str, str]] = []
    if col_axis_vectors:
        ordered = column_names
        if default_id in column_names:
            ordered = [default_id] + [c for c in column_names if c != default_id]
        for col in ordered:
            label = x_label if col == default_id and x_label else f"[{col}] total"
            col_axis_options.append({"id": col, "label": label})

    if not x_titrant and default_id and col_axis_vectors:
        x_titrant = col_axis_vectors.get(default_id) or []
    if not x_label and default_id:
        x_label = f"[{default_id}] total"

    nm = numerics.get("nm") or export_data.get("nm") or []
    y_exp = numerics.get("Y_exp") or export_data.get("Y") or []
    y_fit = numerics.get("Y_fit") or export_data.get("yfit") or []

    def _transpose2d(values: list[list[Any]]) -> list[list[Any]]:
        return [list(row) for row in zip(*values)]

    if (
        isinstance(nm, list)
        and nm
        and isinstance(y_exp, list)
        and y_exp
        and isinstance(y_exp[0], list)
    ):
        if len(y_exp) != len(nm) and len(y_exp[0]) == len(nm):
            y_exp = _transpose2d(y_exp)
        if isinstance(y_fit, list) and y_fit and isinstance(y_fit[0], list):
            if len(y_fit) != len(nm) and len(y_fit[0]) == len(nm):
                y_fit = _transpose2d(y_fit)

    a_matrix = export_data.get("A") or []
    a_nm = export_data.get("A_index") or export_data.get("nm") or nm
    if (
        isinstance(a_nm, list)
        and a_nm
        and isinstance(a_matrix, list)
        and a_matrix
        and isinstance(a_matrix[0], list)
    ):
        if len(a_matrix) != len(a_nm) and len(a_matrix[0]) == len(a_nm):
            a_matrix = _transpose2d(a_matrix)

    spec_fit_overlay = {
        "plotMode": result.get("plot_mode") or "spectra",
        "xTitrant": x_titrant,
        "xLabel": x_label,
        "nm": nm,
        "Yexp": y_exp,
        "Yfit": y_fit,
    }
    spec_molar_abs = {
        "nm": a_nm,
        "A": a_matrix,
        "speciesOptions": dist.get("speciesOptions") or [],
    }
    spec_efa_eig = {
        "eigenvalues": numerics.get("eigenvalues") or [],
    }
    spec_efa_comp = {
        "xTitrant": x_titrant,
        "xLabel": x_label,
        "efaForward": numerics.get("efa_forward") or [],
        "efaBackward": numerics.get("efa_backward") or [],
        "x_axis_empty_label": "Select columns in Column names.",
    }
    spec_efa_comp["axisOptions"] = col_axis_options
    spec_efa_comp["axisVectors"] = col_axis_vectors
    if default_id:
        spec_efa_comp["x_default_id"] = default_id
        spec_efa_comp["x_default"] = col_axis_vectors.get(default_id) or []

    merged = dict(spec_data)
    dist_payload = dict(dist) if isinstance(dist, dict) else {}
    dist_payload["axisOptions"] = col_axis_options
    dist_payload["axisVectors"] = col_axis_vectors
    if default_id:
        dist_payload["x_default_id"] = default_id
        dist_payload["x_default"] = col_axis_vectors.get(default_id) or []
    dist_payload["x_axis_empty_label"] = "Select columns in Column names."
    merged.update(
        {
            "spec_fit_overlay": spec_fit_overlay,
            "spec_species_distribution": dist_payload,
            "spec_molar_absorptivities": spec_molar_abs,
            "spec_efa_eigenvalues": spec_efa_eig,
            "spec_efa_components": spec_efa_comp,
        }
    )
    return merged
