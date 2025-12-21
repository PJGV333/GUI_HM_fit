from __future__ import annotations

import re
from typing import Any


def build_nmr_plot_sources(result: dict[str, Any]) -> dict[str, Any]:
    plot_data = result.get("plotData", {}).get("nmr") or {}
    if not isinstance(plot_data, dict):
        return {}

    export_data = result.get("export_data") or {}
    if not isinstance(export_data, dict):
        export_data = {}

    column_names = [str(c) for c in (export_data.get("column_names") or []) if str(c)]
    c_tot = export_data.get("C_T") or []

    axis_vectors: dict[str, list[Any]] = {}
    if column_names and isinstance(c_tot, list) and c_tot and isinstance(c_tot[0], list):
        for idx, col in enumerate(column_names):
            axis_vectors[col] = [row[idx] if isinstance(row, list) and idx < len(row) else None for row in c_tot]

    x_label = str((plot_data.get("nmr_shifts_fit") or {}).get("xLabel") or "")
    default_id = ""
    if x_label:
        match = re.search(r"\[([^\]]+)\]", x_label)
        if match:
            label = match.group(1)
            if label in column_names:
                default_id = label
    if not default_id and column_names:
        default_id = column_names[0]

    axis_options: list[dict[str, str]] = []
    if column_names:
        ordered = column_names
        if default_id in column_names:
            ordered = [default_id] + [c for c in column_names if c != default_id]
        for col in ordered:
            label = x_label if col == default_id and x_label else f"[{col}] Total (M)"
            axis_options.append({"id": col, "label": label})

    if not axis_vectors:
        axis_options = []

    if not axis_options:
        default_x = list((plot_data.get("nmr_shifts_fit") or {}).get("x") or [])
        default_label = x_label or "Concentration"
        if default_x:
            axis_vectors = {"titrant_total": default_x}
            axis_options = [{"id": "titrant_total", "label": default_label}]
            default_id = "titrant_total"

    merged = dict(plot_data)
    for key in ("nmr_shifts_fit", "nmr_residuals", "nmr_species_distribution"):
        entry = merged.get(key)
        if not isinstance(entry, dict):
            entry = {}
        else:
            entry = dict(entry)
        if axis_vectors and axis_options:
            entry["axisVectors"] = axis_vectors
            entry["axisOptions"] = axis_options
            if default_id:
                entry["x_default_id"] = default_id
        merged[key] = entry

    return merged
