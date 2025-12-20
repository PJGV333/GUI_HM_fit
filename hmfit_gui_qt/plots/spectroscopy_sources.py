from __future__ import annotations

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
    }

    merged = dict(spec_data)
    merged.update(
        {
            "spec_fit_overlay": spec_fit_overlay,
            "spec_molar_absorptivities": spec_molar_abs,
            "spec_efa_eigenvalues": spec_efa_eig,
            "spec_efa_components": spec_efa_comp,
        }
    )
    return merged
