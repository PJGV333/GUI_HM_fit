from __future__ import annotations

from typing import Any, Callable

from hmfit_gui_qt.plots.plot_registry import (
    PlotBuildResult,
    PlotControls,
    PlotDescriptor,
    PlotSeries,
    axis_options,
    build_species_distribution,
    export_series_csv,
    species_options,
)


def _build_fit_overlay(
    title: str,
    data: dict[str, Any],
    _controls: PlotControls,
) -> PlotBuildResult:
    plot_mode = str(data.get("plotMode") or "spectra")
    nm = data.get("nm") or []
    y_exp = data.get("Yexp") or []
    y_fit = data.get("Yfit") or []
    x_t = data.get("xTitrant") or []

    series: list[PlotSeries] = []
    if plot_mode == "isotherms":
        k = min(len(nm), 10)
        for i in range(k):
            wl = nm[i]
            y_obs = y_exp[i] if i < len(y_exp) else []
            y_fit_i = y_fit[i] if i < len(y_fit) else []
            series.append(
                PlotSeries(
                    x=x_t,
                    y=y_obs,
                    label=f"{wl} obs",
                    mode="markers",
                    style={"markersize": 7},
                )
            )
            series.append(
                PlotSeries(
                    x=x_t,
                    y=y_fit_i,
                    label=f"{wl} fit",
                    mode="lines",
                    style={"linewidth": 2.0, "linestyle": ":"},
                )
            )
        x_label = str(data.get("xLabel") or "[X]")
    else:
        n_steps = 0
        if y_exp and isinstance(y_exp[0], list):
            n_steps = len(y_exp[0])
        for j in range(n_steps):
            y_obs = [row[j] if row is not None and len(row) > j else None for row in y_exp]
            y_fit_j = [row[j] if row is not None and len(row) > j else None for row in y_fit]
            series.append(
                PlotSeries(
                    x=nm,
                    y=y_obs,
                    label=f"exp {j + 1}",
                    mode="lines",
                    style={"linewidth": 1.0, "color": "black", "alpha": 0.35},
                    show_legend=j == 0,
                )
            )
            series.append(
                PlotSeries(
                    x=nm,
                    y=y_fit_j,
                    label=f"fit {j + 1}",
                    mode="lines",
                    style={"linewidth": 1.5, "linestyle": ":", "color": "#ef4444", "alpha": 0.8},
                    show_legend=j == 0,
                )
            )
        x_label = "\u03bb (nm)"

    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="Y observed (u. a.)")


def _build_molar_absorptivities(
    title: str,
    data: dict[str, Any],
    _controls: PlotControls,
) -> PlotBuildResult:
    nm = data.get("nm") or []
    a_matrix = data.get("A") or []
    species_opts = data.get("speciesOptions") or []
    n_species = len(a_matrix[0]) if a_matrix and isinstance(a_matrix[0], list) else 0
    is_spectrum = isinstance(nm, list) and len(nm) > 10

    series: list[PlotSeries] = []
    for s in range(n_species):
        y = [row[s] if row is not None and len(row) > s else None for row in a_matrix]
        label = str(species_opts[s].get("label") if s < len(species_opts) else f"sp{s + 1}")
        series.append(
            PlotSeries(
                x=nm,
                y=y,
                label=label,
                mode="lines" if is_spectrum else "lines+markers",
                style={"linewidth": 2.0, "markersize": 6 if not is_spectrum else 0},
            )
        )

    return PlotBuildResult(series=series, title=title, x_label="\u03bb (nm)", y_label="Epsilon (u. a.)")


def _build_efa_eigenvalues(
    title: str,
    data: dict[str, Any],
    _controls: PlotControls,
) -> PlotBuildResult:
    ev = data.get("eigenvalues") or []
    x = list(range(1, len(ev) + 1))
    y = [None if v is None or v <= 0 else _safe_log10(v) for v in ev]
    series = [
        PlotSeries(
            x=x,
            y=y,
            label="log10(EV)",
            mode="lines+markers",
            style={"markersize": 7},
        )
    ]
    return PlotBuildResult(series=series, title=title, x_label="# eigenvalues", y_label="log10(EV)")


def _resolve_spec_axis(data: dict[str, Any], controls: PlotControls) -> tuple[list[Any], str]:
    axis_vectors = data.get("axisVectors") or {}
    axis_options = data.get("axisOptions") or []
    axis_id = str(controls.dist_x_axis_id or "")
    x_label = str(data.get("xLabel") or "[X]")

    x = list(axis_vectors.get(axis_id) or [])
    if not x and axis_vectors:
        default_id = str(data.get("x_default_id") or "")
        if default_id and default_id in axis_vectors:
            axis_id = default_id
        else:
            axis_id = next(iter(axis_vectors.keys()))
        x = list(axis_vectors.get(axis_id) or [])

    for opt in axis_options:
        if str(opt.get("id") or "") == axis_id:
            x_label = str(opt.get("label") or x_label)
            break

    return x, x_label


def _build_efa_components(
    title: str,
    data: dict[str, Any],
    _controls: PlotControls,
) -> PlotBuildResult:
    x, x_label = _resolve_spec_axis(data, _controls)
    if not x:
        x = data.get("xTitrant") or []
    fwd = data.get("efaForward") or []
    bwd = data.get("efaBackward") or []
    n_comp = len(fwd[0]) if fwd and isinstance(fwd[0], list) else 0

    series: list[PlotSeries] = []
    shown_forward = False
    shown_backward = False
    for c in range(n_comp):
        y_f = [_safe_log10(row[c]) if row is not None and len(row) > c else None for row in fwd]
        y_b = [_safe_log10(row[c]) if row is not None and len(row) > c else None for row in bwd]
        series.append(
            PlotSeries(
                x=x,
                y=y_f,
                label="Forward",
                mode="lines",
                style={"linewidth": 2.0, "color": "#1f77b4"},
                show_legend=not shown_forward,
            )
        )
        shown_forward = True
        series.append(
            PlotSeries(
                x=x,
                y=y_b,
                label="Backward",
                mode="lines",
                style={"linewidth": 2.0, "linestyle": ":", "color": "#ff7f0e"},
                show_legend=not shown_backward,
            )
        )
        shown_backward = True

    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="log10(EV)")


def _safe_log10(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if v <= 0:
        return None
    import math

    return math.log10(v)


def build_spectroscopy_registry(
    selected_columns_getter: Callable[[], list[str]] | None = None,
) -> dict[str, PlotDescriptor]:
    def _filtered_axes(data: dict[str, Any]) -> list[tuple[str, str]]:
        axes = axis_options(data)
        if selected_columns_getter is None:
            return axes
        selected = {str(c) for c in (selected_columns_getter() or []) if str(c).strip()}
        if not selected:
            return []
        return [(axis_id, label) for axis_id, label in axes if axis_id in selected]

    return {
        "spec_species_distribution": PlotDescriptor(
            id="spec_species_distribution",
            display_name="Species distribution",
            supports_axes=True,
            supports_series=True,
            supports_vary=False,
            series_selection_key="dist_y_selected",
            get_available_axes=_filtered_axes,
            get_available_series=species_options,
            get_available_vary=lambda _data: [],
            build=build_species_distribution,
            export_csv=export_series_csv,
            axis_selection_key="dist_x_axis_id",
        ),
        "spec_fit_overlay": PlotDescriptor(
            id="spec_fit_overlay",
            display_name="Experimental vs fitted spectra",
            supports_axes=False,
            supports_series=False,
            supports_vary=False,
            series_selection_key=None,
            get_available_axes=lambda _data: [],
            get_available_series=lambda _data: [],
            get_available_vary=lambda _data: [],
            build=_build_fit_overlay,
            export_csv=lambda _data, _controls, _build: None,
        ),
        "spec_molar_absorptivities": PlotDescriptor(
            id="spec_molar_absorptivities",
            display_name="Molar absorptivities",
            supports_axes=False,
            supports_series=False,
            supports_vary=False,
            series_selection_key=None,
            get_available_axes=lambda _data: [],
            get_available_series=lambda _data: [],
            get_available_vary=lambda _data: [],
            build=_build_molar_absorptivities,
            export_csv=lambda _data, _controls, _build: None,
        ),
        "spec_efa_eigenvalues": PlotDescriptor(
            id="spec_efa_eigenvalues",
            display_name="EFA eigenvalues",
            supports_axes=False,
            supports_series=False,
            supports_vary=False,
            series_selection_key=None,
            get_available_axes=lambda _data: [],
            get_available_series=lambda _data: [],
            get_available_vary=lambda _data: [],
            build=_build_efa_eigenvalues,
            export_csv=lambda _data, _controls, _build: None,
        ),
        "spec_efa_components": PlotDescriptor(
            id="spec_efa_components",
            display_name="EFA forward/backward",
            supports_axes=True,
            supports_series=False,
            supports_vary=False,
            series_selection_key=None,
            get_available_axes=_filtered_axes,
            get_available_series=lambda _data: [],
            get_available_vary=lambda _data: [],
            build=_build_efa_components,
            export_csv=lambda _data, _controls, _build: None,
            axis_selection_key="dist_x_axis_id",
        ),
    }
