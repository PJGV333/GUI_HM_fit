from __future__ import annotations

from typing import Any

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


def _create_gap_indices(x_raw: list[float]) -> set[int]:
    gap_indices: set[int] = set()
    for i in range(1, len(x_raw)):
        if x_raw[i] < x_raw[i - 1]:
            gap_indices.add(i)
    return gap_indices


def _create_gapped_arrays(x_arr: list[float], y_arr: list[float], gap_indices: set[int]) -> tuple[list[Any], list[Any]]:
    if not gap_indices:
        return list(x_arr), list(y_arr)
    new_x: list[Any] = []
    new_y: list[Any] = []
    for i, x in enumerate(x_arr):
        if i in gap_indices:
            new_x.append(None)
            new_y.append(None)
        new_x.append(x)
        new_y.append(y_arr[i] if i < len(y_arr) else None)
    return new_x, new_y


def _build_nmr_shifts_fit(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
) -> PlotBuildResult:
    x_raw = data.get("x") or []
    signal_options = data.get("signalOptions") or []
    signals = data.get("signals") or {}

    if not controls.nmr_signals_selected:
        controls.nmr_signals_selected = {str(opt.get("id") or "") for opt in signal_options if opt.get("id")}

    gap_indices = _create_gap_indices(x_raw)
    series: list[PlotSeries] = []

    for opt in signal_options:
        sig_id = str(opt.get("id") or "")
        if not sig_id or sig_id not in controls.nmr_signals_selected:
            continue
        label = str(opt.get("label") or sig_id)
        sig = signals.get(sig_id) or {}
        obs = sig.get("obs") or []
        fit = sig.get("fit") or []
        x_obs, y_obs = _create_gapped_arrays(x_raw, obs, gap_indices)
        x_fit, y_fit = _create_gapped_arrays(x_raw, fit, gap_indices)
        series.append(
            PlotSeries(
                x=x_obs,
                y=y_obs,
                label=f"{label} obs",
                mode="markers",
                style={"markersize": 8},
            )
        )
        series.append(
            PlotSeries(
                x=x_fit,
                y=y_fit,
                label=f"{label} fit",
                mode="lines",
                style={"linewidth": 2.0, "linestyle": ":"},
            )
        )

    x_label = str(data.get("xLabel") or "Concentration")
    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="\u0394\u03b4 (ppm)")


def _build_nmr_residuals(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
) -> PlotBuildResult:
    x = data.get("x") or []
    signal_options = data.get("signalOptions") or []
    signals = data.get("signals") or {}

    if not controls.nmr_resid_selected:
        controls.nmr_resid_selected = {str(opt.get("id") or "") for opt in signal_options if opt.get("id")}

    series: list[PlotSeries] = []
    for opt in signal_options:
        sig_id = str(opt.get("id") or "")
        if not sig_id or sig_id not in controls.nmr_resid_selected:
            continue
        label = str(opt.get("label") or sig_id)
        sig = signals.get(sig_id) or {}
        resid = sig.get("resid") or []
        series.append(
            PlotSeries(
                x=x,
                y=resid,
                label=f"{label} resid",
                mode="markers",
                style={"markersize": 6},
            )
        )

    x_label = str(data.get("xLabel") or "Concentration")
    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="Residuals (ppm)")


def _export_nmr_shifts_csv(
    data: dict[str, Any],
    controls: PlotControls,
    build: PlotBuildResult,
) -> tuple[list[str], list[list[Any]]] | None:
    x = data.get("x") or []
    signal_options = data.get("signalOptions") or []
    signals = data.get("signals") or {}
    if not x:
        return None

    selected_set = set(controls.nmr_signals_selected)
    if not selected_set:
        selected_set = {str(opt.get("id") or "") for opt in signal_options if opt.get("id")}

    ordered_selected: list[str] = []
    for opt in signal_options:
        sig_id = str(opt.get("id") or "")
        if sig_id and sig_id in selected_set:
            ordered_selected.append(sig_id)

    headers = ["X"]
    if build.series and len(build.series) == 2 * len(ordered_selected):
        headers.extend([s.label for s in build.series])
    else:
        for opt in signal_options:
            sig_id = str(opt.get("id") or "")
            if sig_id in selected_set:
                label = str(opt.get("label") or sig_id)
                headers.extend([f"{label}_obs", f"{label}_fit"])

    rows: list[list[Any]] = []
    for i, xv in enumerate(x):
        row: list[Any] = [xv]
        for sig_id in ordered_selected:
            sig = signals.get(sig_id) or {}
            obs = sig.get("obs") or []
            fit = sig.get("fit") or []
            row.append(obs[i] if i < len(obs) else "")
            row.append(fit[i] if i < len(fit) else "")
        rows.append(row)
    return headers, rows


def _export_nmr_residuals_csv(
    data: dict[str, Any],
    controls: PlotControls,
    build: PlotBuildResult,
) -> tuple[list[str], list[list[Any]]] | None:
    x = data.get("x") or []
    signal_options = data.get("signalOptions") or []
    signals = data.get("signals") or {}
    if not x:
        return None

    selected_set = set(controls.nmr_resid_selected)
    if not selected_set:
        selected_set = {str(opt.get("id") or "") for opt in signal_options if opt.get("id")}

    ordered_selected: list[str] = []
    for opt in signal_options:
        sig_id = str(opt.get("id") or "")
        if sig_id and sig_id in selected_set:
            ordered_selected.append(sig_id)

    headers = ["X"]
    if build.series and len(build.series) == len(ordered_selected):
        headers.extend([s.label for s in build.series])
    else:
        for opt in signal_options:
            sig_id = str(opt.get("id") or "")
            if sig_id in selected_set:
                label = str(opt.get("label") or sig_id)
                headers.append(f"{label}_resid")

    rows: list[list[Any]] = []
    for i, xv in enumerate(x):
        row: list[Any] = [xv]
        for sig_id in ordered_selected:
            sig = signals.get(sig_id) or {}
            resid = sig.get("resid") or []
            row.append(resid[i] if i < len(resid) else "")
        rows.append(row)
    return headers, rows


def build_nmr_registry() -> dict[str, PlotDescriptor]:
    return {
        "nmr_shifts_fit": PlotDescriptor(
            id="nmr_shifts_fit",
            display_name="Chemical shifts fit",
            supports_axes=False,
            supports_series=True,
            supports_vary=False,
            series_selection_key="nmr_signals_selected",
            get_available_axes=lambda _data: [],
            get_available_series=lambda data: [
                (str(opt.get("id") or ""), str(opt.get("label") or opt.get("id") or ""))
                for opt in (data.get("signalOptions") or [])
                if opt.get("id")
            ],
            get_available_vary=lambda _data: [],
            build=_build_nmr_shifts_fit,
            export_csv=_export_nmr_shifts_csv,
        ),
        "nmr_species_distribution": PlotDescriptor(
            id="nmr_species_distribution",
            display_name="Species distribution",
            supports_axes=True,
            supports_series=True,
            supports_vary=False,
            series_selection_key="dist_y_selected",
            get_available_axes=axis_options,
            get_available_series=species_options,
            get_available_vary=lambda _data: [],
            build=build_species_distribution,
            export_csv=export_series_csv,
        ),
        "nmr_residuals": PlotDescriptor(
            id="nmr_residuals",
            display_name="Residuals",
            supports_axes=False,
            supports_series=True,
            supports_vary=False,
            series_selection_key="nmr_resid_selected",
            get_available_axes=lambda _data: [],
            get_available_series=lambda data: [
                (str(opt.get("id") or ""), str(opt.get("label") or opt.get("id") or ""))
                for opt in (data.get("signalOptions") or [])
                if opt.get("id")
            ],
            get_available_vary=lambda _data: [],
            build=_build_nmr_residuals,
            export_csv=_export_nmr_residuals_csv,
        ),
    }
