from __future__ import annotations

from typing import Any

from hmfit_gui_qt.plots.plot_registry import (
    PlotBuildResult,
    PlotControls,
    PlotDescriptor,
    PlotSeries,
    export_series_csv,
)


def _series_options(data: dict[str, Any]) -> list[tuple[str, str]]:
    opts = data.get("seriesOptions") or []
    out: list[tuple[str, str]] = []
    for opt in opts:
        opt_id = str(opt.get("id") or "")
        label = str(opt.get("label") or opt_id)
        if opt_id:
            out.append((opt_id, label))
    return out


def _build_series_plot(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
    *,
    selection_key: str,
) -> PlotBuildResult:
    series_opts = _series_options(data)
    series_data = data.get("seriesData") or {}
    if not series_opts:
        return PlotBuildResult(series=[], title=title, x_label="", y_label="")

    selected = set(getattr(controls, selection_key, set()))
    if not selected:
        selected = {sid for sid, _ in series_opts}
        setattr(controls, selection_key, selected)

    series: list[PlotSeries] = []
    for sid, label in series_opts:
        if sid not in selected:
            continue
        payload = series_data.get(sid) or {}
        x = payload.get("x") or []
        y = payload.get("y") or []
        mode = str(payload.get("mode") or "lines")
        style = payload.get("style") or {}
        series.append(
            PlotSeries(
                x=x,
                y=y,
                label=label,
                mode=mode,
                style=dict(style),
            )
        )

    return PlotBuildResult(
        series=series,
        title=title,
        x_label=str(data.get("x_label") or ""),
        y_label=str(data.get("y_label") or ""),
    )


def build_kinetics_registry() -> dict[str, PlotDescriptor]:
    return {
        "kinetics_concentrations": PlotDescriptor(
            id="kinetics_concentrations",
            display_name="C(t)",
            supports_axes=False,
            supports_series=True,
            supports_vary=False,
            series_selection_key="kinetics_c_selected",
            get_available_axes=lambda _data: [],
            get_available_series=_series_options,
            get_available_vary=lambda _data: [],
            build=lambda title, data, controls: _build_series_plot(
                title, data, controls, selection_key="kinetics_c_selected"
            ),
            export_csv=export_series_csv,
        ),
        "kinetics_d_fit": PlotDescriptor(
            id="kinetics_d_fit",
            display_name="D vs D_hat",
            supports_axes=False,
            supports_series=True,
            supports_vary=False,
            series_selection_key="kinetics_d_selected",
            get_available_axes=lambda _data: [],
            get_available_series=_series_options,
            get_available_vary=lambda _data: [],
            build=lambda title, data, controls: _build_series_plot(
                title, data, controls, selection_key="kinetics_d_selected"
            ),
            export_csv=export_series_csv,
        ),
        "kinetics_residuals": PlotDescriptor(
            id="kinetics_residuals",
            display_name="Residuals",
            supports_axes=False,
            supports_series=True,
            supports_vary=False,
            series_selection_key="kinetics_resid_selected",
            get_available_axes=lambda _data: [],
            get_available_series=_series_options,
            get_available_vary=lambda _data: [],
            build=lambda title, data, controls: _build_series_plot(
                title, data, controls, selection_key="kinetics_resid_selected"
            ),
            export_csv=export_series_csv,
        ),
        "kinetics_a_profiles": PlotDescriptor(
            id="kinetics_a_profiles",
            display_name="A profiles",
            supports_axes=False,
            supports_series=True,
            supports_vary=False,
            series_selection_key="kinetics_a_selected",
            get_available_axes=lambda _data: [],
            get_available_series=_series_options,
            get_available_vary=lambda _data: [],
            build=lambda title, data, controls: _build_series_plot(
                title, data, controls, selection_key="kinetics_a_selected"
            ),
            export_csv=export_series_csv,
        ),
    }

