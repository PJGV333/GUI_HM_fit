from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class PlotPreset:
    id: str
    title: str
    kind: str = "plotly"


@dataclass
class PlotOverrides:
    title_text: str = ""
    x_label: str = ""
    y_label: str = ""
    trace_names: dict[str, str] = field(default_factory=dict)


@dataclass
class PlotDefaults:
    title_text: str = ""
    x_label: str = ""
    y_label: str = ""
    trace_names: dict[str, str] = field(default_factory=dict)


@dataclass
class PlotControls:
    dist_x_axis_id: str = ""
    nmr_x_axis_id: str = ""
    dist_y_selected: set[str] = field(default_factory=set)
    nmr_signals_selected: set[str] = field(default_factory=set)
    nmr_resid_selected: set[str] = field(default_factory=set)
    vary_along: str = ""


@dataclass
class PlotSeries:
    x: Sequence[float]
    y: Sequence[float]
    label: str
    mode: str = "lines"
    style: dict[str, Any] = field(default_factory=dict)
    show_legend: bool = True


@dataclass
class PlotBuildResult:
    series: list[PlotSeries]
    title: str
    x_label: str
    y_label: str


@dataclass(frozen=True)
class PlotDescriptor:
    id: str
    display_name: str
    supports_axes: bool
    supports_series: bool
    supports_vary: bool
    series_selection_key: str | None
    get_available_axes: Callable[[dict[str, Any]], list[tuple[str, str]]]
    get_available_series: Callable[[dict[str, Any]], list[tuple[str, str]]]
    get_available_vary: Callable[[dict[str, Any]], list[tuple[str, str]]]
    build: Callable[[str, dict[str, Any], PlotControls], PlotBuildResult]
    export_csv: Callable[[dict[str, Any], PlotControls, PlotBuildResult], tuple[list[str], list[list[Any]]] | None]
    axis_selection_key: str | None = None


@dataclass
class PlotState:
    available_plots: list[PlotPreset] = field(default_factory=list)
    active_plot_index: int = 0
    plot_data: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, PlotOverrides] = field(default_factory=dict)
    defaults: dict[str, PlotDefaults] = field(default_factory=dict)
    controls: PlotControls = field(default_factory=PlotControls)
    last_builds: dict[str, PlotBuildResult] = field(default_factory=dict)


def axis_options(data: dict[str, Any]) -> list[tuple[str, str]]:
    opts = data.get("axisOptions") or []
    out: list[tuple[str, str]] = []
    for opt in opts:
        opt_id = str(opt.get("id") or "")
        label = str(opt.get("label") or opt_id)
        if opt_id:
            out.append((opt_id, label))
    return out


def species_options(data: dict[str, Any]) -> list[tuple[str, str]]:
    opts = data.get("speciesOptions") or []
    out: list[tuple[str, str]] = []
    for opt in opts:
        opt_id = str(opt.get("id") or "")
        label = str(opt.get("label") or opt_id)
        if opt_id:
            out.append((opt_id, label))
    return out


def build_species_distribution(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
) -> PlotBuildResult:
    axis_vectors = data.get("axisVectors") or {}
    axis_opts = data.get("axisOptions") or []
    x_axis_id = controls.dist_x_axis_id or "titrant_total"
    x = axis_vectors.get(x_axis_id) or data.get("x_default") or []
    x_label = "Concentration"
    for opt in axis_opts:
        if str(opt.get("id") or "") == x_axis_id:
            x_label = str(opt.get("label") or x_label)
            break

    species_opts = data.get("speciesOptions") or []
    if not controls.dist_y_selected:
        controls.dist_y_selected = {str(opt.get("id") or "") for opt in species_opts if opt.get("id")}

    c_by_species = data.get("C_by_species") or {}
    series: list[PlotSeries] = []
    for opt in species_opts:
        sp_id = str(opt.get("id") or "")
        if not sp_id or sp_id not in controls.dist_y_selected:
            continue
        label = str(opt.get("label") or sp_id)
        y = c_by_species.get(sp_id) or []
        if not y:
            continue
        series.append(
            PlotSeries(
                x=x,
                y=y,
                label=label,
                mode="lines+markers",
                style={"linewidth": 2.0, "markersize": 6},
            )
        )

    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="[Species], M")


def export_series_csv(
    _data: dict[str, Any],
    _controls: PlotControls,
    build: PlotBuildResult,
) -> tuple[list[str], list[list[Any]]] | None:
    if not build.series:
        return None
    x = list(build.series[0].x)
    header = [build.x_label] + [s.label for s in build.series]
    rows: list[list[Any]] = []
    for i, xv in enumerate(x):
        row = [xv]
        for s in build.series:
            row.append(s.y[i] if i < len(s.y) else "")
        rows.append(row)
    return header, rows
