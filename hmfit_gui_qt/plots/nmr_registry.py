from __future__ import annotations

from typing import Any

from hmfit_gui_qt.plots.plot_registry import (
    PlotBuildResult,
    PlotControls,
    PlotDescriptor,
    PlotSeries,
    axis_options,
    species_options,
)


def _resolve_nmr_axis(
    data: dict[str, Any],
    controls: PlotControls,
) -> tuple[list[Any], str]:
    axis_vectors = data.get("axisVectors") or {}
    axis_options = data.get("axisOptions") or []
    axis_id = str(controls.nmr_x_axis_id or "")
    x: list[Any] = []
    x_label = str(data.get("xLabel") or "Concentration")

    if axis_id and axis_id in axis_vectors:
        x = list(axis_vectors.get(axis_id) or [])
    elif axis_vectors:
        default_id = str(data.get("x_default_id") or "")
        if default_id and default_id in axis_vectors:
            controls.nmr_x_axis_id = default_id
            x = list(axis_vectors.get(default_id) or [])
        elif axis_vectors:
            first_id = next(iter(axis_vectors.keys()))
            controls.nmr_x_axis_id = first_id
            x = list(axis_vectors.get(first_id) or [])
    else:
        x = list(data.get("x") or [])

    axis_id = str(controls.nmr_x_axis_id or "")
    for opt in axis_options:
        if str(opt.get("id") or "") == axis_id:
            x_label = str(opt.get("label") or x_label)
            break

    return x, x_label


def _segment_indices_by_x_reset(x_vals: list[Any], tol: float | None = None) -> list[list[int]]:
    numeric: list[float] = []
    for val in x_vals:
        try:
            numeric.append(float(val))
        except Exception:
            continue
    if not numeric:
        return []
    max_abs = max(abs(v) for v in numeric)
    tol_val = float(tol) if tol is not None else max(1e-12, 1e-6 * max_abs)

    segments: list[list[int]] = []
    current: list[int] = []
    prev: float | None = None
    for idx, val in enumerate(x_vals):
        try:
            x = float(val)
        except Exception:
            if current:
                segments.append(current)
                current = []
            prev = None
            continue
        if prev is not None and x < prev - tol_val:
            if current:
                segments.append(current)
            current = []
        current.append(idx)
        prev = x
    if current:
        segments.append(current)
    return segments


def _segment_xy(
    x_vals: list[Any],
    y_vals: list[Any],
    indices: list[int],
    *,
    sort_by_x: bool,
) -> tuple[list[Any], list[Any]]:
    pairs: list[tuple[Any, Any]] = []
    for idx in indices:
        if idx >= len(x_vals) or idx >= len(y_vals):
            continue
        xv = x_vals[idx]
        yv = y_vals[idx]
        if xv is None or yv is None:
            continue
        pairs.append((xv, yv))
    if not pairs:
        return [], []
    if sort_by_x:
        pairs.sort(key=lambda t: t[0])
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def _ensure_lengths(x_vals: list[Any], y_vals: list[Any], label: str) -> None:
    if len(x_vals) != len(y_vals):
        raise ValueError(f"{label}: length mismatch (x={len(x_vals)} y={len(y_vals)})")


def _build_nmr_shifts_fit(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
) -> PlotBuildResult:
    signal_options = data.get("signalOptions") or []
    signals = data.get("signals") or {}

    if not controls.nmr_signals_selected:
        controls.nmr_signals_selected = {str(opt.get("id") or "") for opt in signal_options if opt.get("id")}

    x_raw, x_label = _resolve_nmr_axis(data, controls)
    segments = _segment_indices_by_x_reset(x_raw)
    if not segments and x_raw:
        segments = [list(range(len(x_raw)))]
    series: list[PlotSeries] = []

    for opt in signal_options:
        sig_id = str(opt.get("id") or "")
        if not sig_id or sig_id not in controls.nmr_signals_selected:
            continue
        label = str(opt.get("label") or sig_id)
        sig = signals.get(sig_id) or {}
        obs = sig.get("obs") or []
        fit = sig.get("fit") or []
        _ensure_lengths(x_raw, obs, f"{label} obs")
        _ensure_lengths(x_raw, fit, f"{label} fit")
        show_obs = True
        show_fit = True
        for seg in segments:
            x_obs, y_obs = _segment_xy(x_raw, obs, seg, sort_by_x=False)
            if x_obs:
                series.append(
                    PlotSeries(
                        x=x_obs,
                        y=y_obs,
                        label=f"{label} obs",
                        mode="markers",
                        style={"markersize": 8},
                        show_legend=show_obs,
                    )
                )
                show_obs = False
            x_fit, y_fit = _segment_xy(x_raw, fit, seg, sort_by_x=True)
            if x_fit:
                series.append(
                    PlotSeries(
                        x=x_fit,
                        y=y_fit,
                        label=f"{label} fit",
                        mode="lines",
                        style={"linewidth": 2.0, "linestyle": ":"},
                        show_legend=show_fit,
                    )
                )
                show_fit = False

    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="\u0394\u03b4 (ppm)")


def _build_nmr_residuals(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
) -> PlotBuildResult:
    signal_options = data.get("signalOptions") or []
    signals = data.get("signals") or {}

    if not controls.nmr_resid_selected:
        controls.nmr_resid_selected = {str(opt.get("id") or "") for opt in signal_options if opt.get("id")}

    x, x_label = _resolve_nmr_axis(data, controls)
    segments = _segment_indices_by_x_reset(x)
    if not segments and x:
        segments = [list(range(len(x)))]
    series: list[PlotSeries] = []
    for opt in signal_options:
        sig_id = str(opt.get("id") or "")
        if not sig_id or sig_id not in controls.nmr_resid_selected:
            continue
        label = str(opt.get("label") or sig_id)
        sig = signals.get(sig_id) or {}
        resid = sig.get("resid") or []
        if not resid:
            obs = sig.get("obs") or []
            fit = sig.get("fit") or []
            if obs and fit:
                _ensure_lengths(obs, fit, f"{label} obs/fit")
                resid = [o - f for o, f in zip(obs, fit)]
        _ensure_lengths(x, resid, f"{label} resid")
        show_resid = True
        for seg in segments:
            x_seg, y_seg = _segment_xy(x, resid, seg, sort_by_x=False)
            if not x_seg:
                continue
            series.append(
                PlotSeries(
                    x=x_seg,
                    y=y_seg,
                    label=f"{label} resid",
                    mode="markers",
                    style={"markersize": 6},
                    show_legend=show_resid,
                )
            )
            show_resid = False

    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="Residuals (ppm)")


def _build_nmr_species_distribution(
    title: str,
    data: dict[str, Any],
    controls: PlotControls,
) -> PlotBuildResult:
    x, x_label = _resolve_nmr_axis(data, controls)
    species_opts = data.get("speciesOptions") or []
    if not controls.dist_y_selected:
        controls.dist_y_selected = {str(opt.get("id") or "") for opt in species_opts if opt.get("id")}

    c_by_species = data.get("C_by_species") or {}
    segments = _segment_indices_by_x_reset(x)
    if not segments and x:
        segments = [list(range(len(x)))]

    series: list[PlotSeries] = []
    for opt in species_opts:
        sp_id = str(opt.get("id") or "")
        if not sp_id or sp_id not in controls.dist_y_selected:
            continue
        label = str(opt.get("label") or sp_id)
        y = c_by_species.get(sp_id) or []
        _ensure_lengths(x, y, f"{label} species")
        show_leg = True
        for seg in segments:
            x_seg, y_seg = _segment_xy(x, y, seg, sort_by_x=True)
            if not x_seg:
                continue
            series.append(
                PlotSeries(
                    x=x_seg,
                    y=y_seg,
                    label=label,
                    mode="lines+markers",
                    style={"linewidth": 2.0, "markersize": 6},
                    show_legend=show_leg,
                )
            )
            show_leg = False

    return PlotBuildResult(series=series, title=title, x_label=x_label, y_label="[Species], M")


def _export_nmr_shifts_csv(
    data: dict[str, Any],
    controls: PlotControls,
    build: PlotBuildResult,
) -> tuple[list[str], list[list[Any]]] | None:
    x, _x_label = _resolve_nmr_axis(data, controls)
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
    x, _x_label = _resolve_nmr_axis(data, controls)
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


def _export_nmr_species_distribution_csv(
    data: dict[str, Any],
    controls: PlotControls,
    _build: PlotBuildResult,
) -> tuple[list[str], list[list[Any]]] | None:
    x, x_label = _resolve_nmr_axis(data, controls)
    species_opts = data.get("speciesOptions") or []
    c_by_species = data.get("C_by_species") or {}
    if not x:
        return None

    selected = set(controls.dist_y_selected)
    if not selected:
        selected = {str(opt.get("id") or "") for opt in species_opts if opt.get("id")}

    headers = [x_label]
    ordered: list[str] = []
    for opt in species_opts:
        sp_id = str(opt.get("id") or "")
        if sp_id and sp_id in selected:
            ordered.append(sp_id)
            headers.append(str(opt.get("label") or sp_id))

    rows: list[list[Any]] = []
    for i, xv in enumerate(x):
        row: list[Any] = [xv]
        for sp_id in ordered:
            y = c_by_species.get(sp_id) or []
            row.append(y[i] if i < len(y) else "")
        rows.append(row)
    return headers, rows


def build_nmr_registry() -> dict[str, PlotDescriptor]:
    return {
        "nmr_shifts_fit": PlotDescriptor(
            id="nmr_shifts_fit",
            display_name="Chemical shifts fit",
            supports_axes=True,
            supports_series=True,
            supports_vary=False,
            series_selection_key="nmr_signals_selected",
            get_available_axes=axis_options,
            get_available_series=lambda data: [
                (str(opt.get("id") or ""), str(opt.get("label") or opt.get("id") or ""))
                for opt in (data.get("signalOptions") or [])
                if opt.get("id")
            ],
            get_available_vary=lambda _data: [],
            build=_build_nmr_shifts_fit,
            export_csv=_export_nmr_shifts_csv,
            axis_selection_key="nmr_x_axis_id",
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
            build=_build_nmr_species_distribution,
            export_csv=_export_nmr_species_distribution_csv,
            axis_selection_key="nmr_x_axis_id",
        ),
        "nmr_residuals": PlotDescriptor(
            id="nmr_residuals",
            display_name="Residuals",
            supports_axes=True,
            supports_series=True,
            supports_vary=False,
            series_selection_key="nmr_resid_selected",
            get_available_axes=axis_options,
            get_available_series=lambda data: [
                (str(opt.get("id") or ""), str(opt.get("label") or opt.get("id") or ""))
                for opt in (data.get("signalOptions") or [])
                if opt.get("id")
            ],
            get_available_vary=lambda _data: [],
            build=_build_nmr_residuals,
            export_csv=_export_nmr_residuals_csv,
            axis_selection_key="nmr_x_axis_id",
        ),
    }
