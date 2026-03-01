from __future__ import annotations

from typing import Any, Callable
import warnings

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox

from hmfit_gui_qt.plots.plot_registry import (
    PlotBuildResult,
    PlotDefaults,
    PlotDescriptor,
    PlotOverrides,
    PlotPreset,
    PlotSeries,
    PlotState,
)


class PlotController:
    _HOVER_LEGEND_THRESHOLD = 12

    def __init__(
        self,
        *,
        canvas,
        log,
        model_opt_plots,
        btn_prev,
        btn_next,
        lbl_title,
        registry: dict[str, PlotDescriptor],
        build_plot_data: Callable[[dict[str, Any]], dict[str, Any]],
        legacy_title_for_key: Callable[[str], str],
        legacy_order: list[str],
        default_title: str,
        is_running: Callable[[], bool],
    ) -> None:
        self._canvas = canvas
        self._log = log
        self._model_opt_plots = model_opt_plots
        self._btn_prev = btn_prev
        self._btn_next = btn_next
        self._lbl_title = lbl_title
        self._registry = registry
        self._build_plot_data = build_plot_data
        self._legacy_title_for_key = legacy_title_for_key
        self._legacy_order = list(legacy_order or [])
        self._default_title = str(default_title or "")
        self._is_running = is_running
        self._plot_state = PlotState()
        self._layout_fallback_count = 0
        self._hover_cid: int | None = None
        self._hover_annotation = None
        self._hover_targets: list[tuple[Any, str]] = []

    def wire_controls(self) -> None:
        self._model_opt_plots.combo_preset.currentIndexChanged.connect(self._on_plot_preset_changed)
        self._model_opt_plots.combo_x_axis.currentIndexChanged.connect(self._on_plot_x_axis_changed)
        self._model_opt_plots.list_y_series.itemSelectionChanged.connect(self._on_plot_y_series_changed)
        self._model_opt_plots.combo_vary_along.currentIndexChanged.connect(self._on_plot_vary_changed)
        self._model_opt_plots.combo_trace.currentIndexChanged.connect(self._on_plot_trace_changed)
        self._model_opt_plots.btn_apply_plot_edit.clicked.connect(self._on_plot_edit_apply)
        self._model_opt_plots.btn_reset_plot_edit.clicked.connect(self._on_plot_edit_reset)
        self._model_opt_plots.btn_export_png.clicked.connect(self._on_export_plot_png)
        self._model_opt_plots.btn_export_csv.clicked.connect(self._on_export_plot_csv)
        self._model_opt_plots.btn_export_png.setToolTip("")
        self._model_opt_plots.btn_export_csv.setToolTip("")

    def reset(self) -> None:
        self._disable_hover_labels()
        self._plot_state = PlotState()
        self._refresh_preset_combo()
        self.update_plot_nav()
        self._sync_plot_controls_for_active_plot()

    def build_from_result(self, result: dict[str, Any]) -> None:
        self._disable_hover_labels()
        self._plot_state = PlotState()
        available = result.get("availablePlots") or []
        if isinstance(available, list) and available:
            plots: list[PlotPreset] = []
            for item in available:
                pid = str(item.get("id") or "")
                if not pid:
                    continue
                plots.append(
                    PlotPreset(
                        id=pid,
                        title=str(item.get("title") or pid),
                        kind=str(item.get("kind") or "plotly"),
                    )
                )
            self._plot_state.available_plots = plots
            self._plot_state.active_plot_index = 0
            self._plot_state.plot_data = self._build_plot_data(result)
            self._plot_state.controls.dist_x_axis_id = "titrant_total"
            self._plot_state.controls.nmr_x_axis_id = ""
            self._plot_state.controls.dist_y_selected.clear()
            self._plot_state.controls.nmr_signals_selected.clear()
            self._plot_state.controls.nmr_resid_selected.clear()
            self._plot_state.controls.vary_along = ""
        else:
            graphs = result.get("legacy_graphs") or result.get("graphs") or {}
            if not isinstance(graphs, dict):
                graphs = {}
            pages: list[PlotPreset] = []
            for key in self._legacy_order:
                if graphs.get(key):
                    pages.append(PlotPreset(id=str(key), title=self._legacy_title_for_key(str(key)), kind="image"))
            for key in graphs.keys():
                if key in self._legacy_order:
                    continue
                if graphs.get(key):
                    pages.append(PlotPreset(id=str(key), title=self._legacy_title_for_key(str(key)), kind="image"))
            self._plot_state.available_plots = pages
            self._plot_state.active_plot_index = 0
            self._plot_state.plot_data = {
                p.id: {"png_base64": graphs.get(p.id) or ""} for p in pages
            }

        self._refresh_preset_combo()

    def render_current_plot(self) -> None:
        plot = self._active_plot()
        if not plot:
            self._disable_hover_labels()
            self._canvas.clear()
            self.update_plot_nav()
            self._sync_plot_controls_for_active_plot()
            return

        if self._plot_state.active_plot_index >= len(self._plot_state.available_plots):
            self._plot_state.active_plot_index = 0
            plot = self._active_plot()
            if not plot:
                self._canvas.clear()
                self.update_plot_nav()
                self._sync_plot_controls_for_active_plot()
                return

        data = self._plot_state.plot_data.get(plot.id) or {}
        descriptor = self._registry.get(plot.id)
        if descriptor and data:
            try:
                build = descriptor.build(plot.title, data, self._plot_state.controls)
            except Exception as exc:
                self._canvas.show_message(f"Plot render failed: {exc}")
                self._log.append_text(f"Plot render failed: {exc}")
                if plot.id in self._plot_state.last_builds:
                    del self._plot_state.last_builds[plot.id]
            else:
                build = self._apply_plot_overrides(plot.id, build)
                self._plot_state.last_builds[plot.id] = build
                self._draw_plot(build)
        elif data.get("png_base64"):
            self._canvas.show_image_base64(str(data.get("png_base64")), title=plot.title)
            self._disable_hover_labels()
            if plot.id in self._plot_state.last_builds:
                del self._plot_state.last_builds[plot.id]
        else:
            self._disable_hover_labels()
            self._canvas.show_message("No data for this plot")
            if plot.id in self._plot_state.last_builds:
                del self._plot_state.last_builds[plot.id]

        self.update_plot_nav()
        self._sync_plot_controls_for_active_plot()

    def update_plot_nav(self) -> None:
        has_pages = bool(self._plot_state.available_plots)
        can_nav = bool(has_pages and len(self._plot_state.available_plots) > 1)
        running = bool(self._is_running())
        if self._btn_prev is not None:
            self._btn_prev.setEnabled(can_nav and not running)
        if self._btn_next is not None:
            self._btn_next.setEnabled(can_nav and not running)
        if self._lbl_title is not None:
            plot = self._active_plot()
            if plot:
                self._lbl_title.setText(str(plot.title or ""))
            else:
                self._lbl_title.setText(self._default_title)

    def navigate(self, delta: int) -> None:
        plots = self._plot_state.available_plots
        if not plots:
            return
        n = len(plots)
        if n <= 1:
            return
        self._plot_state.active_plot_index = (self._plot_state.active_plot_index + int(delta)) % n
        self.render_current_plot()

    def _active_plot(self) -> PlotPreset | None:
        if not self._plot_state.available_plots:
            return None
        idx = self._plot_state.active_plot_index
        if idx < 0 or idx >= len(self._plot_state.available_plots):
            return None
        return self._plot_state.available_plots[idx]

    def _refresh_preset_combo(self) -> None:
        combo = self._model_opt_plots.combo_preset
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select a preset...", "")
        for plot in self._plot_state.available_plots:
            combo.addItem(plot.title, plot.id)
        active = self._active_plot()
        if active:
            ix = combo.findData(active.id)
            if ix >= 0:
                combo.setCurrentIndex(ix)
        combo.blockSignals(False)

    def _apply_plot_overrides(self, plot_id: str, build: PlotBuildResult) -> PlotBuildResult:
        defaults = self._plot_state.defaults.get(plot_id)
        if defaults is None:
            defaults = PlotDefaults(
                title_text=build.title,
                x_label=build.x_label,
                y_label=build.y_label,
                trace_names={str(i): s.label for i, s in enumerate(build.series)},
            )
            self._plot_state.defaults[plot_id] = defaults

        overrides = self._plot_state.overrides.get(plot_id)
        if overrides:
            title = overrides.title_text
            x_label = overrides.x_label
            y_label = overrides.y_label
        else:
            title = build.title
            x_label = build.x_label
            y_label = build.y_label

        series: list[PlotSeries] = []
        for i, s in enumerate(build.series):
            name = s.label
            if overrides and overrides.trace_names.get(str(i)):
                name = overrides.trace_names[str(i)]
            series.append(
                PlotSeries(
                    x=s.x,
                    y=s.y,
                    label=name,
                    mode=s.mode,
                    style=s.style,
                    show_legend=s.show_legend,
                )
            )

        return PlotBuildResult(series=series, title=title, x_label=x_label, y_label=y_label)

    def _draw_plot(self, build: PlotBuildResult) -> None:
        ax = self._canvas.ax
        ax.clear()
        if not build.series:
            self._disable_hover_labels()
            self._canvas.show_message("No data for this plot")
            return

        drawn_series: list[tuple[Any, str]] = []
        for series in build.series:
            label = series.label if series.show_legend else f"_{series.label}"
            mode = series.mode
            style = dict(series.style or {})
            if mode == "markers":
                line = ax.plot(series.x, series.y, linestyle="None", marker="o", label=label, **style)[0]
            elif mode == "lines+markers":
                style.setdefault("marker", "o")
                line = ax.plot(series.x, series.y, label=label, **style)[0]
            else:
                line = ax.plot(series.x, series.y, label=label, **style)[0]
            if series.show_legend:
                drawn_series.append((line, str(series.label)))

        ax.set_title(build.title)
        ax.set_xlabel(build.x_label)
        ax.set_ylabel(build.y_label)
        ax.grid(True, alpha=0.3)
        legend_external = False
        use_hover_labels = len(drawn_series) >= self._HOVER_LEGEND_THRESHOLD
        if drawn_series:
            if use_hover_labels:
                self._enable_hover_labels(ax, drawn_series)
                ax.text(
                    0.01,
                    0.99,
                    "Hover markers to identify series",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#6b7280",
                )
            else:
                self._disable_hover_labels()
                legend_external = self._apply_legend_layout(ax, build)
        else:
            self._disable_hover_labels()
        self._apply_axis_label_layout(ax)
        self._apply_figure_layout(external_legend=legend_external)
        self._canvas.draw_idle()

    def _apply_legend_layout(self, ax, build: PlotBuildResult) -> bool:
        n_legend = sum(1 for s in build.series if s.show_legend)
        if n_legend <= 0:
            return False
        if n_legend > 8:
            ncol = min(4, max(2, int(np.ceil(n_legend / 6.0))))
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=ncol,
                fontsize=8,
                frameon=False,
            )
            return True
        ax.legend(loc="best", fontsize=9)
        return False

    def _apply_axis_label_layout(self, ax) -> None:
        x_labels = [str(lbl.get_text() or "").strip() for lbl in ax.get_xticklabels()]
        non_empty_x = [txt for txt in x_labels if txt]
        dense_x = len(non_empty_x) >= 8 or any(len(txt) > 10 for txt in non_empty_x)
        if dense_x:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_horizontalalignment("right")
            ax.tick_params(axis="x", labelsize=9)

        y_labels = [str(lbl.get_text() or "").strip() for lbl in ax.get_yticklabels()]
        non_empty_y = [txt for txt in y_labels if txt]
        dense_y = len(non_empty_y) >= 10 or any(len(txt) > 10 for txt in non_empty_y)
        if dense_y:
            ax.tick_params(axis="y", labelsize=9)

        ax.margins(x=0.02, y=0.05)

    def _apply_figure_layout(self, *, external_legend: bool) -> None:
        fig = self._canvas.figure
        tight_layout_failed = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=UserWarning)
            try:
                fig.tight_layout(pad=1.2)
            except Exception:
                tight_layout_failed = True
            else:
                for warning in caught:
                    message = str(warning.message or "")
                    if "Tight layout not applied" in message:
                        tight_layout_failed = True
                        break

        if not tight_layout_failed:
            return

        # Fallback when decorations overflow the available canvas area.
        bottom = 0.24 if external_legend else 0.16
        fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=bottom)
        if self._layout_fallback_count < 3:
            self._layout_fallback_count += 1
            self._log.append_text(
                "Plot layout fallback applied (labels/legend were too dense for tight layout)."
            )

    def _enable_hover_labels(self, ax, targets: list[tuple[Any, str]]) -> None:
        self._disable_hover_labels()
        self._hover_targets = list(targets)
        for artist, _label in self._hover_targets:
            try:
                artist.set_pickradius(6)
            except Exception:
                pass

        self._hover_annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#111827", "ec": "#374151", "alpha": 0.95},
            color="#f9fafb",
            fontsize=8,
        )
        self._hover_annotation.set_visible(False)
        self._hover_cid = self._canvas.mpl_connect("motion_notify_event", self._on_hover_motion)

    def _disable_hover_labels(self) -> None:
        if self._hover_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._hover_cid)
            except Exception:
                pass
            self._hover_cid = None
        self._hover_targets = []
        if self._hover_annotation is not None:
            try:
                self._hover_annotation.remove()
            except Exception:
                pass
            self._hover_annotation = None

    def _on_hover_motion(self, event) -> None:
        annotation = self._hover_annotation
        if annotation is None:
            return
        ax = self._canvas.ax
        needs_redraw = False
        if event.inaxes != ax:
            if annotation.get_visible():
                annotation.set_visible(False)
                needs_redraw = True
            if needs_redraw:
                self._canvas.draw_idle()
            return

        hit_found = False
        for artist, label in self._hover_targets:
            contains, info = artist.contains(event)
            if not contains:
                continue
            idx = None
            if isinstance(info, dict):
                indices = info.get("ind")
                if isinstance(indices, (list, tuple, np.ndarray)) and len(indices) > 0:
                    try:
                        idx = int(indices[0])
                    except Exception:
                        idx = None
            x_val = event.xdata
            y_val = event.ydata
            if idx is not None:
                try:
                    x_data = np.asarray(artist.get_xdata(), dtype=float)
                    y_data = np.asarray(artist.get_ydata(), dtype=float)
                    if 0 <= idx < x_data.size and idx < y_data.size:
                        x_val = float(x_data[idx])
                        y_val = float(y_data[idx])
                except Exception:
                    pass
            if x_val is None or y_val is None:
                continue
            prev_xy = tuple(annotation.xy)
            next_xy = (x_val, y_val)
            next_text = str(label)
            if prev_xy != next_xy:
                annotation.xy = next_xy
                needs_redraw = True
            if annotation.get_text() != next_text:
                annotation.set_text(next_text)
                needs_redraw = True
            if not annotation.get_visible():
                annotation.set_visible(True)
                needs_redraw = True
            hit_found = True
            break

        if not hit_found and annotation.get_visible():
            annotation.set_visible(False)
            needs_redraw = True
        if needs_redraw:
            self._canvas.draw_idle()

    def _get_series_selection_key(self, descriptor: PlotDescriptor) -> str:
        return descriptor.series_selection_key or "dist_y_selected"

    def _get_axis_selection_key(self, descriptor: PlotDescriptor) -> str:
        return descriptor.axis_selection_key or "dist_x_axis_id"

    def _get_axis_selection(self, descriptor: PlotDescriptor) -> str:
        key = self._get_axis_selection_key(descriptor)
        return str(getattr(self._plot_state.controls, key, ""))

    def _set_axis_selection(self, descriptor: PlotDescriptor, axis_id: str) -> None:
        key = self._get_axis_selection_key(descriptor)
        setattr(self._plot_state.controls, key, str(axis_id or ""))

    def _get_series_selection(self, descriptor: PlotDescriptor) -> set[str]:
        key = self._get_series_selection_key(descriptor)
        return set(getattr(self._plot_state.controls, key, set()))

    def _set_series_selection(self, descriptor: PlotDescriptor, selection: set[str]) -> None:
        key = self._get_series_selection_key(descriptor)
        setattr(self._plot_state.controls, key, set(selection))

    def _sync_plot_controls_for_active_plot(self) -> None:
        preset_combo = self._model_opt_plots.combo_preset
        x_axis_combo = self._model_opt_plots.combo_x_axis
        y_series_list = self._model_opt_plots.list_y_series
        vary_combo = self._model_opt_plots.combo_vary_along
        edit_title = self._model_opt_plots.edit_title
        edit_xlabel = self._model_opt_plots.edit_xlabel
        edit_ylabel = self._model_opt_plots.edit_ylabel
        trace_combo = self._model_opt_plots.combo_trace
        trace_name = self._model_opt_plots.edit_trace_name
        export_png = self._model_opt_plots.btn_export_png
        export_csv = self._model_opt_plots.btn_export_csv

        plot = self._active_plot()
        if not plot:
            self._refresh_preset_combo()
            for widget in (x_axis_combo, vary_combo):
                widget.blockSignals(True)
                widget.clear()
                widget.addItem("Select X axis..." if widget is x_axis_combo else "Auto", "")
                widget.setEnabled(False)
                widget.setToolTip("Run Process Data first.")
                widget.blockSignals(False)
            y_series_list.blockSignals(True)
            y_series_list.clear()
            y_series_list.setEnabled(False)
            y_series_list.setToolTip("Run Process Data first.")
            y_series_list.blockSignals(False)
            for edit in (edit_title, edit_xlabel, edit_ylabel, trace_name):
                edit.setText("")
            trace_combo.blockSignals(True)
            trace_combo.clear()
            trace_combo.addItem("Select trace...", "")
            trace_combo.blockSignals(False)
            export_png.setEnabled(False)
            export_csv.setEnabled(False)
            return

        self._refresh_preset_combo()

        data = self._plot_state.plot_data.get(plot.id) or {}
        descriptor = self._registry.get(plot.id)
        build = self._plot_state.last_builds.get(plot.id)

        supports_controls = bool(descriptor and (descriptor.supports_axes or descriptor.supports_series))
        if descriptor is None:
            x_axis_combo.blockSignals(True)
            x_axis_combo.clear()
            x_axis_combo.addItem("Not applicable for this preset", "")
            x_axis_combo.setEnabled(False)
            x_axis_combo.setToolTip("Not applicable for this preset.")
            x_axis_combo.blockSignals(False)

            y_series_list.blockSignals(True)
            y_series_list.clear()
            item = QListWidgetItem("Not applicable for this preset", y_series_list)
            item.setData(Qt.ItemDataRole.UserRole, "")
            y_series_list.setEnabled(False)
            y_series_list.setToolTip("Not applicable for this preset.")
            y_series_list.blockSignals(False)
        elif not data:
            x_axis_combo.blockSignals(True)
            x_axis_combo.clear()
            x_axis_combo.addItem("No data for this plot", "")
            x_axis_combo.setEnabled(False)
            x_axis_combo.setToolTip("No data for this plot.")
            x_axis_combo.blockSignals(False)

            y_series_list.blockSignals(True)
            y_series_list.clear()
            item = QListWidgetItem("No data for this plot", y_series_list)
            item.setData(Qt.ItemDataRole.UserRole, "")
            y_series_list.setEnabled(False)
            y_series_list.setToolTip("No data for this plot.")
            y_series_list.blockSignals(False)
        elif supports_controls:
            x_axis_combo.blockSignals(True)
            x_axis_combo.clear()
            if descriptor.supports_axes:
                axes = descriptor.get_available_axes(data)
                if axes:
                    axis_ids = {axis_id for axis_id, _ in axes}
                    selected_axis_id = self._get_axis_selection(descriptor)
                    if selected_axis_id not in axis_ids:
                        default_id = str(data.get("x_default_id") or "")
                        if default_id in axis_ids:
                            selected_axis_id = default_id
                        else:
                            selected_axis_id = axes[0][0]
                        self._set_axis_selection(descriptor, selected_axis_id)
                    for axis_id, label in axes:
                        x_axis_combo.addItem(label, axis_id)
                    ix = x_axis_combo.findData(selected_axis_id)
                    if ix >= 0:
                        x_axis_combo.setCurrentIndex(ix)
                    x_axis_combo.setEnabled(True)
                    x_axis_combo.setToolTip("")
                else:
                    empty_label = str(data.get("x_axis_empty_label") or "No data for this plot")
                    x_axis_combo.addItem(empty_label, "")
                    x_axis_combo.setEnabled(False)
                    x_axis_combo.setToolTip(empty_label)
            else:
                x_axis_combo.addItem("Not applicable for this preset", "")
                x_axis_combo.setEnabled(False)
                x_axis_combo.setToolTip("Not applicable for this preset.")
            x_axis_combo.blockSignals(False)

            y_series_list.blockSignals(True)
            y_series_list.clear()
            if descriptor.supports_series:
                series_opts = descriptor.get_available_series(data)
                if series_opts:
                    selected = self._get_series_selection(descriptor)
                    if not selected:
                        selected = {sid for sid, _ in series_opts}
                        self._set_series_selection(descriptor, selected)
                    for sid, label in series_opts:
                        item = QListWidgetItem(label, y_series_list)
                        item.setData(Qt.ItemDataRole.UserRole, sid)
                        item.setSelected(sid in selected)
                    y_series_list.setEnabled(True)
                    y_series_list.setToolTip("")
                else:
                    item = QListWidgetItem("No data for this plot", y_series_list)
                    item.setData(Qt.ItemDataRole.UserRole, "")
                    y_series_list.setEnabled(False)
                    y_series_list.setToolTip("No data for this plot.")
            else:
                item = QListWidgetItem("Not applicable for this preset", y_series_list)
                item.setData(Qt.ItemDataRole.UserRole, "")
                y_series_list.setEnabled(False)
                y_series_list.setToolTip("Not applicable for this preset.")
            y_series_list.blockSignals(False)
        else:
            x_axis_combo.blockSignals(True)
            x_axis_combo.clear()
            x_axis_combo.addItem("Not applicable for this preset", "")
            x_axis_combo.setEnabled(False)
            x_axis_combo.setToolTip("Not applicable for this preset.")
            x_axis_combo.blockSignals(False)

            y_series_list.blockSignals(True)
            y_series_list.clear()
            item = QListWidgetItem("Not applicable for this preset", y_series_list)
            item.setData(Qt.ItemDataRole.UserRole, "")
            y_series_list.setEnabled(False)
            y_series_list.setToolTip("Not applicable for this preset.")
            y_series_list.blockSignals(False)

        vary_combo.blockSignals(True)
        vary_combo.clear()
        vary_combo.addItem("Auto", "")
        vary_combo.setEnabled(False)
        vary_combo.setToolTip("Not applicable for this preset.")
        vary_combo.blockSignals(False)

        if build:
            edit_title.setText(build.title)
            edit_xlabel.setText(build.x_label)
            edit_ylabel.setText(build.y_label)
            trace_combo.blockSignals(True)
            trace_combo.clear()
            trace_combo.addItem("Select trace...", "")
            for idx, series in enumerate(build.series):
                trace_combo.addItem(f"{idx}: {series.label}", idx)
            trace_combo.setCurrentIndex(0)
            trace_combo.blockSignals(False)
            trace_name.setText("")
        else:
            for edit in (edit_title, edit_xlabel, edit_ylabel, trace_name):
                edit.setText("")
            trace_combo.blockSignals(True)
            trace_combo.clear()
            trace_combo.addItem("Select trace...", "")
            trace_combo.blockSignals(False)

        running = bool(self._is_running())
        can_export_png = bool(plot and data) and not running
        can_export_csv = False
        if descriptor and build:
            can_export_csv = bool(descriptor.export_csv(data, self._plot_state.controls, build))
        export_png.setEnabled(can_export_png)
        export_csv.setEnabled(can_export_csv and not running)

    def _on_plot_preset_changed(self) -> None:
        plot_id = str(self._model_opt_plots.combo_preset.currentData() or "")
        if not plot_id:
            return
        for i, plot in enumerate(self._plot_state.available_plots):
            if plot.id == plot_id:
                self._plot_state.active_plot_index = i
                self.render_current_plot()
                return

    def _on_plot_x_axis_changed(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        descriptor = self._registry.get(plot.id)
        if not descriptor or not descriptor.supports_axes:
            return
        axis_id = str(self._model_opt_plots.combo_x_axis.currentData() or "")
        if axis_id:
            self._set_axis_selection(descriptor, axis_id)
            self.render_current_plot()

    def _on_plot_y_series_changed(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        descriptor = self._registry.get(plot.id)
        if not descriptor or not descriptor.supports_series:
            return
        selected: set[str] = set()
        for item in self._model_opt_plots.list_y_series.selectedItems():
            sid = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if sid:
                selected.add(sid)
        self._set_series_selection(descriptor, selected)
        self.render_current_plot()

    def _on_plot_vary_changed(self) -> None:
        self._plot_state.controls.vary_along = str(self._model_opt_plots.combo_vary_along.currentData() or "")
        if self._plot_state.available_plots:
            self.render_current_plot()

    def _on_plot_trace_changed(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        idx = self._model_opt_plots.combo_trace.currentData()
        build = self._plot_state.last_builds.get(plot.id)
        if build is None or idx is None:
            self._model_opt_plots.edit_trace_name.setText("")
            return
        try:
            i = int(idx)
        except Exception:
            self._model_opt_plots.edit_trace_name.setText("")
            return
        if i < 0 or i >= len(build.series):
            self._model_opt_plots.edit_trace_name.setText("")
            return
        self._model_opt_plots.edit_trace_name.setText(build.series[i].label)

    def _on_plot_edit_apply(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        title = str(self._model_opt_plots.edit_title.text() or "").strip()
        x_label = str(self._model_opt_plots.edit_xlabel.text() or "").strip()
        y_label = str(self._model_opt_plots.edit_ylabel.text() or "").strip()
        overrides = self._plot_state.overrides.get(plot.id) or PlotOverrides()
        overrides.title_text = title
        overrides.x_label = x_label
        overrides.y_label = y_label

        trace_idx = self._model_opt_plots.combo_trace.currentData()
        new_name = str(self._model_opt_plots.edit_trace_name.text() or "").strip()
        if new_name and trace_idx is not None:
            try:
                i = int(trace_idx)
            except Exception:
                i = -1
            if i >= 0:
                overrides.trace_names[str(i)] = new_name

        self._plot_state.overrides[plot.id] = overrides
        self.render_current_plot()
        self._log.append_text(f"Applied plot edits for {plot.id}.")

    def _on_plot_edit_reset(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        defaults = self._plot_state.defaults.get(plot.id)
        if defaults:
            self._plot_state.overrides[plot.id] = PlotOverrides(
                title_text=defaults.title_text,
                x_label=defaults.x_label,
                y_label=defaults.y_label,
                trace_names=dict(defaults.trace_names),
            )
            self.render_current_plot()
            if plot.id in self._plot_state.overrides:
                del self._plot_state.overrides[plot.id]
        else:
            if plot.id in self._plot_state.overrides:
                del self._plot_state.overrides[plot.id]
            self.render_current_plot()
        self._log.append_text(f"Reset plot edits for {plot.id}.")

    def _on_export_plot_png(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._model_opt_plots, "Export plot PNG", f"{plot.id}.png", "PNG (*.png)"
        )
        if not path:
            return
        try:
            self._canvas.figure.savefig(path, dpi=300, bbox_inches="tight")
            self._log.append_text(f"Exported: {path}")
        except Exception as exc:
            self._log.append_text(f"Export failed: {exc}")
            QMessageBox.warning(self._model_opt_plots, "Export failed", str(exc))

    def _on_export_plot_csv(self) -> None:
        plot = self._active_plot()
        if not plot:
            return
        descriptor = self._registry.get(plot.id)
        data = self._plot_state.plot_data.get(plot.id) or {}
        build = self._plot_state.last_builds.get(plot.id)
        if not descriptor or build is None:
            self._log.append_text("No data to export.")
            return
        payload = descriptor.export_csv(data, self._plot_state.controls, build)
        if not payload:
            self._log.append_text("CSV export not available for this preset.")
            return

        header, rows = payload
        path, _ = QFileDialog.getSaveFileName(
            self._model_opt_plots, "Export plot CSV", f"{plot.id}.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import csv

            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
            self._log.append_text(f"Exported: {path}")
        except Exception as exc:
            self._log.append_text(f"Export failed: {exc}")
            QMessageBox.warning(self._model_opt_plots, "Export failed", str(exc))
