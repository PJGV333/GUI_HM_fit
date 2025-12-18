from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import wx

def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    out: list[float] = []
    try:
        for v in values:
            out.append(_to_float_or_nan(v))
    except Exception:
        return []
    return out


@dataclass
class PlotTrace:
    name: str
    y: list[float] = field(default_factory=list)
    x: Optional[list[float]] = None


@dataclass
class PlotDefinition:
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    traces: list[PlotTrace] = field(default_factory=list)
    png_base64: Optional[str] = None
    x_default: Optional[list[float]] = None


def _set_choice_items(choice: wx.Choice, items: list[str]) -> None:
    if hasattr(choice, "SetItems"):
        choice.SetItems(items)
    else:
        choice.Clear()
        for it in items:
            choice.Append(it)
    choice.Enable(True)
    choice.Refresh()


class PlotsTabPanel(wx.Panel):
    def __init__(self, parent: wx.Window, *, technique_panel: Any, module_key: str):
        super().__init__(parent)

        self._technique_panel = technique_panel
        self._module_key = str(module_key)

        self._result: dict[str, Any] | None = None
        self._config: dict[str, Any] | None = None

        self._presets: list[dict[str, Any]] = []
        self._preset_ids: list[str] = []
        self._preset_titles: list[str] = []

        self._original_def_by_preset: dict[str, PlotDefinition] = {}
        self._current_def: PlotDefinition | None = None
        self._current_preset_id: str | None = None
        self._current_fig = None

        self._build_ui()
        self._set_enabled(False)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        root = wx.BoxSizer(wx.VERTICAL)

        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(0, 1)
        grid.AddGrowableCol(1, 1)

        def field(label: str, ctrl: wx.Window) -> wx.Sizer:
            s = wx.BoxSizer(wx.VERTICAL)
            s.Add(wx.StaticText(self, label=label), 0, wx.BOTTOM, 2)
            s.Add(ctrl, 0, wx.EXPAND)
            return s

        # Row 1
        self.choice_plot_receptor = wx.Choice(self, choices=["Auto"])
        self.choice_plot_guest = wx.Choice(self, choices=["Auto"])
        grid.Add(field("Receptor or Ligand", self.choice_plot_receptor), 1, wx.EXPAND)
        grid.Add(field("Guest, Metal or Titrant", self.choice_plot_guest), 1, wx.EXPAND)

        # Row 2
        self.choice_preset = wx.Choice(self, choices=[])
        self.choice_xaxis = wx.Choice(
            self,
            choices=[
                "Auto",
                "titration point",
                "Guest equivalents",
                "Guest concentration",
                "Host concentration",
            ],
        )
        self.choice_xaxis.SetSelection(0)
        grid.Add(field("Preset", self.choice_preset), 1, wx.EXPAND)
        grid.Add(field("X axis", self.choice_xaxis), 1, wx.EXPAND)

        # Row 3
        self.list_yseries = wx.ListBox(self, style=wx.LB_MULTIPLE)
        self.choice_vary = wx.Choice(self, choices=["Auto", "titration", "wavelength/channel"])
        self.choice_vary.SetSelection(0)
        grid.Add(field("Y series", self.list_yseries), 1, wx.EXPAND)
        grid.Add(field("Vary along", self.choice_vary), 1, wx.EXPAND)

        root.Add(grid, 0, wx.EXPAND | wx.ALL, 10)

        # --- Edit plot ---
        box = wx.StaticBoxSizer(wx.StaticBox(self, label="Edit plot"), wx.VERTICAL)
        edit_grid = wx.FlexGridSizer(rows=0, cols=2, vgap=8, hgap=10)
        edit_grid.AddGrowableCol(0, 1)
        edit_grid.AddGrowableCol(1, 1)

        self.txt_title = wx.TextCtrl(self)
        self.txt_xlabel = wx.TextCtrl(self)
        self.txt_ylabel = wx.TextCtrl(self)
        self.choice_trace = wx.Choice(self, choices=[])
        self.txt_new_trace_name = wx.TextCtrl(self)

        edit_grid.Add(field("Title", self.txt_title), 1, wx.EXPAND)
        edit_grid.Add(field("X axis label", self.txt_xlabel), 1, wx.EXPAND)
        edit_grid.Add(field("Y axis label", self.txt_ylabel), 1, wx.EXPAND)
        edit_grid.Add(field("Trace", self.choice_trace), 1, wx.EXPAND)
        edit_grid.Add(field("New trace name", self.txt_new_trace_name), 1, wx.EXPAND)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_apply = wx.Button(self, label="Apply")
        self.btn_reset = wx.Button(self, label="Reset")
        btn_row.AddStretchSpacer()
        btn_row.Add(self.btn_apply, 0, wx.RIGHT, 6)
        btn_row.Add(self.btn_reset, 0)

        box.Add(edit_grid, 0, wx.EXPAND | wx.ALL, 10)
        box.Add(btn_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # --- Export ---
        export_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_export_png = wx.Button(self, label="Export PNG")
        self.btn_export_csv = wx.Button(self, label="Export CSV")
        export_row.AddStretchSpacer()
        export_row.Add(self.btn_export_png, 0, wx.RIGHT, 6)
        export_row.Add(self.btn_export_csv, 0)
        root.Add(export_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.SetSizer(root)

        # Bind events
        self.choice_preset.Bind(wx.EVT_CHOICE, self._on_preset_changed)
        self.choice_xaxis.Bind(wx.EVT_CHOICE, self._on_axis_changed)
        self.choice_plot_receptor.Bind(wx.EVT_CHOICE, self._on_axis_changed)
        self.choice_plot_guest.Bind(wx.EVT_CHOICE, self._on_axis_changed)
        self.list_yseries.Bind(wx.EVT_LISTBOX, self._on_yseries_changed)
        self.btn_apply.Bind(wx.EVT_BUTTON, self._on_apply)
        self.btn_reset.Bind(wx.EVT_BUTTON, self._on_reset)
        self.btn_export_png.Bind(wx.EVT_BUTTON, self._on_export_png)
        self.btn_export_csv.Bind(wx.EVT_BUTTON, self._on_export_csv)

    def _set_enabled(self, enabled: bool) -> None:
        for ctrl in (
            self.choice_plot_receptor,
            self.choice_plot_guest,
            self.choice_preset,
            self.choice_xaxis,
            self.list_yseries,
            self.choice_vary,
            self.txt_title,
            self.txt_xlabel,
            self.txt_ylabel,
            self.choice_trace,
            self.txt_new_trace_name,
            self.btn_apply,
            self.btn_reset,
            self.btn_export_png,
            self.btn_export_csv,
        ):
            ctrl.Enable(bool(enabled))

    # ---------------- Data binding ----------------
    def set_result(self, result: dict[str, Any] | None, *, config: dict[str, Any] | None = None) -> None:
        self._result = result if isinstance(result, dict) else None
        self._config = dict(config or {})
        self._original_def_by_preset.clear()
        self._current_def = None
        self._current_preset_id = None
        self._current_fig = None

        if not self._result:
            self._presets = []
            self._preset_ids = []
            self._preset_titles = []
            _set_choice_items(self.choice_preset, [])
            self.list_yseries.Set([])
            _set_choice_items(self.choice_trace, [])
            self._set_enabled(False)
            return

        self._set_enabled(True)

        # Row 1 dropdowns
        conc_cols = self._get_conc_columns()
        _set_choice_items(self.choice_plot_receptor, ["Auto"] + conc_cols)
        _set_choice_items(self.choice_plot_guest, ["Auto"] + conc_cols)
        self.choice_plot_receptor.SetSelection(0)
        self.choice_plot_guest.SetSelection(0)

        # Presets
        self._presets = self._build_presets(self._result)
        self._preset_ids = [p["id"] for p in self._presets]
        self._preset_titles = [p["title"] for p in self._presets]
        _set_choice_items(self.choice_preset, self._preset_titles)

        if self._presets:
            self.choice_preset.SetSelection(0)
            self.load_preset(self._preset_ids[0])
        else:
            self._current_def = None
            self._current_preset_id = None

    def _get_conc_columns(self) -> list[str]:
        if self._config and isinstance(self._config.get("column_names"), list):
            return [str(x) for x in self._config.get("column_names") or []]
        try:
            # fallback: current technique panel has checkboxes for all conc columns
            keys = list(getattr(self._technique_panel, "vars_columnas", {}).keys())
            return [str(x) for x in keys]
        except Exception:
            return []

    def _build_presets(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        presets: list[dict[str, Any]] = []

        available = result.get("availablePlots")
        if isinstance(available, list) and available:
            for item in available:
                if isinstance(item, dict):
                    pid = str(item.get("id") or item.get("title") or "")
                    if not pid:
                        continue
                    title = str(item.get("title") or pid)
                    kind = str(item.get("kind") or "")
                    presets.append({"id": pid, "title": title, "kind": kind})
                else:
                    pid = str(item)
                    presets.append({"id": pid, "title": pid, "kind": ""})
            return presets

        plot_data = result.get("plot_data")
        if isinstance(plot_data, dict) and plot_data:
            for key in plot_data.keys():
                pid = str(key)
                presets.append({"id": pid, "title": pid, "kind": ""})
            return presets

        graphs = result.get("graphs") or result.get("legacy_graphs") or {}
        if isinstance(graphs, dict):
            for key in graphs.keys():
                pid = str(key)
                presets.append({"id": pid, "title": pid, "kind": "image"})

        return presets

    # ---------------- Plot building/rendering ----------------
    def load_preset(self, preset_id: str) -> None:
        if not preset_id or not self._result:
            return

        preset_id = str(preset_id)
        self._current_preset_id = preset_id

        title = preset_id
        try:
            idx = self._preset_ids.index(preset_id)
            title = self._preset_titles[idx]
        except Exception:
            pass

        plot_def = self._build_plot_definition(self._result, preset_id, title)
        self._original_def_by_preset[preset_id] = copy.deepcopy(plot_def)
        self._current_def = copy.deepcopy(plot_def)

        self._sync_controls_from_current_def()
        self._render_current()

    def _plotdata(self) -> dict[str, Any]:
        if not self._result:
            return {}
        pd = self._result.get("plotData") or {}
        if isinstance(pd, dict):
            sub = pd.get(self._module_key)
            if isinstance(sub, dict):
                return sub
        return {}

    def _build_plot_definition(self, result: dict[str, Any], preset_id: str, title: str) -> PlotDefinition:
        pd = self._plotdata()
        entry = pd.get(preset_id)

        # 1) PNG-base64 plotData (fast path)
        if isinstance(entry, dict) and entry.get("png_base64"):
            return PlotDefinition(
                title=title,
                x_label="",
                y_label="",
                png_base64=str(entry.get("png_base64") or ""),
            )

        # 2) Spectroscopy species distribution (arrays)
        if isinstance(entry, dict) and "C_by_species" in entry and "x_default" in entry:
            x = _as_float_list(entry.get("x_default") or [])
            traces = []
            c_by = entry.get("C_by_species") or {}
            for name, y in (c_by.items() if isinstance(c_by, dict) else []):
                traces.append(PlotTrace(name=str(name), y=_as_float_list(y or []), x=None))
            x_label = ""
            axis_opts = entry.get("axisOptions") or []
            for opt in axis_opts if isinstance(axis_opts, list) else []:
                if isinstance(opt, dict) and opt.get("id") == "titrant_total":
                    x_label = str(opt.get("label") or "")
                    break
            return PlotDefinition(
                title=title,
                x_label=x_label,
                y_label="Concentration (M)",
                traces=traces,
                x_default=list(x),
            )

        # 3) NMR plotData (arrays)
        if isinstance(entry, dict) and "signals" in entry and "x" in entry:
            x = _as_float_list(entry.get("x") or [])
            x_label = str(entry.get("xLabel") or "")

            traces = []
            signal_options = entry.get("signalOptions") or []
            signals = entry.get("signals") or {}
            if isinstance(signal_options, list) and isinstance(signals, dict):
                for opt in signal_options:
                    if not isinstance(opt, dict):
                        continue
                    sig_id = opt.get("id")
                    sig_label = str(opt.get("label") or sig_id or "")
                    if not sig_id or sig_id not in signals:
                        continue
                    sdata = signals.get(sig_id) or {}
                    if preset_id == "nmr_shifts_fit":
                        traces.append(
                            PlotTrace(
                                name=f"{sig_label} obs",
                                y=_as_float_list(sdata.get("obs") or []),
                                x=None,
                            )
                        )
                        traces.append(
                            PlotTrace(
                                name=f"{sig_label} fit",
                                y=_as_float_list(sdata.get("fit") or []),
                                x=None,
                            )
                        )
                        y_label = "δ (ppm)"
                    else:
                        # residuals
                        traces.append(
                            PlotTrace(
                                name=sig_label,
                                y=_as_float_list(sdata.get("resid") or []),
                                x=None,
                            )
                        )
                        y_label = "Residuals (ppm)"

                return PlotDefinition(
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    traces=traces,
                    x_default=x,
                )

        # 4) Legacy graphs fallback
        graphs = result.get("graphs") or result.get("legacy_graphs") or {}
        if isinstance(graphs, dict) and graphs.get(preset_id):
            return PlotDefinition(
                title=title,
                png_base64=str(graphs.get(preset_id) or ""),
            )

        return PlotDefinition(title=title)

    def _sync_controls_from_current_def(self) -> None:
        d = self._current_def
        if d is None:
            return

        self.txt_title.SetValue(str(d.title or ""))
        self.txt_xlabel.SetValue(str(d.x_label or ""))
        self.txt_ylabel.SetValue(str(d.y_label or ""))
        self.txt_new_trace_name.SetValue("")

        trace_names = [t.name for t in (d.traces or [])]
        self.list_yseries.Set(trace_names)
        try:
            # wx.ListBox doesn't consistently expose DeselectAll across wx versions.
            for i in range(self.list_yseries.GetCount()):
                self.list_yseries.Deselect(i)
        except Exception:
            pass

        _set_choice_items(self.choice_trace, trace_names)
        if trace_names:
            self.choice_trace.SetSelection(0)
        else:
            self.choice_trace.SetSelection(wx.NOT_FOUND)

        is_image = bool(d.png_base64)
        self.choice_xaxis.Enable(not is_image)
        self.list_yseries.Enable(not is_image)
        self.choice_vary.Enable(not is_image)
        self.choice_trace.Enable(not is_image and bool(trace_names))
        self.txt_new_trace_name.Enable(not is_image and bool(trace_names))

    def _selected_trace_indices(self) -> list[int]:
        if not self._current_def or not self._current_def.traces:
            return []
        idxs = list(self.list_yseries.GetSelections() or [])
        if not idxs:
            return list(range(len(self._current_def.traces)))
        return [int(i) for i in idxs if 0 <= int(i) < len(self._current_def.traces)]

    def _compute_x_vector(self, n: int) -> list[float] | None:
        if not self._current_def:
            return None

        mode = str(self.choice_xaxis.GetStringSelection() or "Auto").strip()
        if mode == "Auto":
            if self._current_def.x_default is not None:
                return list(self._current_def.x_default)
            return None

        if mode == "titration point":
            return [float(i + 1) for i in range(int(n))]

        host, guest = self._get_host_guest_totals()
        if mode == "Guest concentration":
            return list(guest) if guest is not None else None
        if mode == "Host concentration":
            return list(host) if host is not None else None
        if mode == "Guest equivalents":
            if host is None or guest is None:
                return None
            x = []
            for h, g in zip(host, guest, strict=False):
                if h is None or g is None or h == 0:
                    x.append(float("nan"))
                else:
                    x.append(float(g) / float(h))
            return x

        return None

    def _get_host_guest_totals(self):
        if not self._result:
            return None, None

        export = self._result.get("export_data") or {}
        ct = export.get("C_T")
        if ct is None:
            return None, None

        col_names = export.get("column_names") or (self._config.get("column_names") if self._config else None) or []
        if not isinstance(col_names, list):
            col_names = []

        try:
            import numpy as np

            ct_arr = np.asarray(ct, dtype=float)
        except Exception:
            return None, None

        host_name = (self.choice_plot_receptor.GetStringSelection() or "Auto").strip()
        guest_name = (self.choice_plot_guest.GetStringSelection() or "Auto").strip()
        if host_name == "Auto":
            host_name = str((self._config or {}).get("receptor_label") or "")
        if guest_name == "Auto":
            guest_name = str((self._config or {}).get("guest_label") or "")

        host = None
        guest = None
        try:
            if host_name and host_name in col_names:
                host = ct_arr[:, col_names.index(host_name)]
            if guest_name and guest_name in col_names:
                guest = ct_arr[:, col_names.index(guest_name)]
        except Exception:
            return None, None

        return host.tolist() if host is not None else None, guest.tolist() if guest is not None else None

    def _render_current(self) -> None:
        d = self._current_def
        if d is None:
            return

        try:
            if d.png_base64:
                from hmfit_core.plots import figure_from_png_base64

                fig = figure_from_png_base64(str(d.png_base64), title=str(d.title or ""))
                self._current_fig = fig
                self._technique_panel.update_canvas_figure(fig)
                return

            from matplotlib.figure import Figure

            traces_idx = self._selected_trace_indices()
            traces = [d.traces[i] for i in traces_idx] if d.traces else []

            n = 0
            if d.x_default is not None:
                n = len(d.x_default)
            elif traces:
                n = len(traces[0].y)

            x_override = self._compute_x_vector(n) if n else None

            fig = Figure(figsize=(6.0, 4.0), dpi=150)
            ax = fig.add_subplot(111)

            for tr in traces:
                y = tr.y or []
                x = tr.x or x_override or d.x_default
                if x is None:
                    x = [float(i + 1) for i in range(len(y))]
                m = min(len(x), len(y))
                ax.plot(x[:m], y[:m], marker="o", linestyle="-", label=str(tr.name))

            ax.set_title(str(d.title or ""))
            ax.set_xlabel(str(d.x_label or ""))
            ax.set_ylabel(str(d.y_label or ""))
            if len(traces) > 1:
                ax.legend()
            fig.tight_layout()

            self._current_fig = fig
            self._technique_panel.update_canvas_figure(fig)
        except Exception as exc:
            wx.MessageBox(str(exc), "Plot error", wx.OK | wx.ICON_ERROR)

    # ---------------- Events ----------------
    def _on_preset_changed(self, evt) -> None:
        try:
            idx = int(self.choice_preset.GetSelection())
        except Exception:
            idx = wx.NOT_FOUND
        if idx < 0 or idx >= len(self._preset_ids):
            return
        self.load_preset(self._preset_ids[idx])
        evt.Skip()

    def _on_axis_changed(self, evt) -> None:
        if self._current_def and not self._current_def.png_base64:
            self._render_current()
        evt.Skip()

    def _on_yseries_changed(self, evt) -> None:
        if self._current_def and not self._current_def.png_base64:
            self._render_current()
        evt.Skip()

    def _on_apply(self, evt) -> None:
        d = self._current_def
        if d is None:
            return

        d.title = str(self.txt_title.GetValue() or "")
        d.x_label = str(self.txt_xlabel.GetValue() or "")
        d.y_label = str(self.txt_ylabel.GetValue() or "")

        if d.traces:
            idx = self.choice_trace.GetSelection()
            new_name = str(self.txt_new_trace_name.GetValue() or "").strip()
            if new_name and 0 <= idx < len(d.traces):
                d.traces[idx].name = new_name
                self.txt_new_trace_name.SetValue("")
                self._sync_controls_from_current_def()

        self._render_current()
        evt.Skip()

    def _on_reset(self, evt) -> None:
        if not self._current_preset_id:
            return
        orig = self._original_def_by_preset.get(self._current_preset_id)
        if orig is None:
            return
        self._current_def = copy.deepcopy(orig)
        self._sync_controls_from_current_def()
        self._render_current()
        evt.Skip()

    def _on_export_png(self, evt) -> None:
        if self._current_fig is None:
            wx.MessageBox("No plot to export yet.", "Export PNG", wx.OK | wx.ICON_INFORMATION)
            return

        with wx.FileDialog(
            self,
            "Export PNG",
            wildcard="PNG (*.png)|*.png",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
            if not path.lower().endswith(".png"):
                path += ".png"

        try:
            self._current_fig.savefig(path, format="png", dpi=200)
        except Exception as exc:
            wx.MessageBox(str(exc), "Export PNG", wx.OK | wx.ICON_ERROR)
            return

        wx.MessageBox(f"Saved: {path}", "Export PNG", wx.OK | wx.ICON_INFORMATION)
        evt.Skip()

    def _on_export_csv(self, evt) -> None:
        d = self._current_def
        if d is None or not d.traces:
            wx.MessageBox("No series data to export for this preset.", "Export CSV", wx.OK | wx.ICON_INFORMATION)
            return

        with wx.FileDialog(
            self,
            "Export CSV",
            wildcard="CSV (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
            if not path.lower().endswith(".csv"):
                path += ".csv"

        idxs = self._selected_trace_indices()
        traces = [d.traces[i] for i in idxs]

        n = 0
        if d.x_default is not None:
            n = len(d.x_default)
        elif traces:
            n = max(len(t.y or []) for t in traces)

        x = self._compute_x_vector(n) or (list(d.x_default) if d.x_default is not None else None)
        if x is None:
            x = [float(i + 1) for i in range(int(n))]

        n = len(x)
        data: dict[str, list[float]] = {"x": [_to_float_or_nan(v) for v in x]}
        for i, tr in enumerate(traces, start=1):
            name = str(tr.name or f"trace_{i}")
            y = [_to_float_or_nan(v) for v in (tr.y or [])]
            if len(y) < n:
                y = y + [float("nan")] * (n - len(y))
            elif len(y) > n:
                y = y[:n]
            data[name] = y

        try:
            import pandas as pd

            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
        except Exception as exc:
            wx.MessageBox(str(exc), "Export CSV", wx.OK | wx.ICON_ERROR)
            return

        wx.MessageBox(f"Saved: {path}", "Export CSV", wx.OK | wx.ICON_INFORMATION)
        evt.Skip()
