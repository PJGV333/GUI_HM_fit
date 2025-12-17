from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import wx
import wx.grid as gridlib

from hmfit_wx import excel as excel_utils
from hmfit_wx.images import as_bool_cell, as_float_or_none, bitmap_from_base64_png


@dataclass(frozen=True)
class NMRInputs:
    file_path: Path
    nmr_sheet: str
    conc_sheet: str
    column_names: List[str]
    signal_names: List[str]
    receptor_label: str
    guest_label: str
    model_grid: List[List[float]]  # (n_spec_total x n_comp)
    non_abs_species: List[int]
    algorithm: str
    model_settings: str
    optimizer: str
    initial_k: List[float]
    bounds: List[List[Optional[float]]]  # [[min,max], ...]
    fixed: List[bool]


class NMRPanel(wx.Panel):
    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None

        self._file_path: Optional[Path] = None
        self._sheets: List[str] = []

        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = wx.Panel(splitter)
        right = wx.Panel(splitter)
        splitter.SplitVertically(left, right, sashPosition=520)

        self._build_left(left)
        self._build_right(right)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(splitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def _build_left(self, parent: wx.Panel) -> None:
        root = wx.BoxSizer(wx.VERTICAL)

        file_box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Data")
        file_row = wx.BoxSizer(wx.HORIZONTAL)
        self.file_txt = wx.TextCtrl(parent, style=wx.TE_READONLY)
        browse_btn = wx.Button(parent, label="Browse…")
        browse_btn.Bind(wx.EVT_BUTTON, self._on_browse)
        file_row.Add(self.file_txt, 1, wx.EXPAND | wx.RIGHT, 8)
        file_row.Add(browse_btn, 0)
        file_box.Add(file_row, 0, wx.EXPAND | wx.ALL, 6)

        grid_sheets = wx.FlexGridSizer(rows=2, cols=2, vgap=6, hgap=8)
        grid_sheets.AddGrowableCol(1, 1)
        grid_sheets.Add(wx.StaticText(parent, label="NMR sheet:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.nmr_choice = wx.Choice(parent, choices=[])
        self.nmr_choice.Bind(wx.EVT_CHOICE, self._on_nmr_sheet_changed)
        grid_sheets.Add(self.nmr_choice, 1, wx.EXPAND)
        grid_sheets.Add(wx.StaticText(parent, label="Conc. sheet:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.conc_choice = wx.Choice(parent, choices=[])
        self.conc_choice.Bind(wx.EVT_CHOICE, self._on_conc_sheet_changed)
        grid_sheets.Add(self.conc_choice, 1, wx.EXPAND)
        file_box.Add(grid_sheets, 0, wx.EXPAND | wx.ALL, 6)

        sig_box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Signals (NMR)")
        self.signal_cols = wx.CheckListBox(parent, choices=[])
        sig_box.Add(self.signal_cols, 1, wx.EXPAND | wx.ALL, 6)

        cols_box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Concentration columns")
        self.conc_cols = wx.CheckListBox(parent, choices=[])
        self.conc_cols.Bind(wx.EVT_CHECKLISTBOX, self._on_conc_columns_changed)
        cols_box.Add(self.conc_cols, 1, wx.EXPAND | wx.ALL, 6)

        roles_box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Roles")
        roles_grid = wx.FlexGridSizer(rows=2, cols=2, vgap=6, hgap=8)
        roles_grid.AddGrowableCol(1, 1)
        roles_grid.Add(wx.StaticText(parent, label="Receptor:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.receptor_choice = wx.Choice(parent, choices=[])
        roles_grid.Add(self.receptor_choice, 1, wx.EXPAND)
        roles_grid.Add(wx.StaticText(parent, label="Guest:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.guest_choice = wx.Choice(parent, choices=[])
        roles_grid.Add(self.guest_choice, 1, wx.EXPAND)
        roles_box.Add(roles_grid, 0, wx.EXPAND | wx.ALL, 6)

        model_box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Model")
        model_top = wx.BoxSizer(wx.HORIZONTAL)
        model_top.Add(wx.StaticText(parent, label="nComp:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.ncomp_spin = wx.SpinCtrl(parent, min=1, max=10, initial=2)
        model_top.Add(self.ncomp_spin, 0, wx.RIGHT, 12)
        model_top.Add(wx.StaticText(parent, label="nComplex:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.ncomplex_spin = wx.SpinCtrl(parent, min=1, max=50, initial=1)
        model_top.Add(self.ncomplex_spin, 0, wx.RIGHT, 12)
        gen_btn = wx.Button(parent, label="Generate grids")
        gen_btn.Bind(wx.EVT_BUTTON, self._on_generate_grids)
        model_top.Add(gen_btn, 0)
        model_box.Add(model_top, 0, wx.EXPAND | wx.ALL, 6)

        self.model_grid = gridlib.Grid(parent)
        self.model_grid.CreateGrid(0, 0)
        self.model_grid.SetMinSize((-1, 200))
        model_box.Add(self.model_grid, 0, wx.EXPAND | wx.ALL, 6)

        self.nonabs_chk = wx.CheckListBox(parent, choices=[])
        model_box.Add(wx.StaticText(parent, label="Non-absorbing species:"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        model_box.Add(self.nonabs_chk, 0, wx.EXPAND | wx.ALL, 6)

        opt_box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Optimization")
        opt_grid = wx.FlexGridSizer(rows=3, cols=2, vgap=6, hgap=8)
        opt_grid.AddGrowableCol(1, 1)
        opt_grid.Add(wx.StaticText(parent, label="Algorithm:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.algorithm_choice = wx.Choice(parent, choices=["Newton-Raphson", "Levenberg-Marquardt"])
        self.algorithm_choice.SetSelection(0)
        opt_grid.Add(self.algorithm_choice, 1, wx.EXPAND)
        opt_grid.Add(wx.StaticText(parent, label="Model settings:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.model_settings_choice = wx.Choice(parent, choices=["Free", "Step by step", "Non-cooperative"])
        self.model_settings_choice.SetSelection(0)
        self.model_settings_choice.Bind(wx.EVT_CHOICE, self._on_model_settings_changed)
        opt_grid.Add(self.model_settings_choice, 1, wx.EXPAND)
        opt_grid.Add(wx.StaticText(parent, label="Optimizer:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.optimizer_choice = wx.Choice(
            parent,
            choices=[
                "powell",
                "nelder-mead",
                "trust-constr",
                "cg",
                "bfgs",
                "l-bfgs-b",
                "tnc",
                "cobyla",
                "slsqp",
                "differential_evolution",
            ],
        )
        self.optimizer_choice.SetSelection(0)
        opt_grid.Add(self.optimizer_choice, 1, wx.EXPAND)
        opt_box.Add(opt_grid, 0, wx.EXPAND | wx.ALL, 6)

        self.k_grid = gridlib.Grid(parent)
        self.k_grid.CreateGrid(0, 4)
        self.k_grid.SetColLabelValue(0, "K (log10)")
        self.k_grid.SetColLabelValue(1, "Min")
        self.k_grid.SetColLabelValue(2, "Max")
        self.k_grid.SetColLabelValue(3, "Fixed")
        self.k_grid.SetColFormatBool(3)
        self.k_grid.SetMinSize((-1, 150))
        self.k_grid.Bind(gridlib.EVT_GRID_CELL_CHANGED, self._on_k_grid_changed)
        opt_box.Add(self.k_grid, 0, wx.EXPAND | wx.ALL, 6)

        actions = wx.BoxSizer(wx.HORIZONTAL)
        self.process_btn = wx.Button(parent, label="Process")
        self.process_btn.Bind(wx.EVT_BUTTON, self._on_process)
        actions.Add(self.process_btn, 0, wx.RIGHT, 8)
        self.reset_btn = wx.Button(parent, label="Reset log")
        self.reset_btn.Bind(wx.EVT_BUTTON, lambda _evt: self.log_txt.SetValue(""))
        actions.Add(self.reset_btn, 0)

        root.Add(file_box, 0, wx.EXPAND | wx.ALL, 8)
        root.Add(sig_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.Add(cols_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.Add(roles_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.Add(model_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.Add(opt_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.Add(actions, 0, wx.ALIGN_RIGHT | wx.ALL, 8)

        parent.SetSizer(root)

        self._on_generate_grids(None)

    def _build_right(self, parent: wx.Panel) -> None:
        root = wx.BoxSizer(wx.VERTICAL)

        self.results_nb = wx.Notebook(parent)

        plots_panel = wx.Panel(self.results_nb)
        plots_sizer = wx.BoxSizer(wx.VERTICAL)
        self.plots_nb = wx.Notebook(plots_panel)
        plots_sizer.Add(self.plots_nb, 1, wx.EXPAND)
        plots_panel.SetSizer(plots_sizer)

        self.constants_grid = gridlib.Grid(self.results_nb)
        self.constants_grid.CreateGrid(0, 4)
        self.constants_grid.SetColLabelValue(0, "Name")
        self.constants_grid.SetColLabelValue(1, "log10K")
        self.constants_grid.SetColLabelValue(2, "SE_log10K")
        self.constants_grid.SetColLabelValue(3, "% error")

        self.report_txt = wx.TextCtrl(
            self.results_nb, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2
        )
        mono = wx.Font(
            10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
        )
        self.report_txt.SetFont(mono)

        self.results_nb.AddPage(plots_panel, "Plots")
        self.results_nb.AddPage(self.constants_grid, "Constants")
        self.results_nb.AddPage(self.report_txt, "Report")

        self.log_txt = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY)

        root.Add(self.results_nb, 1, wx.EXPAND | wx.ALL, 8)
        root.Add(wx.StaticText(parent, label="Log:"), 0, wx.LEFT | wx.RIGHT, 8)
        root.Add(self.log_txt, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        parent.SetSizer(root)

    def _append_log(self, msg: str) -> None:
        if not msg.endswith("\n"):
            msg += "\n"
        self.log_txt.AppendText(msg)

    def _set_busy(self, busy: bool) -> None:
        self.process_btn.Enable(not busy)

    def _on_browse(self, _evt: wx.CommandEvent) -> None:
        with wx.FileDialog(
            self,
            message="Select Excel file",
            wildcard="Excel files (*.xlsx;*.xls)|*.xlsx;*.xls|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            self._file_path = Path(dlg.GetPath())
            self.file_txt.SetValue(str(self._file_path))

        try:
            self._sheets = excel_utils.list_sheets(self._file_path)
        except Exception as exc:
            wx.MessageBox(f"Error reading workbook: {exc}", "Error", wx.ICON_ERROR)
            self._sheets = []
            return

        self.nmr_choice.Set(self._sheets)
        self.conc_choice.Set(self._sheets)
        if self._sheets:
            self.nmr_choice.SetSelection(0)
            self.conc_choice.SetSelection(0)
            self._on_nmr_sheet_changed(None)
            self._on_conc_sheet_changed(None)

    def _on_nmr_sheet_changed(self, _evt: Optional[wx.CommandEvent]) -> None:
        if not self._file_path:
            return
        sheet = self.nmr_choice.GetStringSelection()
        if not sheet:
            return
        try:
            cols = excel_utils.list_columns(self._file_path, sheet)
        except Exception as exc:
            wx.MessageBox(f"Error reading columns: {exc}", "Error", wx.ICON_ERROR)
            return
        self.signal_cols.Set(cols)
        for i in range(len(cols)):
            self.signal_cols.Check(i, False)

    def _on_conc_sheet_changed(self, _evt: Optional[wx.CommandEvent]) -> None:
        if not self._file_path:
            return
        sheet = self.conc_choice.GetStringSelection()
        if not sheet:
            return
        try:
            cols = excel_utils.list_columns(self._file_path, sheet)
        except Exception as exc:
            wx.MessageBox(f"Error reading columns: {exc}", "Error", wx.ICON_ERROR)
            return
        self.conc_cols.Set(cols)
        for i in range(len(cols)):
            self.conc_cols.Check(i, False)
        self._sync_roles([])

    def _on_conc_columns_changed(self, _evt: wx.CommandEvent) -> None:
        selected = [self.conc_cols.GetString(i) for i in self.conc_cols.GetCheckedItems()]
        self._sync_roles(selected)

    def _sync_roles(self, selected_columns: List[str]) -> None:
        self.receptor_choice.Set(selected_columns)
        self.guest_choice.Set(selected_columns)
        if selected_columns:
            self.receptor_choice.SetSelection(0)
            self.guest_choice.SetSelection(min(1, len(selected_columns) - 1))

    def _on_generate_grids(self, _evt: Optional[wx.CommandEvent]) -> None:
        ncomp = int(self.ncomp_spin.GetValue())
        ncomplex = int(self.ncomplex_spin.GetValue())
        n_total = ncomp + ncomplex

        self._reset_grid(self.model_grid, n_total, ncomp)
        for c in range(ncomp):
            self.model_grid.SetColLabelValue(c, f"C{c+1}")
        for r in range(n_total):
            self.model_grid.SetRowLabelValue(r, f"sp{r+1}")
            for c in range(ncomp):
                val = 1.0 if (r < ncomp and r == c) else 0.0
                self.model_grid.SetCellValue(r, c, str(val))

        old_checked = set(self.nonabs_chk.GetCheckedItems())
        self.nonabs_chk.Set([f"sp{i+1}" for i in range(n_total)])
        for idx in old_checked:
            if idx < self.nonabs_chk.GetCount():
                self.nonabs_chk.Check(idx, True)

        self._regenerate_k_grid()

    def _on_model_settings_changed(self, _evt: wx.CommandEvent) -> None:
        self._regenerate_k_grid()

    def _regenerate_k_grid(self) -> None:
        ncomplex = int(self.ncomplex_spin.GetValue())
        model_settings = self.model_settings_choice.GetStringSelection()
        n_params = 1 if model_settings == "Non-cooperative" else ncomplex
        if n_params <= 0:
            n_params = 1

        self._reset_grid(self.k_grid, n_params, 4)
        self.k_grid.SetColLabelValue(0, "K (log10)")
        self.k_grid.SetColLabelValue(1, "Min")
        self.k_grid.SetColLabelValue(2, "Max")
        self.k_grid.SetColLabelValue(3, "Fixed")
        for r in range(n_params):
            self.k_grid.SetRowLabelValue(r, f"K{r+1}")
            self.k_grid.SetCellValue(r, 0, "")
            self.k_grid.SetCellValue(r, 1, "")
            self.k_grid.SetCellValue(r, 2, "")
            self.k_grid.SetCellValue(r, 3, "")
        self.k_grid.SetColFormatBool(3)

    def _on_k_grid_changed(self, evt: gridlib.GridEvent) -> None:
        row = evt.GetRow()
        col = evt.GetCol()
        fixed = as_bool_cell(self.k_grid.GetCellValue(row, 3))
        if col == 3:
            self.k_grid.SetReadOnly(row, 1, fixed)
            self.k_grid.SetReadOnly(row, 2, fixed)
            if fixed:
                v = as_float_or_none(self.k_grid.GetCellValue(row, 0))
                if v is not None:
                    self.k_grid.SetCellValue(row, 1, str(v))
                    self.k_grid.SetCellValue(row, 2, str(v))
        if col == 0 and fixed:
            v = as_float_or_none(self.k_grid.GetCellValue(row, 0))
            if v is not None:
                self.k_grid.SetCellValue(row, 1, str(v))
                self.k_grid.SetCellValue(row, 2, str(v))
        evt.Skip()

    @staticmethod
    def _reset_grid(grid: gridlib.Grid, rows: int, cols: int) -> None:
        if grid.GetNumberRows() > 0:
            grid.DeleteRows(0, grid.GetNumberRows())
        if grid.GetNumberCols() > 0:
            grid.DeleteCols(0, grid.GetNumberCols())
        if rows > 0:
            grid.AppendRows(rows)
        if cols > 0:
            grid.AppendCols(cols)

    def _gather_inputs(self) -> NMRInputs:
        if not self._file_path:
            raise ValueError("No Excel file selected.")
        nmr_sheet = self.nmr_choice.GetStringSelection()
        conc_sheet = self.conc_choice.GetStringSelection()
        if not nmr_sheet or not conc_sheet:
            raise ValueError("Select NMR/conc sheets.")

        signal_names = [self.signal_cols.GetString(i) for i in self.signal_cols.GetCheckedItems()]
        if not signal_names:
            raise ValueError("Select at least one NMR signal column.")

        column_names = [self.conc_cols.GetString(i) for i in self.conc_cols.GetCheckedItems()]
        if not column_names:
            raise ValueError("Select at least one concentration column.")

        receptor_label = self.receptor_choice.GetStringSelection()
        guest_label = self.guest_choice.GetStringSelection()
        if not receptor_label or not guest_label:
            raise ValueError("Select receptor/guest columns.")

        n_total = self.model_grid.GetNumberRows()
        n_comp = self.model_grid.GetNumberCols()
        model_grid: List[List[float]] = []
        for r in range(n_total):
            row_vals: List[float] = []
            for c in range(n_comp):
                v = as_float_or_none(self.model_grid.GetCellValue(r, c))
                row_vals.append(float(v) if v is not None else 0.0)
            model_grid.append(row_vals)

        non_abs_species = [int(i) for i in self.nonabs_chk.GetCheckedItems()]

        algorithm = self.algorithm_choice.GetStringSelection()
        model_settings = self.model_settings_choice.GetStringSelection()
        optimizer = self.optimizer_choice.GetStringSelection()

        n_params = self.k_grid.GetNumberRows()
        initial_k: List[float] = []
        bounds: List[List[Optional[float]]] = []
        fixed: List[bool] = []
        for r in range(n_params):
            v = as_float_or_none(self.k_grid.GetCellValue(r, 0))
            v_float = float(v) if v is not None else 1.0
            is_fixed = as_bool_cell(self.k_grid.GetCellValue(r, 3))

            min_v = as_float_or_none(self.k_grid.GetCellValue(r, 1))
            max_v = as_float_or_none(self.k_grid.GetCellValue(r, 2))

            if is_fixed:
                min_v = v_float
                max_v = v_float

            initial_k.append(v_float)
            bounds.append([min_v, max_v])
            fixed.append(bool(is_fixed))

        return NMRInputs(
            file_path=self._file_path,
            nmr_sheet=nmr_sheet,
            conc_sheet=conc_sheet,
            column_names=column_names,
            signal_names=signal_names,
            receptor_label=receptor_label,
            guest_label=guest_label,
            model_grid=model_grid,
            non_abs_species=non_abs_species,
            algorithm=algorithm,
            model_settings=model_settings,
            optimizer=optimizer,
            initial_k=initial_k,
            bounds=bounds,
            fixed=fixed,
        )

    def _on_process(self, _evt: wx.CommandEvent) -> None:
        if self._future and not self._future.done():
            wx.MessageBox("A run is already in progress.", "Busy", wx.ICON_INFORMATION)
            return

        try:
            inputs = self._gather_inputs()
        except Exception as exc:
            wx.MessageBox(str(exc), "Validation error", wx.ICON_WARNING)
            return

        self.log_txt.SetValue("")
        self.report_txt.SetValue("")
        self._clear_plots()
        self._clear_constants()
        self._set_busy(True)

        self._future = self._executor.submit(self._run_processing, inputs)
        self._future.add_done_callback(lambda fut: wx.CallAfter(self._on_done, fut))

    def _progress_callback(self, message: str) -> None:
        wx.CallAfter(self._append_log, str(message))

    def _run_processing(self, inputs: NMRInputs) -> Dict[str, Any]:
        from backend_fastapi.nmr_processor import process_nmr_data, set_progress_callback

        set_progress_callback(self._progress_callback, loop=None)

        # Convert bounds [[min,max], ...] to expected list-of-dicts
        k_bounds: List[Dict[str, float]] = []
        for b in inputs.bounds:
            k_bounds.append({"min": b[0], "max": b[1]})

        return process_nmr_data(
            file_path=str(inputs.file_path),
            spectra_sheet=inputs.nmr_sheet,
            conc_sheet=inputs.conc_sheet,
            column_names=inputs.column_names,
            signal_names=inputs.signal_names,
            receptor_label=inputs.receptor_label,
            guest_label=inputs.guest_label,
            model_matrix=inputs.model_grid,
            k_initial=inputs.initial_k,
            k_bounds=k_bounds,
            algorithm=inputs.algorithm,
            optimizer=inputs.optimizer,
            model_settings=inputs.model_settings,
            non_absorbent_species=inputs.non_abs_species,
            k_fixed=inputs.fixed,
        )

    def _on_done(self, fut: Future) -> None:
        self._set_busy(False)
        try:
            result = fut.result()
        except Exception as exc:
            self._append_log(f"ERROR: {exc}")
            wx.MessageBox(str(exc), "Processing error", wx.ICON_ERROR)
            return

        if isinstance(result, dict) and result.get("error"):
            msg = str(result.get("error"))
            self._append_log(f"ERROR: {msg}")
            wx.MessageBox(msg, "Processing error", wx.ICON_ERROR)
            return

        self._render_results(result if isinstance(result, dict) else {})

    def _render_results(self, result: Dict[str, Any]) -> None:
        self.report_txt.SetValue(str(result.get("results_text") or ""))

        consts = result.get("constants") or []
        self._populate_constants(consts)

        graphs = result.get("graphs") or {}
        self._populate_plots_from_graphs(graphs)

    def _clear_plots(self) -> None:
        while self.plots_nb.GetPageCount() > 0:
            self.plots_nb.DeletePage(0)

    def _populate_plots_from_graphs(self, graphs: Dict[str, Any]) -> None:
        self._clear_plots()
        for key, b64 in graphs.items():
            bmp = bitmap_from_base64_png(str(b64 or ""))
            if bmp is None:
                continue
            panel = wx.ScrolledWindow(self.plots_nb, style=wx.VSCROLL | wx.HSCROLL)
            panel.SetScrollRate(20, 20)
            sizer = wx.BoxSizer(wx.VERTICAL)
            img = wx.StaticBitmap(panel, bitmap=bmp)
            sizer.Add(img, 0, wx.ALL, 8)
            panel.SetSizer(sizer)
            panel.FitInside()
            self.plots_nb.AddPage(panel, key)

    def _clear_constants(self) -> None:
        if self.constants_grid.GetNumberRows() > 0:
            self.constants_grid.DeleteRows(0, self.constants_grid.GetNumberRows())

    def _populate_constants(self, consts: List[Dict[str, Any]]) -> None:
        self._clear_constants()
        if not consts:
            return
        self.constants_grid.AppendRows(len(consts))
        for r, c in enumerate(consts):
            self.constants_grid.SetCellValue(r, 0, str(c.get("name", f"K{r+1}")))
            self.constants_grid.SetCellValue(r, 1, str(c.get("log10K", "")))
            self.constants_grid.SetCellValue(r, 2, str(c.get("SE_log10K", "")))
            self.constants_grid.SetCellValue(r, 3, str(c.get("percent_error", "")))
