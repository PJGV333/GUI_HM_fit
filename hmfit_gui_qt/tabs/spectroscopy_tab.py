from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hmfit_core.api import run_spectroscopy_fit
from hmfit_core.exports import write_results_xlsx
from hmfit_gui_qt.plots.plot_controller import PlotController
from hmfit_gui_qt.plots.spectroscopy_registry import build_spectroscopy_registry
from hmfit_gui_qt.plots.spectroscopy_sources import build_spectroscopy_plot_sources
from hmfit_gui_qt.widgets.log_console import LogConsole
from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsState, ModelOptPlotsWidget
from hmfit_gui_qt.widgets.mpl_canvas import MplCanvas, NavigationToolbar
from hmfit_gui_qt.widgets.channel_spec import DEFAULT_CHANNEL_TOL, parse_channel_spec, parse_channels_spec
from hmfit_gui_qt.workers.fit_worker import FitWorker


def _list_excel_sheets(file_path: str) -> list[str]:
    import pandas as pd

    xls = pd.ExcelFile(file_path)
    return [str(s) for s in xls.sheet_names]


def _list_excel_columns(file_path: str, sheet_name: str) -> list[str]:
    import pandas as pd

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, nrows=0)
    return [str(c) for c in df.columns]


def _count_excel_rows(file_path: str, sheet_name: str) -> int:
    import pandas as pd

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    return int(df.shape[0])


def _list_spectroscopy_axis_values(file_path: str, sheet_name: str) -> list[float]:
    import pandas as pd

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, usecols=[0])
    axis = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    axis = axis.loc[axis.notna()].astype(float)
    return [float(x) for x in axis.to_numpy()]


def _format_axis_val(v: float) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    rounded = round(x)
    if abs(x - rounded) < 1e-9:
        return str(int(rounded))
    return str(x)


def _axis_range_text(axis_values: list[float]) -> str:
    if not axis_values:
        return ""
    return f"{_format_axis_val(min(axis_values))}–{_format_axis_val(max(axis_values))}"


class SpectroscopyTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FitWorker | None = None
        self._thread: QThread | None = None
        self._last_result: dict[str, Any] | None = None
        self._plot_controller: PlotController | None = None
        self._axis_values: list[float] = []
        self._conc_points_count = 0
        self._file_path: str = ""
        self._is_running = False

        self._build_ui()
        if getattr(self.model_opt_plots, "roles_group", None) is not None:
            self.model_opt_plots.roles_group.setVisible(False)
        self._plot_controller = PlotController(
            canvas=self.canvas_main,
            log=self.log,
            model_opt_plots=self.model_opt_plots,
            btn_prev=self.btn_plot_prev,
            btn_next=self.btn_plot_next,
            lbl_title=self.lbl_plot_title,
            registry=build_spectroscopy_registry(self._selected_conc_columns),
            build_plot_data=build_spectroscopy_plot_sources,
            legacy_title_for_key=self._plot_title_for_key,
            legacy_order=["fit", "concentrations", "absorptivities", "eigenvalues", "efa", "residuals"],
            default_title="Main spectra / titration plot",
            is_running=lambda: self._is_running,
        )
        self._wire_plot_controls()
        self._reset_plot_state()
        self.log.append_text("Listo. Carga un archivo para comenzar.")
        self._set_running(False)

    # ---- UI ----
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._main_split = QSplitter(Qt.Orientation.Horizontal, self)
        outer.addWidget(self._main_split, 1)

        # ----- Left panel (scrollable) -----
        left_scroll = QScrollArea(self._main_split)
        left_scroll.setWidgetResizable(True)
        left_container = QWidget(left_scroll)
        left_scroll.setWidget(left_container)
        left_layout = QVBoxLayout(left_container)

        self._data_group = QGroupBox("DATA / INPUT", left_container)
        data_layout = QVBoxLayout(self._data_group)

        # Select Excel File
        data_layout.addWidget(QLabel("Select Excel File", self._data_group))
        file_row = QHBoxLayout()
        self.btn_choose_file = QPushButton("Choose File…", self._data_group)
        self.btn_choose_file.clicked.connect(self._on_choose_file_clicked)
        file_row.addWidget(self.btn_choose_file)
        self.lbl_file_status = QLabel("No file selected", self._data_group)
        self.lbl_file_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_row.addWidget(self.lbl_file_status, 1)
        data_layout.addLayout(file_row)

        # Sheet dropdowns
        sheets_row = QHBoxLayout()
        self.combo_spectra_sheet = QComboBox(self._data_group)
        self.combo_conc_sheet = QComboBox(self._data_group)
        self.combo_spectra_sheet.currentIndexChanged.connect(self._on_spectra_sheet_changed)
        self.combo_conc_sheet.currentIndexChanged.connect(self._on_conc_sheet_changed)

        spec_box = QVBoxLayout()
        spec_box.addWidget(QLabel("Spectra Sheet Name", self._data_group))
        spec_box.addWidget(self.combo_spectra_sheet)
        conc_box = QVBoxLayout()
        conc_box.addWidget(QLabel("Concentration Sheet Name", self._data_group))
        conc_box.addWidget(self.combo_conc_sheet)
        sheets_row.addLayout(spec_box, 1)
        sheets_row.addLayout(conc_box, 1)
        data_layout.addLayout(sheets_row)

        # Channels (All / Custom)
        channels_header = QHBoxLayout()
        channels_header.addWidget(QLabel("Channels", self._data_group))
        self.lbl_channels_range = QLabel("", self._data_group)
        self.lbl_channels_range.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        channels_header.addWidget(self.lbl_channels_range, 1)
        data_layout.addLayout(channels_header)

        channels_top = QHBoxLayout()
        self.combo_channels_mode = QComboBox(self._data_group)
        self.combo_channels_mode.addItem("All", "all")
        self.combo_channels_mode.addItem("Custom", "custom")
        self.combo_channels_mode.currentIndexChanged.connect(self._on_channels_mode_changed)
        channels_top.addWidget(self.combo_channels_mode)
        self.edit_channels_spec = QLineEdit(self._data_group)
        self.edit_channels_spec.setPlaceholderText("all or 250-400, 485, 510-520")
        self.edit_channels_spec.returnPressed.connect(self._on_apply_channels_clicked)
        channels_top.addWidget(self.edit_channels_spec, 1)
        self.btn_apply_channels = QPushButton("Apply", self._data_group)
        self.btn_apply_channels.clicked.connect(self._on_apply_channels_clicked)
        channels_top.addWidget(self.btn_apply_channels)
        data_layout.addLayout(channels_top)

        self.lbl_channels_usage = QLabel("", self._data_group)
        data_layout.addWidget(self.lbl_channels_usage)

        self._channels_custom_box = QWidget(self._data_group)
        channels_custom_layout = QVBoxLayout(self._channels_custom_box)
        channels_custom_layout.setContentsMargins(0, 0, 0, 0)
        self.list_channels = QListWidget(self._channels_custom_box)
        self.list_channels.itemChanged.connect(self._on_channels_selection_changed)
        self.list_channels.setMinimumHeight(120)
        channels_custom_layout.addWidget(self.list_channels, 1)
        ch_btns = QHBoxLayout()
        self.btn_channels_all = QPushButton("Select all", self._channels_custom_box)
        self.btn_channels_all.clicked.connect(lambda: self._set_list_checked(self.list_channels, True))
        ch_btns.addWidget(self.btn_channels_all)
        self.btn_channels_none = QPushButton("Select none", self._channels_custom_box)
        self.btn_channels_none.clicked.connect(lambda: self._set_list_checked(self.list_channels, False))
        ch_btns.addWidget(self.btn_channels_none)
        ch_btns.addStretch(1)
        channels_custom_layout.addLayout(ch_btns)
        data_layout.addWidget(self._channels_custom_box)

        # Column names (concentration columns)
        data_layout.addWidget(QLabel("Column names", self._data_group))
        self._columns_scroll = QScrollArea(self._data_group)
        self._columns_scroll.setWidgetResizable(True)
        self._columns_widget = QWidget(self._columns_scroll)
        self._columns_layout = QVBoxLayout(self._columns_widget)
        self._columns_layout.setContentsMargins(0, 0, 0, 0)
        self._columns_scroll.setWidget(self._columns_widget)
        self._columns_scroll.setMinimumHeight(90)
        data_layout.addWidget(self._columns_scroll)

        # EFA
        efa_row = QHBoxLayout()
        efa_row.addWidget(QLabel("EFA Eigenvalues", self._data_group))
        self.chk_efa = QCheckBox("EFA", self._data_group)
        self.chk_efa.setChecked(True)
        efa_row.addWidget(self.chk_efa)
        self.spin_efa_eigen = QSpinBox(self._data_group)
        self.spin_efa_eigen.setRange(0, 999)
        self.spin_efa_eigen.setValue(0)
        efa_row.addWidget(self.spin_efa_eigen)
        efa_row.addStretch(1)
        data_layout.addLayout(efa_row)

        left_layout.addWidget(self._data_group)

        # Sub-tabs: Model / Optimization / Plots
        self.model_opt_plots = ModelOptPlotsWidget(left_container)
        self.model_opt_plots.model_defined.connect(self._on_model_defined)
        left_layout.addWidget(self.model_opt_plots, 1)

        # Actions row (Tauri-like)
        actions_group = QGroupBox("", left_container)
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_import = QPushButton("Import Config", actions_group)
        self.btn_import.clicked.connect(self._on_import_clicked)
        actions_layout.addWidget(self.btn_import)

        self.btn_export = QPushButton("Export Config", actions_group)
        self.btn_export.clicked.connect(self._on_export_clicked)
        actions_layout.addWidget(self.btn_export)

        actions_layout.addStretch(1)

        self.btn_reset = QPushButton("Reset Calculation", actions_group)
        self.btn_reset.clicked.connect(self.reset_tab)
        actions_layout.addWidget(self.btn_reset)

        self.btn_process = QPushButton("Process Data", actions_group)
        self.btn_process.clicked.connect(self._on_process_clicked)
        actions_layout.addWidget(self.btn_process)

        self.btn_cancel = QPushButton("Cancel", actions_group)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_cancel.setEnabled(False)
        actions_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save results", actions_group)
        self.btn_save.clicked.connect(self._on_save_results_clicked)
        self.btn_save.setEnabled(False)
        actions_layout.addWidget(self.btn_save)

        left_layout.addWidget(actions_group)

        # ----- Right panel: main plot + diagnostics -----
        right_split = QSplitter(Qt.Orientation.Vertical, self._main_split)

        main_plot_panel = QWidget(right_split)
        main_plot_layout = QVBoxLayout(main_plot_panel)
        main_plot_layout.setContentsMargins(0, 0, 0, 0)

        nav = QHBoxLayout()
        self.btn_plot_prev = QPushButton("Prev", main_plot_panel)
        self.btn_plot_prev.clicked.connect(lambda: self._navigate_plot(-1))
        nav.addWidget(self.btn_plot_prev)
        self.lbl_plot_title = QLabel("Main spectra / titration plot", main_plot_panel)
        self.lbl_plot_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.lbl_plot_title, 1)
        self.btn_plot_next = QPushButton("Next", main_plot_panel)
        self.btn_plot_next.clicked.connect(lambda: self._navigate_plot(1))
        nav.addWidget(self.btn_plot_next)
        main_plot_layout.addLayout(nav)

        self.canvas_main = MplCanvas(main_plot_panel)
        main_plot_layout.addWidget(self.canvas_main, 1)
        main_plot_layout.addWidget(NavigationToolbar(self.canvas_main, main_plot_panel))

        diag_panel = QWidget(right_split)
        diag_layout = QVBoxLayout(diag_panel)
        diag_layout.setContentsMargins(0, 0, 0, 0)
        diag_layout.addWidget(QLabel("Diagnostics / Log", diag_panel))

        log_controls = QHBoxLayout()
        self.chk_autoscroll = QCheckBox("Autoscroll", diag_panel)
        self.chk_autoscroll.setChecked(True)
        log_controls.addWidget(self.chk_autoscroll)
        log_controls.addStretch(1)
        self.btn_clear_log = QPushButton("Clear log", diag_panel)
        log_controls.addWidget(self.btn_clear_log)
        diag_layout.addLayout(log_controls)

        self.log = LogConsole(diag_panel)
        self.log.setMinimumHeight(160)
        self.chk_autoscroll.toggled.connect(self.log.set_autoscroll)
        self.btn_clear_log.clicked.connect(self.log.clear)
        diag_layout.addWidget(self.log, 1)

        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

        self._main_split.setStretchFactor(0, 0)
        self._main_split.setStretchFactor(1, 1)

        self._update_plot_nav()
        self._apply_channels_mode_ui()

    # ---- Plot navigation (Prev/Next) ----
    def _plot_title_for_key(self, key: str) -> str:
        titles = {
            "fit": "Experimental vs fitted spectra",
            "concentrations": "Concentration profile",
            "absorptivities": "Molar absorptivities",
            "eigenvalues": "EFA eigenvalues",
            "efa": "EFA forward/backward",
            "residuals": "Residuals",
        }
        return titles.get(str(key), str(key))

    def _wire_plot_controls(self) -> None:
        if self._plot_controller is not None:
            self._plot_controller.wire_controls()

    def _reset_plot_state(self) -> None:
        if self._plot_controller is not None:
            self._plot_controller.reset()

    def _build_plot_state_from_result(self, result: dict[str, Any]) -> None:
        if self._plot_controller is not None:
            self._plot_controller.build_from_result(result)

    def _render_current_plot(self) -> None:
        if self._plot_controller is not None:
            self._plot_controller.render_current_plot()

    def _update_plot_nav(self) -> None:
        if self._plot_controller is not None:
            self._plot_controller.update_plot_nav()

    def _navigate_plot(self, delta: int) -> None:
        if self._plot_controller is not None:
            self._plot_controller.navigate(delta)

    # ---- Excel / Data selection ----
    def _on_choose_file_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel file", "", "Excel (*.xlsx)")
        if not path:
            return
        self._set_file_path(path)

    def _set_file_path(self, file_path: str) -> None:
        path = Path(str(file_path or ""))
        if not path.exists():
            QMessageBox.warning(self, "Missing file", f"Excel file not found:\n{path}")
            return
        if path.suffix.lower() != ".xlsx":
            QMessageBox.warning(self, "Unsupported file", "Only .xlsx files are supported.")
            return

        self._file_path = str(path)
        self.lbl_file_status.setText(self._file_path)
        self._load_sheets()

    def _load_sheets(self) -> None:
        if not self._file_path:
            return
        try:
            sheets = _list_excel_sheets(self._file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            return

        sheets_with_blank = [""] + sheets
        self.combo_spectra_sheet.blockSignals(True)
        self.combo_conc_sheet.blockSignals(True)
        self.combo_spectra_sheet.clear()
        self.combo_conc_sheet.clear()
        self.combo_spectra_sheet.addItems(sheets_with_blank)
        self.combo_conc_sheet.addItems(sheets_with_blank)
        self.combo_spectra_sheet.setCurrentIndex(0)
        self.combo_conc_sheet.setCurrentIndex(0)
        self.combo_spectra_sheet.blockSignals(False)
        self.combo_conc_sheet.blockSignals(False)

        self._axis_values = []
        self._conc_points_count = 0
        self.lbl_channels_range.setText("")
        self._populate_channels_list([])
        self._clear_conc_columns()

    def _on_spectra_sheet_changed(self) -> None:
        sheet = self.combo_spectra_sheet.currentText().strip()
        if not self._file_path or not sheet:
            self._axis_values = []
            self.lbl_channels_range.setText("")
            self._populate_channels_list([])
            self._update_channels_usage()
            return
        try:
            axis = _list_spectroscopy_axis_values(self._file_path, sheet)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            self._axis_values = []
            self.lbl_channels_range.setText("")
            self._populate_channels_list([])
            self._update_channels_usage()
            return
        self._axis_values = axis
        if axis:
            self.lbl_channels_range.setText(f"{_axis_range_text(axis)} ({len(axis)})")
        else:
            self.lbl_channels_range.setText("")
        self._populate_channels_list(axis)
        if str(self.combo_channels_mode.currentData() or "all") == "all":
            self._set_list_checked(self.list_channels, True)
        self._update_channels_usage()

    def _on_conc_sheet_changed(self) -> None:
        sheet = self.combo_conc_sheet.currentText().strip()
        if not self._file_path or not sheet:
            self._conc_points_count = 0
            self._clear_conc_columns()
            self._update_efa_eigen_range()
            return
        try:
            cols = _list_excel_columns(self._file_path, sheet)
            self._conc_points_count = _count_excel_rows(self._file_path, sheet)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            self._conc_points_count = 0
            self._clear_conc_columns()
            self._update_efa_eigen_range()
            return
        self._populate_conc_columns(cols)
        self._update_efa_eigen_range()

    def _clear_conc_columns(self) -> None:
        while self._columns_layout.count():
            item = self._columns_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.model_opt_plots.set_available_conc_columns([])

    def _populate_conc_columns(self, columns: list[str]) -> None:
        self._clear_conc_columns()
        if not columns:
            self._columns_layout.addWidget(QLabel("Select a concentration sheet to load columns…", self._columns_widget))
            return

        from PySide6.QtWidgets import QCheckBox

        for col in columns:
            cb = QCheckBox(str(col), self._columns_widget)
            cb.setChecked(True)  # match Tauri default (all enabled if no selection)
            cb.toggled.connect(self._on_conc_columns_toggled)
            self._columns_layout.addWidget(cb)

        self._columns_layout.addStretch(1)
        self._on_conc_columns_toggled()

    def _on_conc_columns_toggled(self) -> None:
        selected = self._selected_conc_columns()
        self.model_opt_plots.set_available_conc_columns(selected)
        if self._last_result is not None:
            self._render_current_plot()

    def _selected_conc_columns(self) -> list[str]:
        from PySide6.QtWidgets import QCheckBox

        selected: list[str] = []
        for i in range(self._columns_layout.count()):
            w = self._columns_layout.itemAt(i).widget()
            if isinstance(w, QCheckBox) and w.isChecked():
                selected.append(str(w.text()))
        return selected

    # ---- Channels ----
    def _on_channels_mode_changed(self) -> None:
        self._apply_channels_mode_ui()

    def _apply_channels_mode_ui(self) -> None:
        mode = str(self.combo_channels_mode.currentData() or "all")
        is_custom = mode == "custom"
        self._channels_custom_box.setEnabled(bool(is_custom))
        self.edit_channels_spec.setEnabled(bool(is_custom))
        self.btn_apply_channels.setEnabled(bool(is_custom))

        if not is_custom:
            # Mirror Tauri: "All" is explicit.
            self.edit_channels_spec.setText("All")

        # EFA requires full spectrum
        if is_custom:
            if self.chk_efa.isChecked():
                self.chk_efa.setChecked(False)
            self.chk_efa.setEnabled(False)
            self.chk_efa.setToolTip("EFA requiere espectro completo (Channels=All).")
        else:
            self.chk_efa.setEnabled(True)
            self.chk_efa.setToolTip("")

        self._update_channels_usage()

    def _on_apply_channels_clicked(self) -> None:
        raw = str(self.edit_channels_spec.text() or "").strip()
        try:
            self._apply_channels_spec(raw)
        except Exception as exc:
            msg = str(exc)
            self.log.append_text(f"ERROR: {msg}")
            QMessageBox.warning(self, "Channels error", msg)

    def _apply_channels_spec(self, raw: str) -> None:
        resolved_info = parse_channel_spec(raw, self._axis_values, tol=DEFAULT_CHANNEL_TOL)
        if resolved_info.mode == "all":
            ix = self.combo_channels_mode.findData("all")
            if ix >= 0:
                self.combo_channels_mode.setCurrentIndex(ix)
            self._apply_channels_mode_ui()
            if self.list_channels.count():
                self._set_list_checked(self.list_channels, True)
            return

        errors = list(resolved_info.errors or [])
        if errors:
            raise ValueError("\n".join(errors))

        if not resolved_info:
            raise ValueError("No channels matched.")
        for line in resolved_info.mapping_lines:
            self.log.append_text(line)

        ix = self.combo_channels_mode.findData("custom")
        if ix >= 0:
            self.combo_channels_mode.setCurrentIndex(ix)
        self._apply_channels_mode_ui()

        target_set = {float(v) for v in resolved_info}
        self.list_channels.setUpdatesEnabled(False)
        self.list_channels.blockSignals(True)
        try:
            for i in range(self.list_channels.count()):
                it = self.list_channels.item(i)
                if it is None:
                    continue
                v = float(it.data(Qt.ItemDataRole.UserRole))
                it.setCheckState(Qt.CheckState.Checked if v in target_set else Qt.CheckState.Unchecked)
        finally:
            self.list_channels.blockSignals(False)
            self.list_channels.setUpdatesEnabled(True)

        self._update_channels_usage()

    def _populate_channels_list(self, axis_values: list[float]) -> None:
        self.list_channels.blockSignals(True)
        self.list_channels.clear()
        for v in axis_values:
            item = QListWidgetItem(_format_axis_val(v), self.list_channels)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, float(v))
        self.list_channels.blockSignals(False)

    def _set_list_checked(self, widget: QListWidget, checked: bool) -> None:
        widget.setUpdatesEnabled(False)
        widget.blockSignals(True)
        try:
            state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            for i in range(widget.count()):
                it = widget.item(i)
                if it is not None:
                    it.setCheckState(state)
        finally:
            widget.blockSignals(False)
            widget.setUpdatesEnabled(True)
        self._update_channels_usage()

    def _on_channels_selection_changed(self, _item: QListWidgetItem) -> None:
        self._update_channels_usage()

    def _selected_channels(self) -> list[float]:
        out: list[float] = []
        for i in range(self.list_channels.count()):
            it = self.list_channels.item(i)
            if it is None:
                continue
            if it.checkState() == Qt.CheckState.Checked:
                try:
                    out.append(float(it.data(Qt.ItemDataRole.UserRole)))
                except Exception:
                    continue
        return out

    def _update_channels_usage(self) -> None:
        total = len(self._axis_values)
        mode = str(self.combo_channels_mode.currentData() or "all")
        if not total:
            self.lbl_channels_usage.setText("")
            self._update_efa_eigen_range()
            return
        range_text = _axis_range_text(self._axis_values)
        if mode == "all":
            used = total
        else:
            used = len(self._selected_channels())
        self.lbl_channels_usage.setText(f"Using {used} / {total} ({range_text})")
        self._update_efa_eigen_range()

    def _update_efa_eigen_range(self) -> None:
        n_points = int(self._conc_points_count or 0)
        total_channels = len(self._axis_values)
        mode = str(self.combo_channels_mode.currentData() or "all")
        if mode == "all":
            used_channels = total_channels
        else:
            used_channels = len(self._selected_channels())
        n_max = max(0, min(n_points, used_channels))
        self.spin_efa_eigen.setRange(0, n_max)
        if self.spin_efa_eigen.value() > n_max:
            self.spin_efa_eigen.setValue(n_max)

    # ---- Model / Optimization sync ----
    def _on_model_defined(self, _n_components: int, _n_species: int) -> None:
        # Parameter grid is already updated inside ModelOptPlotsWidget.
        pass

    # ---- Run / Cancel / Results ----
    def _set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        self.btn_process.setEnabled(not running)
        self.btn_choose_file.setEnabled(not running)
        self.combo_spectra_sheet.setEnabled(not running)
        self.combo_conc_sheet.setEnabled(not running)
        self.combo_channels_mode.setEnabled(not running)
        self.list_channels.setEnabled(not running)
        self.btn_channels_all.setEnabled(not running)
        self.btn_channels_none.setEnabled(not running)
        self.chk_efa.setEnabled(not running)
        self.spin_efa_eigen.setEnabled(not running)
        self.model_opt_plots.setEnabled(not running)
        self.btn_import.setEnabled(not running)
        self.btn_export.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_save.setEnabled(bool(self._last_result) and bool(self._last_result.get("success", True)) and not running)

        self.btn_process.setText("Processing..." if running else "Process Data")
        if not running:
            # Restore EFA enable/disable according to Channels mode.
            self._apply_channels_mode_ui()
        self._update_plot_nav()

    def _collect_config(self) -> dict[str, Any]:
        if not self._file_path:
            raise ValueError("No Excel file selected.")

        spectra_sheet = self.combo_spectra_sheet.currentText().strip()
        conc_sheet = self.combo_conc_sheet.currentText().strip()
        if not spectra_sheet or not conc_sheet:
            raise ValueError("Select Spectra and Concentration sheets.")

        column_names = self._selected_conc_columns()
        if not column_names:
            raise ValueError("Select at least one concentration column in 'Column names'.")

        channels_mode = str(self.combo_channels_mode.currentData() or "all")
        channels_custom: list[float] = []
        channels_resolved: list[float] = []
        if channels_mode == "custom":
            channels_resolved = self._selected_channels()
            if not channels_resolved:
                raise ValueError("Channels=Custom requires selecting at least one channel.")

            raw_spec = str(self.edit_channels_spec.text() or "").strip()
            if not raw_spec:
                raw_spec = ", ".join(_format_axis_val(v) for v in channels_resolved)

            parsed = parse_channels_spec(raw_spec)
            if parsed.get("mode") == "custom" and not list(parsed.get("errors") or []):
                for tok in list(parsed.get("tokens") or []):
                    if tok.get("type") == "value":
                        channels_custom.append(float(tok.get("value")))
                    elif tok.get("type") == "range":
                        channels_custom.extend([float(tok.get("min")), float(tok.get("max"))])
            else:
                channels_custom = list(channels_resolved)
            channels_raw = raw_spec
        else:
            channels_raw = "All"

        # EFA must be off for Custom channels (wx/Tauri behavior)
        efa_enabled = bool(self.chk_efa.isChecked()) and channels_mode == "all"
        efa_eigenvalues = int(self.spin_efa_eigen.value())

        state = self.model_opt_plots.collect_state()
        receptor_label = state.receptor_label
        guest_label = state.guest_label
        if not receptor_label or not guest_label:
            raise ValueError("Select Receptor and Guest roles (or keep Auto with valid column names).")
        if receptor_label == guest_label:
            raise ValueError("Receptor and Guest cannot be the same column.")

        config = {
            "file_path": self._file_path,
            "spectra_sheet": spectra_sheet,
            "conc_sheet": conc_sheet,
            "column_names": column_names,
            "receptor_label": receptor_label,
            "guest_label": guest_label,
            "efa_enabled": efa_enabled,
            "efa_eigenvalues": efa_eigenvalues,
            "modelo": state.modelo,
            "non_abs_species": state.non_abs_species,
            "algorithm": state.algorithm,
            "model_settings": state.model_settings,
            "optimizer": state.optimizer,
            "initial_k": state.initial_k,
            "bounds": state.bounds,
            "fixed_mask": state.fixed_mask,
            "channels_mode": channels_mode,
            "channels_custom": channels_custom,
            "channels_raw": channels_raw,
            "channels_resolved": channels_resolved,
        }
        return config

    def _on_process_clicked(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "A fit is already running. Cancel it first.")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            self.log.append_text(f"ERROR: {exc}")
            QMessageBox.warning(self, "Config error", str(exc))
            return

        self.log.append_text("Iniciando optimización…")
        self._last_result = None
        self._reset_plot_state()
        self.btn_save.setEnabled(False)
        self.canvas_main.clear()

        self._worker = FitWorker(run_spectroscopy_fit, config=config, parent=self)
        self._thread = self._worker.thread()
        self._worker.progress.connect(self._on_worker_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.result.connect(self._on_fit_result, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_fit_error, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_fit_finished, Qt.ConnectionType.QueuedConnection)
        self._set_running(True)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is None:
            return
        self._worker.request_cancel()
        self.log.append_text("Cancel requested…")

    @Slot(str)
    def _on_worker_progress(self, msg: str) -> None:
        # Runs in the main thread via queued connection (worker emits from its QThread).
        self.log.append_text(str(msg))

    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._last_result = result
        self.btn_save.setEnabled(bool(result.get("success", True)))

        self._build_plot_state_from_result(result)
        self._render_current_plot()

        results_text = result.get("results_text") or ""
        if results_text:
            self.log.append_text("\n" + str(results_text).rstrip() + "\n")

        stats = result.get("statistics") or {}
        rms = stats.get("RMS")
        if rms is not None:
            try:
                self.log.append_text(f"Finalizado. RMS={float(rms):.6g}")
            except Exception:
                self.log.append_text("Finalizado.")
        else:
            self.log.append_text("Finalizado.")

    @Slot(str)
    def _on_fit_error(self, message: str) -> None:
        self.log.append_text(f"ERROR: {message}")
        QMessageBox.critical(self, "Fit error", str(message))

    def _on_fit_finished(self) -> None:
        self._worker = None
        self._thread = None
        self._set_running(False)

    # ---- Import / Export / Reset / Save ----
    def _on_export_clicked(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export config", "spectroscopy_config.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.log.append_text(f"Config exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _on_import_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import config", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return
        try:
            self.load_config(config)
            self.log.append_text(f"Config imported from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Config error", str(exc))

    def load_config(self, config: dict[str, Any]) -> None:
        if self._worker is not None:
            raise RuntimeError("Cancel the running fit before importing config.")

        file_path = str(config.get("file_path") or "")
        if not file_path:
            raise ValueError("Config missing 'file_path'.")
        if not Path(file_path).exists():
            raise ValueError(f"Excel file not found: {file_path}")

        self._set_file_path(file_path)

        spectra_sheet = str(config.get("spectra_sheet") or "")
        conc_sheet = str(config.get("conc_sheet") or "")
        if spectra_sheet:
            ix = self.combo_spectra_sheet.findText(spectra_sheet)
            if ix >= 0:
                self.combo_spectra_sheet.setCurrentIndex(ix)
        if conc_sheet:
            ix = self.combo_conc_sheet.findText(conc_sheet)
            if ix >= 0:
                self.combo_conc_sheet.setCurrentIndex(ix)

        # Ensure dependent UI is loaded
        self._on_spectra_sheet_changed()
        self._on_conc_sheet_changed()

        missing: list[str] = []

        # Restore column_names
        wanted_cols = {str(c) for c in (config.get("column_names") or [])}
        if wanted_cols:
            from PySide6.QtWidgets import QCheckBox

            available = set()
            for i in range(self._columns_layout.count()):
                w = self._columns_layout.itemAt(i).widget()
                if isinstance(w, QCheckBox):
                    available.add(str(w.text()))
                    w.setChecked(str(w.text()) in wanted_cols)

            missing_cols = sorted(wanted_cols - available)
            if missing_cols:
                missing.append(f"Missing concentration columns: {missing_cols}")

        self._on_conc_columns_toggled()

        # Restore channels
        mode = str(config.get("channels_mode") or "all").strip().lower()
        if mode not in {"all", "custom"}:
            mode = "all"
        raw_spec = str(config.get("channels_raw") or "").strip()
        mode_ix = self.combo_channels_mode.findData(mode)
        if mode_ix >= 0:
            self.combo_channels_mode.setCurrentIndex(mode_ix)
        self._apply_channels_mode_ui()

        applied = False
        if mode == "custom" and raw_spec:
            self.edit_channels_spec.setText(raw_spec)
            try:
                self._apply_channels_spec(raw_spec)
                applied = True
            except Exception as exc:
                missing.append(f"Channels spec could not be applied: {exc}")

        if mode == "custom" and not applied:
            wanted = [float(x) for x in (config.get("channels_resolved") or [])]
            if wanted and not raw_spec:
                self.edit_channels_spec.setText(", ".join(_format_axis_val(v) for v in wanted))

            tol = 1e-6
            self.list_channels.setUpdatesEnabled(False)
            self.list_channels.blockSignals(True)
            try:
                for i in range(self.list_channels.count()):
                    it = self.list_channels.item(i)
                    if it is None:
                        continue
                    v = float(it.data(Qt.ItemDataRole.UserRole))
                    it.setCheckState(
                        Qt.CheckState.Checked if any(abs(v - t) <= tol for t in wanted) else Qt.CheckState.Unchecked
                    )
            finally:
                self.list_channels.blockSignals(False)
                self.list_channels.setUpdatesEnabled(True)

            if wanted and not self._selected_channels():
                missing.append("Channels selection could not be matched to the Spectra axis values.")
            self._update_channels_usage()

        # Restore EFA
        self.spin_efa_eigen.setValue(int(config.get("efa_eigenvalues") or 0))
        self.chk_efa.setChecked(bool(config.get("efa_enabled", False)) and mode == "all")
        self._apply_channels_mode_ui()

        # Restore Model/Optimization
        modelo = config.get("modelo") or []
        n_rows = len(modelo) if isinstance(modelo, list) else 0
        n_cols = len(modelo[0]) if n_rows and isinstance(modelo[0], list) else 0
        n_species = max(0, n_rows - n_cols)
        state = ModelOptPlotsState(
            n_components=n_cols,
            n_species=n_species,
            modelo=[[float(x or 0.0) for x in (row or [])] for row in (modelo or [])],
            non_abs_species=list(config.get("non_abs_species") or []),
            algorithm=str(config.get("algorithm") or "Newton-Raphson"),
            model_settings=str(config.get("model_settings") or "Free"),
            optimizer=str(config.get("optimizer") or "powell"),
            initial_k=list(config.get("initial_k") or []),
            bounds=list(config.get("bounds") or []),
            fixed_mask=list(config.get("fixed_mask") or []),
            receptor_label=str(config.get("receptor_label") or ""),
            guest_label=str(config.get("guest_label") or ""),
        )
        self.model_opt_plots.apply_state(state)

        if missing:
            msg = "\n".join(missing)
            self.log.append_text("WARNING:\n" + msg)
            QMessageBox.warning(self, "Config partially applied", msg)

    def reset_tab(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "Cancel the running fit before resetting.")
            return

        self._file_path = ""
        self.lbl_file_status.setText("No file selected")
        self.combo_spectra_sheet.clear()
        self.combo_conc_sheet.clear()
        self._axis_values = []
        self._conc_points_count = 0
        self._populate_channels_list([])
        self.combo_channels_mode.setCurrentIndex(self.combo_channels_mode.findData("all"))
        self.chk_efa.setChecked(True)
        self.spin_efa_eigen.setValue(0)
        self._apply_channels_mode_ui()

        self._clear_conc_columns()
        self.model_opt_plots.reset()

        self.canvas_main.clear()
        self._reset_plot_state()
        self.log.clear()
        self._last_result = None
        self.btn_save.setEnabled(False)

    def _on_save_results_clicked(self) -> None:
        if not self._last_result or not bool(self._last_result.get("success", True)):
            QMessageBox.information(self, "No results", "Run 'Process Data' first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save results", "hmfit_results.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            write_results_xlsx(
                path,
                constants=self._last_result.get("constants") or [],
                statistics=self._last_result.get("statistics") or {},
                results_text=str(self._last_result.get("results_text") or ""),
                export_data=self._last_result.get("export_data") or {},
            )
            self.log.append_text(f"Results saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
