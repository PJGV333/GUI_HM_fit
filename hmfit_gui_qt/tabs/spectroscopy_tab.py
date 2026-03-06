from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    _MIN_CHANNELS_FOR_EFA = 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FitWorker | None = None
        self._thread: QThread | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_config: dict[str, Any] | None = None
        self._last_fit_context: dict[str, Any] | None = None
        self._graph_solver_inputs: dict[str, Any] | None = None
        self._plot_controller: PlotController | None = None
        self._axis_values: list[float] = []
        self._conc_points_count = 0
        self._file_path: str = ""
        self._is_running = False
        self._render_graphs_enabled = True
        self._render_quality = "preview"
        self._multi_start_parallel = False
        self._multi_start_max_workers: int | None = None
        self._log_buffer: list[str] = []
        self._log_buffered_count = 0
        self._log_flush_count = 0
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.setInterval(90)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)

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

        # Baseline controls
        baseline_row = QHBoxLayout()
        baseline_row.addWidget(QLabel("Baseline", self._data_group))
        self.combo_baseline_mode = QComboBox(self._data_group)
        self.combo_baseline_mode.addItem("Off", "off")
        self.combo_baseline_mode.addItem("Range", "range")
        self.combo_baseline_mode.addItem("Auto", "auto")
        baseline_row.addWidget(self.combo_baseline_mode)
        baseline_row.addWidget(QLabel("Start", self._data_group))
        self.edit_baseline_start = QLineEdit(self._data_group)
        self.edit_baseline_start.setText("450.0")
        self.edit_baseline_start.setMaximumWidth(90)
        baseline_row.addWidget(self.edit_baseline_start)
        baseline_row.addWidget(QLabel("End", self._data_group))
        self.edit_baseline_end = QLineEdit(self._data_group)
        self.edit_baseline_end.setText("600.0")
        self.edit_baseline_end.setMaximumWidth(90)
        baseline_row.addWidget(self.edit_baseline_end)
        baseline_row.addWidget(QLabel("q", self._data_group))
        self.spin_baseline_auto_q = QDoubleSpinBox(self._data_group)
        self.spin_baseline_auto_q.setDecimals(3)
        self.spin_baseline_auto_q.setRange(0.0, 1.0)
        self.spin_baseline_auto_q.setSingleStep(0.05)
        self.spin_baseline_auto_q.setValue(0.20)
        baseline_row.addWidget(self.spin_baseline_auto_q)
        baseline_row.addStretch(1)
        data_layout.addLayout(baseline_row)

        # Weighting controls
        weighting_row = QHBoxLayout()
        weighting_row.addWidget(QLabel("Weighting", self._data_group))
        self.combo_weighting_mode = QComboBox(self._data_group)
        self.combo_weighting_mode.addItem("none", "none")
        self.combo_weighting_mode.addItem("std", "std")
        self.combo_weighting_mode.addItem("max", "max")
        weighting_row.addWidget(self.combo_weighting_mode)
        weighting_row.addWidget(QLabel("Power", self._data_group))
        self.edit_weighting_power = QLineEdit(self._data_group)
        self.edit_weighting_power.setText("1.0")
        self.edit_weighting_power.setMaximumWidth(90)
        weighting_row.addWidget(self.edit_weighting_power)
        self.chk_weighting_normalize = QCheckBox("Normalize", self._data_group)
        self.chk_weighting_normalize.setChecked(True)
        weighting_row.addWidget(self.chk_weighting_normalize)
        weighting_row.addStretch(1)
        data_layout.addLayout(weighting_row)

        # Epsilon solver controls
        eps_row = QHBoxLayout()
        eps_row.addWidget(QLabel("ε solver", self._data_group))
        self.combo_eps_solver_mode = QComboBox(self._data_group)
        self.combo_eps_solver_mode.addItem("soft_penalty", "soft_penalty")
        self.combo_eps_solver_mode.addItem("soft_bound", "soft_bound")
        self.combo_eps_solver_mode.addItem("nnls_hard", "nnls_hard")
        eps_row.addWidget(self.combo_eps_solver_mode)
        eps_row.addWidget(QLabel("mu", self._data_group))
        self.edit_eps_mu = QLineEdit(self._data_group)
        self.edit_eps_mu.setText("0.01")
        self.edit_eps_mu.setMaximumWidth(90)
        eps_row.addWidget(self.edit_eps_mu)
        eps_row.addWidget(QLabel("delta_rel", self._data_group))
        self.edit_delta_rel = QLineEdit(self._data_group)
        self.edit_delta_rel.setText("0.01")
        self.edit_delta_rel.setMaximumWidth(90)
        eps_row.addWidget(self.edit_delta_rel)
        eps_row.addWidget(QLabel("alpha_smooth", self._data_group))
        self.edit_alpha_smooth = QLineEdit(self._data_group)
        self.edit_alpha_smooth.setText("0.0")
        self.edit_alpha_smooth.setMaximumWidth(90)
        eps_row.addWidget(self.edit_alpha_smooth)
        eps_row.addStretch(1)
        data_layout.addLayout(eps_row)

        self.combo_baseline_mode.currentIndexChanged.connect(self._update_preprocess_controls)
        self.combo_weighting_mode.currentIndexChanged.connect(self._update_preprocess_controls)
        self.combo_eps_solver_mode.currentIndexChanged.connect(self._update_preprocess_controls)
        self.combo_baseline_mode.setToolTip("Resta una línea base por espectro (off/range/auto).")
        self.edit_baseline_start.setToolTip("Baseline range: inicio en nm.")
        self.edit_baseline_end.setToolTip("Baseline range: fin en nm.")
        self.spin_baseline_auto_q.setToolTip("Baseline auto: cuantíl de λ con menor variación.")
        self.combo_weighting_mode.setToolTip("Reduce la influencia de longitudes con señal ~0.")
        self.edit_weighting_power.setToolTip("Exponente aplicado a pesos espectrales.")
        self.combo_eps_solver_mode.setToolTip("Soft bound permite ε ligeramente negativa hasta -δ.")
        self.edit_eps_mu.setToolTip("Penalización suave para ε negativas.")
        self.edit_delta_rel.setToolTip("δ relativo para límite inferior suave de ε.")
        self.edit_alpha_smooth.setToolTip("Suaviza ε(λ) penalizando curvatura (2ª derivada).")
        self._update_preprocess_controls()

        left_layout.addWidget(self._data_group)

        # Sub-tabs: Model / Optimization / Plots
        self.model_opt_plots = ModelOptPlotsWidget(left_container, enable_equation_editor=True)
        self.model_opt_plots.model_defined.connect(self._on_model_defined)
        self.equation_editor = self.model_opt_plots.equation_editor
        if self.equation_editor is not None:
            self.equation_editor.model_parsed.connect(self._on_equation_model_parsed)
        left_layout.addWidget(self.model_opt_plots, 1)

        # Actions row (historical reference to previous UI buttons)
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

        self.btn_render_graphs = QPushButton("Render graphs", actions_group)
        self.btn_render_graphs.clicked.connect(self._on_render_graphs_clicked)
        self.btn_render_graphs.setEnabled(False)
        actions_layout.addWidget(self.btn_render_graphs)

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
        
        self.chk_show_diag = QCheckBox("Show stability diagnostics", diag_panel)
        self.chk_show_diag.setChecked(False)
        log_controls.addWidget(self.chk_show_diag)

        self.lbl_stability_light = QLabel("Stability: -", diag_panel)
        self.lbl_stability_light.setStyleSheet("font-weight: bold; margin-left: 10px;")
        log_controls.addWidget(self.lbl_stability_light)

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
            cb.setChecked(True)  # match historical default (all enabled if no selection)
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
            # Mirror historical behavior: "All" is explicit.
            self.edit_channels_spec.setText("All")

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

    def _effective_selected_channels(self) -> list[float]:
        mode = str(self.combo_channels_mode.currentData() or "all")
        if mode == "all":
            return list(self._axis_values or [])
        return self._selected_channels()

    def _update_efa_enabled_state(self) -> None:
        num_channels = len(self._effective_selected_channels())
        can_enable = (num_channels >= self._MIN_CHANNELS_FOR_EFA) and (not self._is_running)
        if not can_enable and self.chk_efa.isChecked():
            self.chk_efa.setChecked(False)
        self.chk_efa.setEnabled(can_enable)
        if can_enable:
            self.chk_efa.setToolTip("")
        else:
            self.chk_efa.setToolTip(
                f"EFA requiere al menos {self._MIN_CHANNELS_FOR_EFA} canales seleccionados "
                f"(actual: {num_channels})."
            )

    def _update_channels_usage(self) -> None:
        total = len(self._axis_values)
        if not total:
            self.lbl_channels_usage.setText("")
            self._update_efa_eigen_range()
            self._update_efa_enabled_state()
            return
        range_text = _axis_range_text(self._axis_values)
        used = len(self._effective_selected_channels())
        self.lbl_channels_usage.setText(f"Using {used} / {total} ({range_text})")
        self._update_efa_eigen_range()
        self._update_efa_enabled_state()

    def _update_efa_eigen_range(self) -> None:
        n_points = int(self._conc_points_count or 0)
        used_channels = len(self._effective_selected_channels())
        n_max = max(0, min(n_points, used_channels))
        self.spin_efa_eigen.setRange(0, n_max)
        if self.spin_efa_eigen.value() > n_max:
            self.spin_efa_eigen.setValue(n_max)

    def _parse_float(self, text: str, default: float) -> float:
        try:
            return float(str(text).strip())
        except Exception:
            return float(default)

    def _update_preprocess_controls(self) -> None:
        baseline_mode = str(self.combo_baseline_mode.currentData() or "off")
        self.edit_baseline_start.setEnabled(baseline_mode == "range")
        self.edit_baseline_end.setEnabled(baseline_mode == "range")
        self.spin_baseline_auto_q.setEnabled(baseline_mode == "auto")

        weighting_mode = str(self.combo_weighting_mode.currentData() or "none")
        weighting_active = weighting_mode != "none"
        self.edit_weighting_power.setEnabled(weighting_active)
        self.chk_weighting_normalize.setEnabled(weighting_active)

        solver_mode = str(self.combo_eps_solver_mode.currentData() or "soft_penalty")
        self.edit_eps_mu.setEnabled(solver_mode in {"soft_penalty", "soft_bound"})
        self.edit_delta_rel.setEnabled(solver_mode == "soft_bound")

    # ---- Model / Optimization sync ----
    def _on_model_defined(self, _n_components: int, _n_species: int) -> None:
        # Parameter grid is already updated inside ModelOptPlotsWidget.
        pass

    @Slot(object)
    def _on_equation_model_parsed(self, solver_inputs_obj: object) -> None:
        if not isinstance(solver_inputs_obj, dict):
            return
        solver_inputs = dict(solver_inputs_obj)
        self._graph_solver_inputs = solver_inputs
        try:
            self._apply_solver_inputs_to_classic_matrix(solver_inputs)
        except Exception as exc:
            self.log.append_text(f"[Graph] Could not apply parsed model: {exc}")

    def _apply_solver_inputs_to_classic_matrix(self, solver_inputs: dict[str, Any]) -> None:
        solver_block_raw = solver_inputs.get("solver_inputs")
        solver_block = solver_block_raw if isinstance(solver_block_raw, dict) else {}

        model_raw = solver_block.get("modelo")
        model = np.asarray(model_raw if model_raw is not None else [], dtype=float)
        if model.ndim != 2 or model.size == 0:
            raise ValueError("Parsed model matrix is empty.")

        n_components, nspec_total = model.shape
        if n_components <= 0 or nspec_total <= 0:
            raise ValueError("Parsed model has invalid dimensions.")

        n_complex = max(0, int(nspec_total - n_components))
        self.model_opt_plots.set_model_dimensions(int(n_components), n_complex)
        self.model_opt_plots.set_modelo(model.T.tolist())

        def _extract_species_names(block: object) -> list[str]:
            names_raw: object = block
            if isinstance(block, dict):
                names_raw = block.get("names")
            if not isinstance(names_raw, (list, tuple)):
                return []
            names: list[str] = []
            for value in names_raw:
                text = str(value or "").strip()
                if text:
                    names.append(text)
            return names

        component_names = _extract_species_names(solver_inputs.get("components"))
        complex_names = _extract_species_names(solver_inputs.get("complexes"))
        non_abs_raw = solver_inputs.get("non_abs_species")
        if non_abs_raw is None:
            non_abs_iter: list[object] = []
        elif isinstance(non_abs_raw, np.ndarray):
            non_abs_iter = non_abs_raw.ravel().tolist()
        elif isinstance(non_abs_raw, (list, tuple, set)):
            non_abs_iter = list(non_abs_raw)
        else:
            non_abs_iter = [non_abs_raw]
        non_abs_lookup = {
            str(name or "").strip().casefold()
            for name in non_abs_iter
            if str(name or "").strip()
        }

        nas_raw = solver_block.get("nas")
        nas_from_solver = np.asarray(nas_raw if nas_raw is not None else [], dtype=int).ravel().tolist()
        nas_set = {int(x) for x in nas_from_solver if int(x) >= 0}

        for idx, species_name in enumerate(component_names):
            if idx >= n_components:
                break
            if species_name.casefold() in non_abs_lookup:
                nas_set.add(int(idx))
        for j, species_name in enumerate(complex_names):
            row_idx = int(n_components + j)
            if row_idx >= nspec_total:
                break
            if species_name.casefold() in non_abs_lookup:
                nas_set.add(row_idx)

        self.model_opt_plots.set_non_abs_species(sorted(nas_set))
        self.model_opt_plots.model_table.viewport().update()

        k_raw = solver_block.get("k")
        k_solver = np.asarray(k_raw if k_raw is not None else [], dtype=float).ravel()
        k_values = k_solver.tolist()
        model_sett = str(solver_block.get("model_sett") or "Free")

        # Si el parser vino de una serie por pasos (K1, K2, ...), reflejar en UI
        # exactamente las constantes escritas por el usuario y activar Step by step.
        edge_log_beta_raw = solver_inputs.get("edge_log_beta")
        if edge_log_beta_raw is None:
            edge_log_beta = np.asarray([], dtype=float)
        else:
            edge_log_beta = np.asarray(edge_log_beta_raw, dtype=float).ravel()
        if (
            model_sett == "Free"
            and edge_log_beta.size > 0
            and edge_log_beta.size == k_solver.size
            and np.allclose(np.cumsum(edge_log_beta), k_solver, rtol=1e-8, atol=1e-8)
        ):
            k_values = edge_log_beta.tolist()
            model_sett = "Step by step"

        self.model_opt_plots.set_optimization(
            model_settings=model_sett,
            initial_k=k_values,
            fixed_mask=[False] * len(k_values),
        )

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
        self.combo_baseline_mode.setEnabled(not running)
        self.edit_baseline_start.setEnabled(not running)
        self.edit_baseline_end.setEnabled(not running)
        self.spin_baseline_auto_q.setEnabled(not running)
        self.combo_weighting_mode.setEnabled(not running)
        self.edit_weighting_power.setEnabled(not running)
        self.chk_weighting_normalize.setEnabled(not running)
        self.combo_eps_solver_mode.setEnabled(not running)
        self.edit_eps_mu.setEnabled(not running)
        self.edit_delta_rel.setEnabled(not running)
        self.edit_alpha_smooth.setEnabled(not running)
        if self.equation_editor is not None:
            self.equation_editor.setEnabled(not running)
        self.model_opt_plots.setEnabled(not running)
        self.btn_import.setEnabled(not running)
        self.btn_export.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_save.setEnabled(bool(self._last_result) and bool(self._last_result.get("success", True)) and not running)
        self.btn_render_graphs.setEnabled(
            (not running)
            and bool(self._last_result)
            and (not self._result_has_rendered_graphs(self._last_result))
        )

        self.btn_process.setText("Processing..." if running else "Process Data")
        if not running:
            # Restore EFA enable/disable according to Channels mode.
            self._apply_channels_mode_ui()
            self._update_preprocess_controls()
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

        num_channels = len(self._effective_selected_channels())
        efa_enabled = bool(self.chk_efa.isChecked()) and num_channels >= self._MIN_CHANNELS_FOR_EFA
        efa_eigenvalues = int(self.spin_efa_eigen.value())
        baseline_mode = str(self.combo_baseline_mode.currentData() or "off")
        baseline_start = self._parse_float(self.edit_baseline_start.text(), 450.0)
        baseline_end = self._parse_float(self.edit_baseline_end.text(), 600.0)
        baseline_auto_quantile = float(self.spin_baseline_auto_q.value())
        weighting_mode = str(self.combo_weighting_mode.currentData() or "none")
        weighting_power = self._parse_float(self.edit_weighting_power.text(), 1.0)
        weighting_normalize = bool(self.chk_weighting_normalize.isChecked())
        eps_solver_mode = str(self.combo_eps_solver_mode.currentData() or "soft_penalty")
        eps_mu = self._parse_float(self.edit_eps_mu.text(), 1e-2)
        delta_rel = self._parse_float(self.edit_delta_rel.text(), 0.01)
        alpha_smooth = self._parse_float(self.edit_alpha_smooth.text(), 0.0)
        delta_mode = "relative" if eps_solver_mode == "soft_bound" else "off"

        state = self.model_opt_plots.collect_state()
        receptor_label = state.receptor_label
        guest_label = state.guest_label
        needs_guest = True
        try:
            modelo = np.array(state.modelo, dtype=float).T
            if modelo.ndim == 2 and modelo.size:
                if modelo.shape[0] >= 2:
                    needs_guest = bool(np.any(np.abs(modelo[1, :]) > 0))
                elif modelo.shape[0] == 1:
                    needs_guest = False
        except Exception:
            needs_guest = True
        if not receptor_label:
            raise ValueError("Select Receptor role (or keep Auto with valid column names).")
        if needs_guest and not guest_label:
            raise ValueError("Select Receptor and Guest roles (or keep Auto with valid column names).")
        if needs_guest and receptor_label == guest_label:
            raise ValueError("Receptor and Guest cannot be the same column.")
        if not needs_guest:
            guest_label = ""
            self.model_opt_plots.set_guest_none()

        runs, seeds = self.model_opt_plots.get_multi_start()
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
            "show_stability_diagnostics": self.chk_show_diag.isChecked(),
            "multi_start_runs": runs,
            "baseline_mode": baseline_mode,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "baseline_auto_quantile": baseline_auto_quantile,
            "baseline_apply_per_spectrum": True,
            "weighting_mode": weighting_mode,
            "weighting_eps": 1e-12,
            "weighting_power": weighting_power,
            "weighting_normalize": weighting_normalize,
            "eps_solver_mode": eps_solver_mode,
            "mu": eps_mu,
            "delta_mode": delta_mode,
            "delta_rel": delta_rel,
            "alpha_smooth": alpha_smooth,
            "render_graphs": bool(self._render_graphs_enabled),
            "render_quality": str(self._render_quality),
            "skip_optimization": False,
            "preset_k": None,
            "multi_start_parallel": bool(self._multi_start_parallel),
            "multi_start_max_workers": self._multi_start_max_workers,
            "_progress_throttle_ms": 80,
            "_progress_batch_size": 20,
        }
        config["equation_text"] = (
            self.equation_editor.get_text()
            if hasattr(self, "equation_editor") and self.equation_editor is not None
            else ""
        )
        if not config["equation_text"]:
            config["equation_text"] = ""
        graph_payload = self._graph_solver_inputs if isinstance(self._graph_solver_inputs, dict) else {}
        if graph_payload:
            component_names = list(graph_payload.get("components") or [])
            complex_names = list(graph_payload.get("complexes") or [])
            config["species_names"] = component_names + complex_names
            config["abs_groups"] = graph_payload.get("abs_groups") or {}
        if seeds is not None:
            config["multi_start_seeds"] = seeds
        return config

    @staticmethod
    def _is_critical_log_message(msg: str) -> bool:
        text = str(msg or "").strip().lower()
        if not text:
            return False
        tokens = ("error", "cancel", "cancelled", "critical", "failed", "traceback")
        return any(tok in text for tok in tokens)

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            return
        chunk = "\n".join(self._log_buffer)
        self._log_buffer.clear()
        self._log_flush_count += 1
        self.log.append_text(chunk)

    def _append_log_message(self, msg: str, *, immediate: bool = False) -> None:
        text = str(msg)
        if immediate or self._is_critical_log_message(text):
            self._log_flush_timer.stop()
            self._flush_log_buffer()
            self.log.append_text(text)
            return
        for line in text.splitlines():
            line_text = line.rstrip()
            if line_text:
                self._log_buffer.append(line_text)
                self._log_buffered_count += 1
        if self._log_buffer:
            if len(self._log_buffer) >= 30:
                self._log_flush_timer.stop()
                self._flush_log_buffer()
            elif not self._log_flush_timer.isActive():
                self._log_flush_timer.start()

    @staticmethod
    def _result_has_rendered_graphs(result: dict[str, Any] | None) -> bool:
        if not isinstance(result, dict):
            return False
        graphs = result.get("legacy_graphs") or result.get("graphs") or {}
        if not isinstance(graphs, dict):
            return False
        for key in ("fit", "concentrations", "absorptivities", "eigenvalues", "efa"):
            if graphs.get(key):
                return True
        return False

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

        self._log_flush_timer.stop()
        self._log_buffer.clear()
        self._log_buffered_count = 0
        self._log_flush_count = 0
        self._append_log_message("Iniciando optimización…", immediate=True)
        self._last_config = config
        self._last_result = None
        self._reset_plot_state()
        self.btn_save.setEnabled(False)
        self.canvas_main.clear()

        self._worker = FitWorker(run_spectroscopy_fit, config=config, parent=self)
        self._thread = self._worker.thread()
        self._thread.finished.connect(self._on_fit_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.progress.connect(self._on_worker_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.result.connect(self._on_fit_result, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_fit_error, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_fit_finished, Qt.ConnectionType.QueuedConnection)
        self._set_running(True)
        self._worker.start()

    def _on_render_graphs_clicked(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "A fit is already running. Cancel it first.")
            return
        if not isinstance(self._last_result, dict) or not isinstance(self._last_config, dict):
            QMessageBox.warning(self, "No result", "Run a fit first.")
            return
        export_data = self._last_result.get("export_data") or {}
        k_hat = export_data.get("k") or [c.get("log10K") for c in (self._last_result.get("constants") or [])]
        if not k_hat:
            QMessageBox.warning(self, "Missing constants", "Could not infer fitted constants for deferred render.")
            return

        cfg = dict(self._last_config)
        cfg["render_graphs"] = True
        cfg["skip_optimization"] = True
        cfg["preset_k"] = list(k_hat)
        cfg["multi_start_runs"] = 1
        cfg["multi_start_seeds"] = None
        cfg["_progress_throttle_ms"] = 80
        cfg["_progress_batch_size"] = 20

        self._append_log_message("Generando graficos diferidos...", immediate=True)
        self._worker = FitWorker(run_spectroscopy_fit, config=cfg, parent=self)
        self._thread = self._worker.thread()
        self._thread.finished.connect(self._on_fit_thread_finished, Qt.ConnectionType.QueuedConnection)
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
        self._append_log_message("Cancel requested…", immediate=True)

    @Slot(str)
    def _on_worker_progress(self, msg: str) -> None:
        # Runs in the main thread via queued connection (worker emits from its QThread).
        self._append_log_message(str(msg), immediate=False)

    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._log_flush_timer.stop()
        self._flush_log_buffer()
        self._last_result = result
        self.btn_save.setEnabled(bool(result.get("success", True)))

        self._build_plot_state_from_result(result)
        self._render_current_plot()

        # Update Stability Light
        indicator = result.get("stability_indicator")
        if indicator:
            label = indicator.get("label", "Unknown")
            icon = indicator.get("icon", "")
            cond = indicator.get("cond")
            maxcorr = indicator.get("max_abs_corr")
            reasons = indicator.get("reasons", [])

            info_parts = []
            if cond is not None:
                try:
                    info_parts.append(f"cond={float(cond):.2e}")
                except Exception:
                    info_parts.append("cond=n/a")
            else:
                info_parts.append("cond=n/a")
            if maxcorr is not None:
                try:
                    info_parts.append(f"max|r|={float(maxcorr):.3f}")
                except Exception:
                    info_parts.append("max|r|=n/a")
            info = f"({', '.join(info_parts)})"

            reason_str = ""
            if "singular" in reasons:
                reason_str = " (singular)"
            elif "high correlation" in reasons:
                reason_str = " (high correlation)"

            self.lbl_stability_light.setText(f"Stability: {icon} {label}{reason_str} {info}")
        else:
            self.lbl_stability_light.setText("Stability: -")

        results_text = result.get("results_text") or ""
        if results_text:
            self._append_log_message("\n" + str(results_text).rstrip() + "\n", immediate=True)

        stats = result.get("statistics") or {}
        rms = stats.get("RMS")
        if rms is not None:
            try:
                self._append_log_message(f"Finalizado. RMS={float(rms):.6g}", immediate=True)
            except Exception:
                self._append_log_message("Finalizado.", immediate=True)
        else:
            self._append_log_message("Finalizado.", immediate=True)

        if self._log_buffered_count > 0:
            self._append_log_message(
                f"[UI] Log buffering: buffered={self._log_buffered_count}, flushes={self._log_flush_count}",
                immediate=True,
            )

        graphs_ready = self._result_has_rendered_graphs(result)
        self.btn_render_graphs.setEnabled((not self._is_running) and (not graphs_ready))
        if not graphs_ready:
            self._append_log_message(
                "Graficos diferidos disponibles: usa 'Render graphs' para generarlos sin reoptimizar.",
                immediate=True,
            )

        self._update_errors_context_from_result(result)

    @Slot(str)
    def _on_fit_error(self, message: str) -> None:
        self._log_flush_timer.stop()
        self._flush_log_buffer()
        self._append_log_message(f"ERROR: {message}", immediate=True)
        QMessageBox.critical(self, "Fit error", str(message))

    def _on_fit_finished(self) -> None:
        self._log_flush_timer.stop()
        self._flush_log_buffer()
        self._set_running(False)

    def _on_fit_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

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
        self.chk_efa.setChecked(bool(config.get("efa_enabled", False)))
        self._apply_channels_mode_ui()

        # Restore preprocessing / weighting / epsilon solver
        baseline_mode = str(config.get("baseline_mode") or "off").strip().lower()
        ix_baseline = self.combo_baseline_mode.findData(baseline_mode)
        if ix_baseline >= 0:
            self.combo_baseline_mode.setCurrentIndex(ix_baseline)
        self.edit_baseline_start.setText(str(config.get("baseline_start", 450.0)))
        self.edit_baseline_end.setText(str(config.get("baseline_end", 600.0)))
        self.spin_baseline_auto_q.setValue(self._parse_float(config.get("baseline_auto_quantile", 0.20), 0.20))

        weighting_mode = str(config.get("weighting_mode") or "none").strip().lower()
        ix_weight = self.combo_weighting_mode.findData(weighting_mode)
        if ix_weight >= 0:
            self.combo_weighting_mode.setCurrentIndex(ix_weight)
        self.edit_weighting_power.setText(str(self._parse_float(config.get("weighting_power", 1.0), 1.0)))
        self.chk_weighting_normalize.setChecked(bool(config.get("weighting_normalize", True)))

        eps_solver_mode = str(config.get("eps_solver_mode") or "soft_penalty").strip().lower()
        ix_solver = self.combo_eps_solver_mode.findData(eps_solver_mode)
        if ix_solver >= 0:
            self.combo_eps_solver_mode.setCurrentIndex(ix_solver)
        self.edit_eps_mu.setText(str(self._parse_float(config.get("mu", 1e-2), 1e-2)))
        self.edit_delta_rel.setText(str(self._parse_float(config.get("delta_rel", 0.01), 0.01)))
        self.edit_alpha_smooth.setText(str(self._parse_float(config.get("alpha_smooth", 0.0), 0.0)))
        self._update_preprocess_controls()

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
        self.model_opt_plots.set_multi_start(
            config.get("multi_start_runs", 1),
            config.get("multi_start_seeds"),
        )
        self._render_graphs_enabled = bool(config.get("render_graphs", True))
        rq = str(config.get("render_quality") or "preview").strip().lower()
        if rq not in {"preview", "full", "draft"}:
            rq = "preview"
        self._render_quality = rq
        self._multi_start_parallel = bool(config.get("multi_start_parallel", False))
        workers_val = config.get("multi_start_max_workers")
        try:
            workers_int = int(workers_val) if workers_val is not None else None
        except (TypeError, ValueError):
            workers_int = None
        self._multi_start_max_workers = workers_int if (workers_int is not None and workers_int > 0) else None
        equation_text = str(config.get("equation_text") or "").strip()
        if self.equation_editor is not None:
            if equation_text:
                self.equation_editor.set_text(equation_text)
            else:
                self.equation_editor.clear()

        if "show_stability_diagnostics" in config:
            self.chk_show_diag.setChecked(bool(config["show_stability_diagnostics"]))

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
        self.combo_baseline_mode.setCurrentIndex(self.combo_baseline_mode.findData("off"))
        self.edit_baseline_start.setText("450.0")
        self.edit_baseline_end.setText("600.0")
        self.spin_baseline_auto_q.setValue(0.20)
        self.combo_weighting_mode.setCurrentIndex(self.combo_weighting_mode.findData("none"))
        self.edit_weighting_power.setText("1.0")
        self.chk_weighting_normalize.setChecked(True)
        self.combo_eps_solver_mode.setCurrentIndex(self.combo_eps_solver_mode.findData("soft_penalty"))
        self.edit_eps_mu.setText("0.01")
        self.edit_delta_rel.setText("0.01")
        self.edit_alpha_smooth.setText("0.0")
        self._apply_channels_mode_ui()
        self._update_preprocess_controls()

        self._clear_conc_columns()
        self.model_opt_plots.reset()

        self.canvas_main.clear()
        self._reset_plot_state()
        self.log.clear()
        self.chk_show_diag.setChecked(False)
        self.lbl_stability_light.setText("Stability: -")
        self._last_result = None
        self._last_config = None
        self._last_fit_context = None
        self._graph_solver_inputs = None
        self._render_graphs_enabled = True
        self._render_quality = "preview"
        self._multi_start_parallel = False
        self._multi_start_max_workers = None
        self.btn_save.setEnabled(False)
        self.btn_render_graphs.setEnabled(False)

    def _update_errors_context_from_result(self, result: dict[str, Any]) -> None:
        if not result.get("success", True):
            self.model_opt_plots.set_errors_context(None)
            self._last_fit_context = None
            return
        export_data = result.get("export_data") or {}
        k_hat = export_data.get("k") or [c.get("log10K") for c in result.get("constants") or []]
        if not k_hat:
            self.model_opt_plots.set_errors_context(None)
            self._last_fit_context = None
            return

        context = {
            "technique": "spectro",
            "k_hat": k_hat,
            "C_T": export_data.get("C_T") or [],
            "Y": export_data.get("Y") or [],
            "weights": export_data.get("weights") or [],
            "y_fit_hat": export_data.get("yfit") or [],
            "modelo_solver": export_data.get("modelo") or [],
            "non_abs_species": export_data.get("non_abs_species") or [],
            "abs_group_map": export_data.get("abs_group_map") or [],
            "algorithm": (self._last_config or {}).get("algorithm", "Newton-Raphson"),
            "model_settings": (self._last_config or {}).get("model_settings", "Free"),
            "optimizer": (self._last_config or {}).get("optimizer", "powell"),
            "bounds": (self._last_config or {}).get("bounds", []),
            "fixed_mask": (self._last_config or {}).get("fixed_mask", []),
            "eps_solver_mode": (self._last_config or {}).get("eps_solver_mode", "soft_penalty"),
            "mu": (self._last_config or {}).get("mu", 1e-2),
            "delta_mode": (self._last_config or {}).get("delta_mode", "off"),
            "delta_rel": (self._last_config or {}).get("delta_rel", 0.01),
            "alpha_smooth": (self._last_config or {}).get("alpha_smooth", 0.0),
            "param_names": [f"K{i+1}" for i in range(len(k_hat))],
            "refit_from_data": self._refit_from_data,
        }
        self._last_fit_context = dict(context)
        self.model_opt_plots.set_errors_context(context, auto_compute=True)

    def _refit_from_data(
        self,
        data_star,
        theta0,
        max_iter: int = 30,
        tol: float = 1e-8,
    ):
        try:
            ctx = self._last_fit_context or {}
            if not ctx:
                return np.asarray(theta0, dtype=float), False, {"error": "Missing fit context."}

            from scipy import optimize
            from scipy.optimize import differential_evolution, dual_annealing, basinhopping

            from hmfit_core.processors.spectroscopy_processor import (
                _build_bounds_list,
                _build_smoothness_laplacian,
                _solve_spectral_model,
            )
            from hmfit_core.solvers import NewtonRaphson, LevenbergMarquardt

            Y_star = np.asarray(data_star, dtype=float)
            c_t_raw = ctx.get("C_T")
            C_T = np.asarray(c_t_raw if c_t_raw is not None else [], dtype=float)
            modelo_raw = ctx.get("modelo_solver")
            modelo = np.asarray(modelo_raw if modelo_raw is not None else [], dtype=float)
            nas_raw = ctx.get("non_abs_species")
            nas = list(nas_raw) if nas_raw is not None else []
            group_map_raw = ctx.get("abs_group_map")
            group_map = None
            if group_map_raw is not None:
                group_map_arr = np.asarray(group_map_raw, dtype=float)
                if group_map_arr.size > 0:
                    group_map = group_map_arr
            algorithm = str(ctx.get("algorithm") or "Newton-Raphson")
            model_settings = str(ctx.get("model_settings") or "Free")
            optimizer = str(ctx.get("optimizer") or "powell")
            eps_solver_mode = str(ctx.get("eps_solver_mode") or "soft_penalty").strip().lower()
            eps_mu = self._parse_float(ctx.get("mu", 1e-2), 1e-2)
            delta_mode = str(ctx.get("delta_mode") or "off").strip().lower()
            delta_rel = self._parse_float(ctx.get("delta_rel", 0.01), 0.01)
            alpha_smooth = self._parse_float(ctx.get("alpha_smooth", 0.0), 0.0)
            bounds_raw = ctx.get("bounds")
            bounds_raw = list(bounds_raw) if bounds_raw is not None else []
            weights_raw = ctx.get("weights")
            if weights_raw is None:
                weights = np.ones(Y_star.shape[0], dtype=float)
            else:
                weights = np.asarray(weights_raw, dtype=float).ravel()
                if weights.size != Y_star.shape[0]:
                    weights = np.ones(Y_star.shape[0], dtype=float)
            weights = np.nan_to_num(np.abs(weights), nan=1.0, posinf=1.0, neginf=1.0)
            weights_row = weights.reshape(1, -1)
            smooth_matrix = None
            if alpha_smooth > 0.0 and Y_star.shape[0] >= 3:
                smooth_matrix = _build_smoothness_laplacian(int(Y_star.shape[0]))

            if algorithm == "Newton-Raphson":
                res = NewtonRaphson(C_T, modelo, nas, model_settings)
            elif algorithm == "Levenberg-Marquardt":
                res = LevenbergMarquardt(C_T, modelo, nas, model_settings)
            else:
                return np.asarray(theta0, dtype=float), False, {"error": f"Unknown algorithm: {algorithm}"}

            theta0 = np.asarray(theta0, dtype=float).ravel()
            p0_full = theta0.copy()
            fixed_mask_raw = ctx.get("fixed_mask")
            fixed_mask = (
                np.asarray(fixed_mask_raw, dtype=bool)
                if fixed_mask_raw is not None
                else np.zeros(p0_full.size, dtype=bool)
            )
            bnds = _build_bounds_list(bounds_raw)
            if len(bnds) < p0_full.size:
                bnds.extend([(-np.inf, np.inf)] * (p0_full.size - len(bnds)))
            if len(bnds) > p0_full.size:
                bnds = bnds[: p0_full.size]
            for i, (lb, ub) in enumerate(bnds):
                if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
                    fixed_mask[i] = True
            free_idx = np.where(~fixed_mask)[0]

            def pack(theta_free: np.ndarray) -> np.ndarray:
                k_full = p0_full.copy()
                if free_idx.size:
                    k_full[free_idx] = np.asarray(theta_free, dtype=float).ravel()
                return k_full

            def f_m(theta_free: np.ndarray) -> float:
                try:
                    k_curr_full = pack(theta_free)
                    C = res.concentraciones(k_curr_full)[0]
                    fit = _solve_spectral_model(
                        C,
                        Y_star.T,
                        group_map=group_map,
                        eps_solver_mode=eps_solver_mode,
                        mu=float(eps_mu),
                        delta_mode=delta_mode,
                        delta_rel=float(delta_rel),
                        alpha_smooth=float(alpha_smooth),
                        smooth_matrix=smooth_matrix,
                        max_iters=300,
                    )
                    r = fit["yfit"] - np.asarray(Y_star.T, dtype=float)
                    r_w = r * weights_row
                    rms = float(np.sqrt(np.mean(np.square(r_w))))
                    if np.isnan(rms) or np.isinf(rms):
                        return 1e50
                    return rms
                except Exception:
                    return 1e50

            n_iter = 0
            success = True
            if free_idx.size == 0:
                theta_star = p0_full.copy()
            else:
                k_free0 = p0_full[free_idx]
                bounds_free = [bnds[i] for i in free_idx]
                def _bounds_finite(bounds_list: list[tuple[float | None, float | None]]) -> bool:
                    for lb, ub in bounds_list:
                        if lb is None or ub is None:
                            return False
                        if not np.isfinite(lb) or not np.isfinite(ub):
                            return False
                    return True

                def _run_optimizer(method_name: str):
                    options = {"maxiter": int(max_iter)}
                    if tol is not None:
                        options.update(
                            {
                                "xtol": float(tol),
                                "ftol": float(tol),
                                "gtol": float(tol),
                                "xatol": float(tol),
                                "fatol": float(tol),
                            }
                        )
                    if method_name == "differential_evolution":
                        return differential_evolution(
                            f_m,
                            bounds_free,
                            x0=k_free0,
                            strategy="best1bin",
                            maxiter=int(max_iter),
                            popsize=15,
                            tol=float(tol),
                            mutation=(0.5, 1),
                            recombination=0.7,
                            init="latinhypercube",
                        )
                    if method_name == "dual_annealing":
                        if not _bounds_finite(bounds_free):
                            raise ValueError("Dual annealing requires all bounds to be finite.")
                        return dual_annealing(f_m, bounds_free, maxiter=int(max_iter))
                    if method_name == "basinhopping":
                        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds_free, "options": options}
                        return basinhopping(
                            f_m, k_free0, niter=int(max_iter), minimizer_kwargs=minimizer_kwargs
                        )
                    if method_name == "global_local":
                        if not _bounds_finite(bounds_free):
                            raise ValueError("Global-local optimization requires all bounds to be finite.")
                        global_res = differential_evolution(
                            f_m,
                            bounds_free,
                            x0=k_free0,
                            strategy="best1bin",
                            maxiter=int(max_iter),
                            popsize=15,
                            tol=float(tol),
                            mutation=(0.5, 1),
                            recombination=0.7,
                            init="latinhypercube",
                        )
                        return optimize.minimize(
                            f_m, global_res.x, method="L-BFGS-B", bounds=bounds_free, options=options
                        )
                    return optimize.minimize(
                        f_m, k_free0, method=method_name, bounds=bounds_free, options=options
                    )

                def _valid_opt_res(res) -> bool:
                    if res is None:
                        return False
                    x = getattr(res, "x", None)
                    if x is None:
                        return False
                    x = np.asarray(x, dtype=float).ravel()
                    if x.size != free_idx.size:
                        return False
                    return bool(np.all(np.isfinite(x)))

                opt_res = None
                opt_err = None
                opt_method = optimizer
                try:
                    opt_res = _run_optimizer(optimizer)
                except Exception as exc:
                    opt_err = str(exc)

                if not _valid_opt_res(opt_res):
                    fallback_method = "nelder-mead" if optimizer != "nelder-mead" else None
                    if fallback_method is not None:
                        try:
                            opt_res = _run_optimizer(fallback_method)
                            opt_method = fallback_method
                            opt_err = None
                        except Exception as exc:
                            opt_err = str(exc)

                if not _valid_opt_res(opt_res):
                    return np.asarray(theta0, dtype=float), False, {
                        "error": opt_err or "Optimization failed.",
                        "optimizer": optimizer,
                    }

                theta_star = pack(opt_res.x)
                success = bool(getattr(opt_res, "success", True))
                n_iter = int(getattr(opt_res, "nit", 0) or 0)

            final_rms = f_m(theta_star[free_idx] if free_idx.size else np.array([]))
            theta_star = np.asarray(theta_star, dtype=float).ravel()
            ok = bool(np.isfinite(final_rms) and np.all(np.isfinite(theta_star)))
            info = {
                "n_iter": n_iter,
                "final_rms": final_rms,
                "success": success,
                "optimizer_requested": optimizer,
                "optimizer_used": opt_method if free_idx.size else None,
                "fallback_used": bool(free_idx.size and opt_method != optimizer),
            }
            return theta_star, ok, info
        except Exception as exc:
            return np.asarray(theta0, dtype=float), False, {"error": str(exc)}

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
