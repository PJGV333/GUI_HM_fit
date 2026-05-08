from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hmfit_core.api import run_acid_base_fit
from hmfit_core.acid_base import log_beta_to_pka, pka_to_log_beta
from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
from hmfit_core.exports import write_results_xlsx
from hmfit_gui_qt.widgets.error_analysis_widget import ErrorAnalysisWidget
from hmfit_gui_qt.widgets.log_console import LogConsole
from hmfit_gui_qt.widgets.mpl_canvas import MplCanvas, NavigationToolbar
from hmfit_gui_qt.workers.fit_worker import FitWorker


def _optional_float(text: str) -> float | None:
    value = str(text or "").strip()
    if not value:
        return None
    return float(value)


def _parse_float_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return [float(v) for v in raw.reshape(-1)]
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    text = str(raw).replace(";", ",").strip()
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _table_text(table: QTableWidget, row: int, col: int, default: str = "") -> str:
    item = table.item(row, col)
    if item is None:
        return default
    text = str(item.text() or "").strip()
    return text if text else default


def _set_table_text(table: QTableWidget, row: int, col: int, value: Any) -> None:
    table.setItem(row, col, QTableWidgetItem(str(value)))


class AcidBaseTab(QWidget):
    COMPONENT_COLUMNS = [
        "Component",
        "Total concentration",
        "Base charge",
        "Number of protonation steps",
        "pKa initial values",
        "Use pKa or log_beta",
        "Fixed concentration",
        "Role",
    ]
    SPECIES_COLUMNS = [
        "Species name",
        "Parent component",
        "h_count",
        "Charge",
        "log_beta",
        "Stepwise pKa",
        "Include",
        "Fixed",
    ]
    PARAM_COLUMNS = ["Parameter", "Initial value", "Min", "Max", "Fixed", "Description"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FitWorker | None = None
        self._thread: QThread | None = None
        self._file_path = ""
        self._last_result: dict[str, Any] | None = None
        self._last_config: dict[str, Any] | None = None
        self._plot_pages: list[tuple[str, str]] = []
        self._plot_index = 0
        self._is_running = False
        self._updating_tables = False
        self.errors_panel: ErrorAnalysisWidget | None = None

        self._build_ui()
        self._add_default_component()
        self._refresh_species_table()
        self._refresh_parameter_table()
        self._on_model_type_changed()
        self._on_titrant_type_changed()
        self._set_running(False)
        self.canvas_main.show_message("Import a CSV or Excel file to begin")
        self.log.append_text("Ready. Import potentiometry, spectroscopy, or NMR CSV/TXT/XLSX data.")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._main_split = QSplitter(Qt.Orientation.Horizontal, self)
        outer.addWidget(self._main_split, 1)

        left_scroll = QScrollArea(self._main_split)
        left_scroll.setWidgetResizable(True)
        left_container = QWidget(left_scroll)
        left_scroll.setWidget(left_container)
        left_layout = QVBoxLayout(left_container)

        self._build_data_input_group(left_container, left_layout)

        self.model_opt_tabs = QTabWidget(left_container)
        self._build_model_tab()
        self._build_optimization_tab()
        self._build_plots_tab()
        self._build_errors_tab()
        left_layout.addWidget(self.model_opt_tabs, 1)

        self._build_actions_row(left_container, left_layout)
        self._build_right_panel()

        self._main_split.setStretchFactor(0, 0)
        self._main_split.setStretchFactor(1, 1)

    def _build_data_input_group(self, parent: QWidget, layout: QVBoxLayout) -> None:
        self._data_group = QGroupBox("DATA / INPUT", parent)
        data_layout = QVBoxLayout(self._data_group)
        form = QFormLayout()

        self.combo_data_type = QComboBox(self._data_group)
        self.combo_data_type.addItem("Potentiometry pH/EMF", "potentiometry")
        self.combo_data_type.addItem("Spectroscopy signal vs pH", "spectroscopy")
        self.combo_data_type.addItem("1H NMR shifts vs pH", "nmr")
        self.combo_data_type.currentIndexChanged.connect(self._on_data_type_changed)
        form.addRow("Data type", self.combo_data_type)

        self.combo_sheet = QComboBox(self._data_group)
        self.combo_sheet.setEnabled(False)
        self.combo_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        form.addRow("Excel sheet", self.combo_sheet)
        data_layout.addLayout(form)

        file_row = QHBoxLayout()
        self.btn_choose_file = QPushButton("Choose file...", self._data_group)
        self.btn_choose_file.clicked.connect(self._on_choose_file_clicked)
        file_row.addWidget(self.btn_choose_file)
        self.lbl_file_status = QLabel("No file selected", self._data_group)
        self.lbl_file_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_row.addWidget(self.lbl_file_status, 1)
        data_layout.addLayout(file_row)

        self.preview_text = QPlainTextEdit(self._data_group)
        self.preview_text.setReadOnly(True)
        self.preview_text.setMinimumHeight(130)
        data_layout.addWidget(self.preview_text, 1)
        layout.addWidget(self._data_group)

    def _build_model_tab(self) -> None:
        tab = QWidget(self.model_opt_tabs)
        layout = QVBoxLayout(tab)

        model_form = QFormLayout()
        self.combo_model_type = QComboBox(tab)
        self.combo_model_type.addItem("Simple monoprotic acid/base", "simple_monoprotic")
        self.combo_model_type.addItem("Polyprotic ligand", "polyprotic")
        self.combo_model_type.addItem("Multiple acid-base components", "multiple_components")
        self.combo_model_type.addItem("Custom species table", "custom_species")
        self.combo_model_type.addItem(
            "Coupled acid-base / host-guest model (future)",
            "coupled_future",
        )
        future_idx = self.combo_model_type.findData("coupled_future")
        future_item = self.combo_model_type.model().item(future_idx)
        if future_item is not None:
            future_item.setEnabled(False)
        self.combo_model_type.currentIndexChanged.connect(self._on_model_type_changed)
        model_form.addRow("Model type", self.combo_model_type)
        layout.addLayout(model_form)

        comp_group = QGroupBox("Components", tab)
        comp_layout = QVBoxLayout(comp_group)
        self.components_table = QTableWidget(0, len(self.COMPONENT_COLUMNS), comp_group)
        self.components_table.setHorizontalHeaderLabels(self.COMPONENT_COLUMNS)
        self.components_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.components_table.horizontalHeader().setStretchLastSection(True)
        self.components_table.setMinimumHeight(150)
        self.components_table.itemChanged.connect(self._on_component_table_changed)
        comp_layout.addWidget(self.components_table)
        comp_buttons = QHBoxLayout()
        self.btn_add_component = QPushButton("Add component", comp_group)
        self.btn_add_component.clicked.connect(self._add_default_component)
        comp_buttons.addWidget(self.btn_add_component)
        self.btn_remove_component = QPushButton("Remove selected", comp_group)
        self.btn_remove_component.clicked.connect(self._remove_selected_component)
        comp_buttons.addWidget(self.btn_remove_component)
        self.btn_generate_species = QPushButton("Generate species", comp_group)
        self.btn_generate_species.clicked.connect(self._refresh_species_table)
        comp_buttons.addWidget(self.btn_generate_species)
        comp_buttons.addStretch(1)
        comp_layout.addLayout(comp_buttons)
        layout.addWidget(comp_group)

        species_group = QGroupBox("Species", tab)
        species_layout = QVBoxLayout(species_group)
        self.species_table = QTableWidget(0, len(self.SPECIES_COLUMNS), species_group)
        self.species_table.setHorizontalHeaderLabels(self.SPECIES_COLUMNS)
        self.species_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.species_table.horizontalHeader().setStretchLastSection(True)
        self.species_table.setMinimumHeight(180)
        self.species_table.itemChanged.connect(self._on_species_table_changed)
        species_layout.addWidget(self.species_table)
        species_buttons = QHBoxLayout()
        self.btn_add_species = QPushButton("Add species", species_group)
        self.btn_add_species.clicked.connect(self._add_empty_species_row)
        species_buttons.addWidget(self.btn_add_species)
        self.btn_remove_species = QPushButton("Remove selected", species_group)
        self.btn_remove_species.clicked.connect(self._remove_selected_species)
        species_buttons.addWidget(self.btn_remove_species)
        species_buttons.addStretch(1)
        species_layout.addLayout(species_buttons)
        layout.addWidget(species_group)

        self.titration_group = QGroupBox("Potentiometric titration model", tab)
        titration_layout = QVBoxLayout(self.titration_group)
        titration_form = QFormLayout()
        self.spin_initial_volume = QDoubleSpinBox(self.titration_group)
        self.spin_initial_volume.setDecimals(6)
        self.spin_initial_volume.setRange(1.0e-12, 1.0e9)
        self.spin_initial_volume.setValue(10.0)
        self.spin_titrant_conc = QDoubleSpinBox(self.titration_group)
        self.spin_titrant_conc.setDecimals(8)
        self.spin_titrant_conc.setRange(0.0, 1.0e6)
        self.spin_titrant_conc.setValue(1.0e-3)
        self.spin_titrant_conc.setSingleStep(1.0e-4)
        self.combo_titrant_type = QComboBox(self.titration_group)
        self.combo_titrant_type.addItem("strong base", "base")
        self.combo_titrant_type.addItem("strong acid", "acid")
        self.combo_titrant_type.addItem("custom titrant", "custom")
        self.combo_titrant_type.currentIndexChanged.connect(self._on_titrant_type_changed)
        self.combo_strong_ion = QComboBox(self.titration_group)
        self.combo_strong_ion.addItem("automatic", "automatic")
        self.combo_strong_ion.addItem("manual", "manual")
        self.spin_initial_strong_charge = QDoubleSpinBox(self.titration_group)
        self.spin_initial_strong_charge.setDecimals(8)
        self.spin_initial_strong_charge.setRange(-1.0e6, 1.0e6)
        self.spin_initial_strong_charge.setValue(0.0)
        self.spin_titrant_strong_charge = QDoubleSpinBox(self.titration_group)
        self.spin_titrant_strong_charge.setDecimals(8)
        self.spin_titrant_strong_charge.setRange(-1.0e6, 1.0e6)
        self.spin_titrant_strong_charge.setValue(0.0)
        self.spin_volume_offset = QDoubleSpinBox(self.titration_group)
        self.spin_volume_offset.setDecimals(6)
        self.spin_volume_offset.setRange(-1.0e9, 1.0e9)
        self.spin_volume_offset.setValue(0.0)
        self.spin_ph_min = QDoubleSpinBox(self.titration_group)
        self.spin_ph_min.setDecimals(3)
        self.spin_ph_min.setRange(-10.0, 30.0)
        self.spin_ph_min.setValue(-2.0)
        self.spin_ph_max = QDoubleSpinBox(self.titration_group)
        self.spin_ph_max.setDecimals(3)
        self.spin_ph_max.setRange(-10.0, 30.0)
        self.spin_ph_max.setValue(16.0)

        titration_form.addRow("Initial volume", self.spin_initial_volume)
        titration_form.addRow("Titrant concentration", self.spin_titrant_conc)
        titration_form.addRow("Titrant type", self.combo_titrant_type)
        titration_form.addRow("Strong ion contribution", self.combo_strong_ion)
        titration_form.addRow("Initial strong ion charge", self.spin_initial_strong_charge)
        titration_form.addRow("Titrant strong ion charge", self.spin_titrant_strong_charge)
        titration_form.addRow("Volume offset / blank correction", self.spin_volume_offset)
        ph_bounds_row = QHBoxLayout()
        ph_bounds_row.addWidget(self.spin_ph_min)
        ph_bounds_row.addWidget(QLabel("to", self.titration_group))
        ph_bounds_row.addWidget(self.spin_ph_max)
        titration_form.addRow("pH bounds", ph_bounds_row)
        titration_layout.addLayout(titration_form)

        self.custom_titrant_table = QTableWidget(0, 3, self.titration_group)
        self.custom_titrant_table.setHorizontalHeaderLabels(
            ["species/component", "concentration in titrant", "charge contribution"]
        )
        self.custom_titrant_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.custom_titrant_table.horizontalHeader().setStretchLastSection(True)
        self.custom_titrant_table.setMinimumHeight(90)
        titration_layout.addWidget(self.custom_titrant_table)
        custom_buttons = QHBoxLayout()
        self.btn_add_titrant_row = QPushButton("Add custom titrant row", self.titration_group)
        self.btn_add_titrant_row.clicked.connect(self._add_custom_titrant_row)
        custom_buttons.addWidget(self.btn_add_titrant_row)
        self.btn_remove_titrant_row = QPushButton("Remove selected", self.titration_group)
        self.btn_remove_titrant_row.clicked.connect(self._remove_selected_titrant_row)
        custom_buttons.addWidget(self.btn_remove_titrant_row)
        custom_buttons.addStretch(1)
        titration_layout.addLayout(custom_buttons)
        layout.addWidget(self.titration_group)

        self.model_opt_tabs.addTab(tab, "Model")

    def _build_optimization_tab(self) -> None:
        tab = QWidget(self.model_opt_tabs)
        layout = QVBoxLayout(tab)

        self.parameters_table = QTableWidget(0, len(self.PARAM_COLUMNS), tab)
        self.parameters_table.setHorizontalHeaderLabels(self.PARAM_COLUMNS)
        self.parameters_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.parameters_table.horizontalHeader().setStretchLastSection(True)
        self.parameters_table.setMinimumHeight(220)
        self.parameters_table.setToolTip(
            "pKw is accepted for all acid-base datasets, but it only affects "
            "potentiometric electroneutrality calculations in v1. Spectroscopy "
            "and NMR fits use the measured pH directly."
        )
        layout.addWidget(self.parameters_table, 1)

        pkw_group = QGroupBox("Water autoionization", tab)
        pkw_layout = QFormLayout(pkw_group)
        self.spin_pkw = QDoubleSpinBox(pkw_group)
        self.spin_pkw.setDecimals(4)
        self.spin_pkw.setRange(0.0, 30.0)
        self.spin_pkw.setValue(14.0)
        self.spin_pkw.setToolTip(
            "Kw = 10^(-pKw). This value affects potentiometric electroneutrality "
            "calculations only; imposed-pH spectroscopy and NMR fits do not use "
            "Kw to calculate fractions in v1."
        )
        self.spin_pkw.valueChanged.connect(self._on_pkw_spin_changed)
        pkw_layout.addRow("pKw", self.spin_pkw)
        layout.addWidget(pkw_group)

        opt_buttons = QHBoxLayout()
        self.btn_refresh_params = QPushButton("Refresh parameter table", tab)
        self.btn_refresh_params.clicked.connect(self._refresh_parameter_table)
        opt_buttons.addWidget(self.btn_refresh_params)
        opt_buttons.addStretch(1)
        layout.addLayout(opt_buttons)

        electrode_group = QGroupBox("Electrode / spectroscopy options", tab)
        electrode_layout = QFormLayout(electrode_group)
        self.edit_e0 = QLineEdit("", electrode_group)
        self.edit_e0.setPlaceholderText("fixed E0, mV")
        self.edit_slope = QLineEdit("-59.16", electrode_group)
        self.edit_slope.setPlaceholderText("fixed slope, mV/pH")
        self.chk_fit_electrode = QCheckBox("Fit E0 and slope for EMF data", electrode_group)
        self.chk_baseline = QCheckBox("Include linear baseline in spectroscopy", electrode_group)
        electrode_layout.addRow("Electrode E0", self.edit_e0)
        electrode_layout.addRow("Electrode slope", self.edit_slope)
        electrode_layout.addRow("", self.chk_fit_electrode)
        electrode_layout.addRow("", self.chk_baseline)
        layout.addWidget(electrode_group)

        weight_group = QGroupBox("Residual weights", tab)
        weight_layout = QFormLayout(weight_group)
        self.spin_sigma_ph = QDoubleSpinBox(weight_group)
        self.spin_sigma_ph.setDecimals(6)
        self.spin_sigma_ph.setRange(1.0e-12, 1.0e6)
        self.spin_sigma_ph.setValue(1.0)
        self.spin_sigma_emf = QDoubleSpinBox(weight_group)
        self.spin_sigma_emf.setDecimals(6)
        self.spin_sigma_emf.setRange(1.0e-12, 1.0e6)
        self.spin_sigma_emf.setValue(1.0)
        weight_layout.addRow("sigma pH", self.spin_sigma_ph)
        weight_layout.addRow("sigma EMF", self.spin_sigma_emf)
        layout.addWidget(weight_group)

        self.results_text = QPlainTextEdit(tab)
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(140)
        layout.addWidget(self.results_text)

        self.model_opt_tabs.addTab(tab, "Optimization")

    def _build_plots_tab(self) -> None:
        tab = QWidget(self.model_opt_tabs)
        layout = QVBoxLayout(tab)
        self.plot_options = QListWidget(tab)
        plot_rows = [
            ("fit", "observed/calculated fit"),
            ("residuals", "residuals"),
            ("species_pH", "species vs pH"),
            ("species_volume", "species vs volume"),
            ("concentration_pH", "concentration vs pH"),
            ("concentration_volume", "concentration vs volume"),
            ("spectroscopy_pure", "spectroscopy pure signals if available"),
            ("nmr_limiting", "NMR limiting shifts if available"),
        ]
        for plot_id, label in plot_rows:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, plot_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.plot_options.addItem(item)
        layout.addWidget(self.plot_options, 1)
        self.model_opt_tabs.addTab(tab, "Plots")

    def _build_errors_tab(self) -> None:
        tab = QWidget(self.model_opt_tabs)
        layout = QVBoxLayout(tab)
        self.errors_panel = ErrorAnalysisWidget(self._compute_errors_payload, tab)
        self.errors_panel.set_supported_methods(
            {"analytic", "bootstrap_linear", "bootstrap_full_refit_audit"},
            default="analytic",
        )
        self.errors_panel.combo_error_method.setToolTip(
            "Acid-base errors support analytical covariance, linearized wild bootstrap, "
            "and full-refit bootstrap. One-step LM bootstrap is not available here yet."
        )
        self.errors_panel.output_ready.connect(self._on_errors_output_ready)
        layout.addWidget(self.errors_panel, 1)
        self.model_opt_tabs.addTab(tab, "Errors")

    def _build_actions_row(self, parent: QWidget, layout: QVBoxLayout) -> None:
        actions_group = QGroupBox("", parent)
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_import = QPushButton("Import Config", actions_group)
        self.btn_import.clicked.connect(self._on_import_config_clicked)
        actions_layout.addWidget(self.btn_import)

        self.btn_export = QPushButton("Export Config", actions_group)
        self.btn_export.clicked.connect(self._on_export_config_clicked)
        actions_layout.addWidget(self.btn_export)

        self.btn_reset = QPushButton("Reset Calculation", actions_group)
        self.btn_reset.clicked.connect(self.reset_tab)
        actions_layout.addWidget(self.btn_reset)
        actions_layout.addStretch(1)

        self.btn_process = QPushButton("Process Data", actions_group)
        self.btn_process.clicked.connect(self._on_process_clicked)
        actions_layout.addWidget(self.btn_process)

        self.btn_cancel = QPushButton("Cancel", actions_group)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        actions_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save results", actions_group)
        self.btn_save.clicked.connect(self._on_save_results_clicked)
        actions_layout.addWidget(self.btn_save)

        self.btn_render_graphs = QPushButton("Render graphs", actions_group)
        self.btn_render_graphs.clicked.connect(self._on_render_graphs_clicked)
        actions_layout.addWidget(self.btn_render_graphs)

        layout.addWidget(actions_group)

    def _build_right_panel(self) -> None:
        right_split = QSplitter(Qt.Orientation.Vertical, self._main_split)
        plot_panel = QWidget(right_split)
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        nav = QHBoxLayout()
        self.btn_plot_prev = QPushButton("Prev", plot_panel)
        self.btn_plot_prev.clicked.connect(lambda: self._navigate_plot(-1))
        nav.addWidget(self.btn_plot_prev)
        self.lbl_plot_title = QLabel("Plots", plot_panel)
        self.lbl_plot_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.lbl_plot_title, 1)
        self.btn_plot_next = QPushButton("Next", plot_panel)
        self.btn_plot_next.clicked.connect(lambda: self._navigate_plot(1))
        nav.addWidget(self.btn_plot_next)
        plot_layout.addLayout(nav)

        self.canvas_main = MplCanvas(plot_panel)
        plot_layout.addWidget(self.canvas_main, 1)
        plot_layout.addWidget(NavigationToolbar(self.canvas_main, plot_panel))

        log_panel = QWidget(right_split)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Diagnostics / Log", log_panel))
        self.log = LogConsole(log_panel)
        self.log.setMinimumHeight(160)
        log_layout.addWidget(self.log, 1)

        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

    def _make_combo(self, values: list[str], current: str = "") -> QComboBox:
        combo = QComboBox()
        for value in values:
            combo.addItem(value)
        if current:
            ix = combo.findText(current)
            if ix >= 0:
                combo.setCurrentIndex(ix)
        return combo

    def _make_checkbox(self, checked: bool = False) -> QCheckBox:
        chk = QCheckBox()
        chk.setChecked(bool(checked))
        return chk

    def _add_default_component(self) -> None:
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            row = self.components_table.rowCount()
            self.components_table.insertRow(row)
            name = "L" if row == 0 else f"L{row + 1}"
            defaults = [name, "0.001", "-1", "1", "5.0"]
            for col, value in enumerate(defaults):
                _set_table_text(self.components_table, row, col, value)
            use_combo = self._make_combo(["pKa", "log_beta"], "pKa")
            use_combo.currentIndexChanged.connect(self._on_component_table_changed)
            self.components_table.setCellWidget(row, 5, use_combo)
            fixed = self._make_checkbox(False)
            fixed.stateChanged.connect(self._on_component_table_changed)
            self.components_table.setCellWidget(row, 6, fixed)
            role = self._make_combo(["analyte", "titrant", "background", "spectator", "imposed pH dataset"], "analyte")
            role.currentIndexChanged.connect(self._on_component_table_changed)
            self.components_table.setCellWidget(row, 7, role)
        finally:
            self._updating_tables = was_updating
        self._refresh_species_table()

    def _remove_selected_component(self) -> None:
        rows = sorted({idx.row() for idx in self.components_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.components_table.removeRow(row)
        if not rows:
            return
        self._refresh_species_table()

    def _add_empty_species_row(self) -> None:
        row = self.species_table.rowCount()
        self.species_table.insertRow(row)
        for col in range(6):
            _set_table_text(self.species_table, row, col, "")
        include = self._make_checkbox(True)
        self.species_table.setCellWidget(row, 6, include)
        fixed = self._make_checkbox(False)
        self.species_table.setCellWidget(row, 7, fixed)

    def _remove_selected_species(self) -> None:
        rows = sorted({idx.row() for idx in self.species_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.species_table.removeRow(row)

    def _add_custom_titrant_row(self) -> None:
        row = self.custom_titrant_table.rowCount()
        self.custom_titrant_table.insertRow(row)
        for col, value in enumerate(["", "0.0", "0.0"]):
            _set_table_text(self.custom_titrant_table, row, col, value)

    def _remove_selected_titrant_row(self) -> None:
        rows = sorted({idx.row() for idx in self.custom_titrant_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.custom_titrant_table.removeRow(row)

    def _on_data_type_changed(self) -> None:
        is_pot = str(self.combo_data_type.currentData() or "") == "potentiometry"
        self.titration_group.setVisible(is_pot)
        self._refresh_parameter_table()

    def _on_model_type_changed(self) -> None:
        custom = str(self.combo_model_type.currentData() or "") == "custom_species"
        self.btn_add_species.setEnabled(custom)
        self.btn_remove_species.setEnabled(custom)
        if not custom:
            self._refresh_species_table()
        self._refresh_parameter_table()

    def _on_titrant_type_changed(self) -> None:
        custom = str(self.combo_titrant_type.currentData() or "") == "custom"
        self.custom_titrant_table.setEnabled(custom)
        self.btn_add_titrant_row.setEnabled(custom)
        self.btn_remove_titrant_row.setEnabled(custom)

    def _on_component_table_changed(self) -> None:
        if self._updating_tables:
            return
        try:
            if str(self.combo_model_type.currentData() or "") != "custom_species":
                self._refresh_species_table()
            self._refresh_parameter_table()
        except Exception:
            # Component rows can be temporarily inconsistent while the user edits
            # step counts and comma-separated pKa/log_beta values.
            pass

    def _on_species_table_changed(self) -> None:
        if self._updating_tables:
            return
        try:
            self._refresh_parameter_table()
        except Exception:
            pass

    def _component_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.components_table.rowCount()):
            name = _table_text(self.components_table, row, 0, f"L{row + 1}")
            concentration = float(_table_text(self.components_table, row, 1, "0.0"))
            base_charge = int(float(_table_text(self.components_table, row, 2, "0")))
            n_steps = int(float(_table_text(self.components_table, row, 3, "0")))
            values = _parse_float_list(_table_text(self.components_table, row, 4, ""))
            use_widget = self.components_table.cellWidget(row, 5)
            use_log_beta = bool(isinstance(use_widget, QComboBox) and use_widget.currentText() == "log_beta")
            fixed_widget = self.components_table.cellWidget(row, 6)
            fixed = bool(isinstance(fixed_widget, QCheckBox) and fixed_widget.isChecked())
            role_widget = self.components_table.cellWidget(row, 7)
            role = role_widget.currentText() if isinstance(role_widget, QComboBox) else "analyte"
            if n_steps < 0:
                raise ValueError(f"Component {name!r} has negative protonation steps.")
            if n_steps and len(values) != n_steps:
                raise ValueError(
                    f"Component {name!r} expects {n_steps} pKa/log_beta values; got {len(values)}."
                )
            rows.append(
                {
                    "name": name,
                    "analytical_concentration": concentration,
                    "base_charge": base_charge,
                    "n_steps": n_steps,
                    "values": values,
                    "pka": [] if use_log_beta else values,
                    "log_beta": values if use_log_beta else pka_to_log_beta(values),
                    "use_log_beta": use_log_beta,
                    "fixed_concentration": fixed,
                    "role": str(role),
                }
            )
        return rows

    def _generated_species_rows(self, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
        species: list[dict[str, Any]] = []
        for comp in components:
            name = str(comp["name"])
            base_charge = int(comp["base_charge"])
            log_beta = [float(v) for v in comp.get("log_beta", [])]
            pka = log_beta_to_pka(log_beta) if log_beta else []
            species.append(
                {
                    "name": name,
                    "component": name,
                    "h_count": 0,
                    "charge": base_charge,
                    "log_beta": 0.0,
                    "stepwise_pka": "",
                    "include": True,
                    "fixed": True,
                }
            )
            for h_count, lb in enumerate(log_beta, start=1):
                prefix = "H" if h_count == 1 else f"H{h_count}"
                species.append(
                    {
                        "name": f"{prefix}{name}",
                        "component": name,
                        "h_count": h_count,
                        "charge": base_charge + h_count,
                        "log_beta": float(lb),
                        "stepwise_pka": float(pka[h_count - 1]) if h_count - 1 < len(pka) else "",
                        "include": True,
                        "fixed": False,
                    }
                )
        return species

    def _refresh_species_table(self) -> None:
        if self._updating_tables:
            return
        self._updating_tables = True
        try:
            components = self._component_rows()
            species = self._generated_species_rows(components)
            self.species_table.setRowCount(0)
            for row_data in species:
                row = self.species_table.rowCount()
                self.species_table.insertRow(row)
                values = [
                    row_data["name"],
                    row_data["component"],
                    row_data["h_count"],
                    row_data["charge"],
                    f"{float(row_data['log_beta']):.10g}",
                    row_data["stepwise_pka"],
                ]
                for col, value in enumerate(values):
                    _set_table_text(self.species_table, row, col, value)
                include = self._make_checkbox(bool(row_data.get("include", True)))
                self.species_table.setCellWidget(row, 6, include)
                fixed = self._make_checkbox(bool(row_data.get("fixed", False)))
                self.species_table.setCellWidget(row, 7, fixed)
        except Exception:
            # Leave current table in place while the user is editing partial input.
            pass
        finally:
            self._updating_tables = False

    def _refresh_parameter_table(self) -> None:
        if self._updating_tables:
            return
        self._updating_tables = True
        try:
            components = self._component_rows()
            rows: list[tuple[str, str, str, str, bool, str]] = []
            for comp in components:
                values = comp["log_beta"] if comp["use_log_beta"] else comp["pka"]
                prefix = "log_beta" if comp["use_log_beta"] else "pKa"
                for idx, value in enumerate(values, start=1):
                    rows.append(
                        (
                            f"{prefix}{idx}",
                            f"{float(value):.8g}",
                            "-5" if prefix == "pKa" else "-50",
                            "25" if prefix == "pKa" else "50",
                            False,
                            f"{prefix} for component {comp['name']}",
                        )
                    )
            first = components[0] if components else {}
            rows.extend(
                [
                    (
                        "analyte concentration",
                        f"{float(first.get('analytical_concentration', 0.0)):.8g}",
                        "0",
                        "",
                        True,
                        "Analytical concentration for the primary analyte.",
                    ),
                    (
                        "titrant concentration",
                        f"{float(self.spin_titrant_conc.value()):.8g}" if hasattr(self, "spin_titrant_conc") else "0.001",
                        "0",
                        "",
                        True,
                        "Nominal titrant concentration.",
                    ),
                    (
                        "volume offset",
                        f"{float(self.spin_volume_offset.value()):.8g}" if hasattr(self, "spin_volume_offset") else "0",
                        "",
                        "",
                        True,
                        "Blank correction / effective volume offset.",
                    ),
                    ("electrode E0", _optional_float(self.edit_e0.text()) if hasattr(self, "edit_e0") else "", "", "", True, "Electrode intercept, mV."),
                    ("electrode slope", self.edit_slope.text() if hasattr(self, "edit_slope") else "-59.16", "-120", "120", True, "Electrode slope, mV/pH."),
                    ("pKw", "14.0000", "0", "30", True, "Kw = 10^(-pKw); affects potentiometric electroneutrality only."),
                    ("baseline", "0", "", "", True, "Optional spectroscopy baseline parameter."),
                ]
            )
            old_pkw = self._parameter_value("pKw", default=None)
            self.parameters_table.setRowCount(0)
            for row_data in rows:
                row = self.parameters_table.rowCount()
                self.parameters_table.insertRow(row)
                parameter, initial, min_val, max_val, fixed, description = row_data
                if parameter == "pKw" and old_pkw not in (None, ""):
                    initial = str(old_pkw)
                pkw_tooltip = (
                    "Kw = 10^(-pKw). This value affects potentiometric electroneutrality "
                    "calculations only; imposed-pH spectroscopy and NMR fits do not use "
                    "Kw to calculate fractions in v1."
                )
                for col, value in enumerate([parameter, initial, min_val, max_val]):
                    _set_table_text(self.parameters_table, row, col, value)
                    if parameter == "pKw":
                        item = self.parameters_table.item(row, col)
                        if item is not None:
                            item.setToolTip(pkw_tooltip)
                chk = self._make_checkbox(fixed)
                if parameter == "pKw":
                    chk.setToolTip(pkw_tooltip)
                self.parameters_table.setCellWidget(row, 4, chk)
                _set_table_text(self.parameters_table, row, 5, description)
                if parameter == "pKw":
                    item = self.parameters_table.item(row, 5)
                    if item is not None:
                        item.setToolTip(pkw_tooltip)
                    if hasattr(self, "spin_pkw"):
                        try:
                            pkw_for_spin = float(initial)
                        except (TypeError, ValueError):
                            pkw_for_spin = 14.0
                        pkw_for_spin = min(30.0, max(0.0, pkw_for_spin))
                        self.spin_pkw.blockSignals(True)
                        self.spin_pkw.setValue(pkw_for_spin)
                        self.spin_pkw.blockSignals(False)
        finally:
            self._updating_tables = False

    def _on_pkw_spin_changed(self, value: float) -> None:
        if self._updating_tables:
            return
        self._set_parameter_value("pKw", f"{float(value):.4f}", update_spin=False)

    def _parameter_value(self, name: str, default: Any = "") -> Any:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                value = _table_text(self.parameters_table, row, 1, "")
                return default if value == "" else value
        return default

    def _species_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.species_table.rowCount()):
            include_widget = self.species_table.cellWidget(row, 6)
            include = True if not isinstance(include_widget, QCheckBox) else include_widget.isChecked()
            fixed_widget = self.species_table.cellWidget(row, 7)
            fixed = False if not isinstance(fixed_widget, QCheckBox) else fixed_widget.isChecked()
            rows.append(
                {
                    "name": _table_text(self.species_table, row, 0, ""),
                    "component": _table_text(self.species_table, row, 1, ""),
                    "h_count": int(float(_table_text(self.species_table, row, 2, "0"))),
                    "charge": int(float(_table_text(self.species_table, row, 3, "0"))),
                    "log_beta": float(_table_text(self.species_table, row, 4, "0")),
                    "fixed": fixed,
                    "include": include,
                }
            )
        return [row for row in rows if row["name"] and row["component"] and bool(row.get("include", True))]

    def _acid_base_model_config(self) -> dict[str, Any]:
        components = self._component_rows()
        model_components = []
        for comp in components:
            model_components.append(
                {
                    "name": comp["name"],
                    "analytical_concentration": float(comp["analytical_concentration"]),
                    "base_charge": int(comp["base_charge"]),
                    "n_steps": int(comp["n_steps"]),
                    "pka": [float(v) for v in comp["pka"]],
                    "log_beta": [float(v) for v in comp["log_beta"]],
                    "use_log_beta": bool(comp["use_log_beta"]),
                    "role": str(comp["role"]),
                    "fixed_concentration": bool(comp["fixed_concentration"]),
                }
            )
        return {
            "model_type": str(self.combo_model_type.currentData() or "simple_monoprotic"),
            "components": model_components,
            "species": self._species_rows(),
        }

    def _first_analyte_component(self, model: dict[str, Any]) -> dict[str, Any]:
        components = list(model.get("components") or [])
        for comp in components:
            if str(comp.get("role") or "").lower() == "analyte":
                return comp
        if components:
            return components[0]
        raise ValueError("Define at least one acid-base component.")

    def _selected_plot_ids(self) -> set[str]:
        selected: set[str] = set()
        for row in range(self.plot_options.count()):
            item = self.plot_options.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected.add(str(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def _custom_titrant_config(self) -> tuple[dict[str, float], float]:
        titrant_conc: dict[str, float] = {}
        strong_charge = 0.0
        for row in range(self.custom_titrant_table.rowCount()):
            name = _table_text(self.custom_titrant_table, row, 0, "")
            if not name:
                continue
            conc = float(_table_text(self.custom_titrant_table, row, 1, "0"))
            charge = float(_table_text(self.custom_titrant_table, row, 2, "0"))
            titrant_conc[name] = conc
            strong_charge += conc * charge
        return titrant_conc, float(strong_charge)

    def _on_choose_file_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            "",
            "Data files (*.csv *.txt *.xlsx);;CSV (*.csv *.txt);;Excel (*.xlsx)",
        )
        if not path:
            return
        self._set_file_path(path)

    def _set_file_path(self, file_path: str) -> None:
        path = Path(str(file_path or ""))
        if not path.exists():
            QMessageBox.warning(self, "Missing file", f"Data file not found:\n{path}")
            return
        self._file_path = str(path)
        self.lbl_file_status.setText(self._file_path)
        self.combo_sheet.blockSignals(True)
        self.combo_sheet.clear()
        suffix = path.suffix.lower()
        if suffix == ".xlsx":
            try:
                sheets = pd.ExcelFile(path).sheet_names
            except Exception as exc:
                self.combo_sheet.setEnabled(False)
                self.preview_text.setPlainText(f"Could not read Excel workbook: {exc}")
                self.combo_sheet.blockSignals(False)
                return
            for sheet in sheets:
                self.combo_sheet.addItem(str(sheet), str(sheet))
            self.combo_sheet.setEnabled(bool(sheets))
        else:
            self.combo_sheet.addItem("CSV/TXT", "")
            self.combo_sheet.setEnabled(False)
        self.combo_sheet.blockSignals(False)
        self._preview_selected_sheet()

    def _on_sheet_changed(self) -> None:
        self._preview_selected_sheet()

    def _preview_selected_sheet(self) -> None:
        path = Path(str(self._file_path or ""))
        if not path.exists():
            return
        try:
            if path.suffix.lower() == ".xlsx":
                sheet = self.combo_sheet.currentData()
                if sheet in (None, ""):
                    sheet = 0
                df = pd.read_excel(path, sheet_name=sheet, nrows=8)
            else:
                df = pd.read_csv(path, nrows=8)
            self.preview_text.setPlainText(df.to_string(index=False))
        except Exception as exc:
            self.preview_text.setPlainText(f"Could not preview data file: {exc}")

    def _collect_config(self) -> dict[str, Any]:
        if not self._file_path:
            raise ValueError("No data file selected.")
        model = self._acid_base_model_config()
        analyte = self._first_analyte_component(model)
        pkw = float(self._parameter_value("pKw", default="14.0000"))
        if not 0.0 <= pkw <= 30.0:
            raise ValueError("pKw must be between 0 and 30.")
        kw = 10.0 ** (-pkw)
        first_pka = list(analyte.get("pka") or log_beta_to_pka(list(analyte.get("log_beta") or [])))
        titrant_concentrations, custom_strong_charge = self._custom_titrant_config()
        strong_mode = str(self.combo_strong_ion.currentData() or "automatic")
        titrant_type = str(self.combo_titrant_type.currentData() or "base")

        cfg = {
            "file_path": self._file_path,
            "sheet_name": str(self.combo_sheet.currentData() or ""),
            "data_type": str(self.combo_data_type.currentData() or "potentiometry"),
            "acid_base_model": model,
            # Legacy keys retained for existing code paths/configs.
            "component_name": str(analyte.get("name") or "L"),
            "pka_initial": ", ".join(str(v) for v in first_pka) if first_pka else "5.0",
            "analyte_concentration": float(analyte.get("analytical_concentration", 0.0) or 0.0),
            "base_charge": int(analyte.get("base_charge", -1) or -1),
            "initial_volume": float(self.spin_initial_volume.value()),
            "titrant_concentration": float(self.spin_titrant_conc.value()),
            "titrant_type": titrant_type,
            "strong_ion_mode": strong_mode,
            "volume_offset": float(self.spin_volume_offset.value()),
            "pH_bounds": [float(self.spin_ph_min.value()), float(self.spin_ph_max.value())],
            "electrode_e0": _optional_float(self.edit_e0.text()),
            "electrode_slope": _optional_float(self.edit_slope.text()),
            "fit_electrode": bool(self.chk_fit_electrode.isChecked()),
            "sigma_pH": float(self.spin_sigma_ph.value()),
            "sigma_E": float(self.spin_sigma_emf.value()),
            "pkw": pkw,
            "kw": kw,
            "baseline": bool(self.chk_baseline.isChecked()),
        }
        if strong_mode == "manual":
            cfg["initial_strong_charge"] = float(self.spin_initial_strong_charge.value())
            cfg["titrant_strong_charge"] = float(self.spin_titrant_strong_charge.value())
        if titrant_type == "custom":
            cfg["titrant_concentrations"] = titrant_concentrations
            cfg["titrant_strong_charge"] = custom_strong_charge
        return cfg

    def _set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        controls = [
            self.btn_process,
            self.btn_choose_file,
            self.btn_import,
            self.btn_export,
            self.btn_reset,
            self.btn_render_graphs,
        ]
        for control in controls:
            control.setEnabled(not running)
        self.combo_sheet.setEnabled(
            (not running)
            and bool(self._file_path)
            and Path(str(self._file_path)).suffix.lower() == ".xlsx"
            and self.combo_sheet.count() > 0
        )
        self.btn_cancel.setEnabled(running)
        self.btn_save.setEnabled((not running) and bool(self._last_result))
        self.btn_process.setText("Processing..." if running else "Process Data")
        self._update_plot_nav()

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
        self._last_config = config
        self._last_result = None
        self._plot_pages = []
        self._plot_index = 0
        self.results_text.clear()
        if self.errors_panel is not None:
            self.errors_panel.set_errors_context(None)
        self.canvas_main.clear()
        self.log.append_text("Starting acid-base fit...")

        self._worker = FitWorker(run_acid_base_fit, config=config, parent=self)
        self._thread = self._worker.thread()
        self._thread.finished.connect(self._on_fit_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.progress.connect(self._on_worker_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.result.connect(self._on_fit_result, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_fit_error, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_fit_finished, Qt.ConnectionType.QueuedConnection)
        self._set_running(True)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.log.append_text("Cancel requested.")

    @Slot(str)
    def _on_worker_progress(self, msg: str) -> None:
        self.log.append_text(str(msg))

    @Slot(object)
    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._last_result = result
        text = str(result.get("results_text") or "")
        self.results_text.setPlainText(text)
        if text:
            self.log.append_text(text)
        self._build_plot_pages(result)
        self._update_errors_context_from_result(result)
        self._plot_index = 0
        self._render_current_plot()
        self.btn_save.setEnabled(True)

    @Slot(str)
    def _on_fit_error(self, message: str) -> None:
        self.log.append_text(f"ERROR: {message}")
        QMessageBox.critical(self, "Fit error", str(message))

    def _on_fit_finished(self) -> None:
        self._set_running(False)

    def _on_fit_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    def _build_plot_pages(self, result: dict[str, Any]) -> None:
        selected = self._selected_plot_ids()
        self._plot_pages = []
        graphs = result.get("legacy_graphs") or result.get("graphs") or {}
        titles = {item.get("id"): item.get("title") for item in (result.get("availablePlots") or [])}
        for key, png in graphs.items():
            if png and (not selected or str(key) in selected):
                self._plot_pages.append((str(key), str(titles.get(key) or key)))

    def _compute_errors_payload(
        self,
        payload: dict[str, Any],
        *,
        progress_cb=None,
        cancel_cb=None,
    ) -> dict[str, Any]:
        context = dict(payload.get("ctx") or {})
        options = dict(payload.get("options") or {})
        return compute_errors_acid_base_from_context(
            context,
            options=options,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    def _update_errors_context_from_result(self, result: dict[str, Any], *, auto_compute: bool = True) -> None:
        if self.errors_panel is None:
            return
        if not result.get("success", True):
            self.errors_panel.set_errors_context(None)
            return
        context = result.get("errors_context")
        if not isinstance(context, dict) or not context:
            self.errors_panel.set_errors_context(None)
            return
        self.errors_panel.set_errors_context(context, auto_compute=auto_compute)

    def _on_errors_output_ready(self, output: object) -> None:
        if not isinstance(output, dict):
            return
        summary = str(output.get("summary") or "").strip()
        if summary:
            self.log.append_text(summary)
        if not isinstance(self._last_result, dict):
            return
        export_data = self._last_result.setdefault("export_data", {})
        if not isinstance(export_data, dict):
            return
        frames = dict(output.get("export_frames") or {})
        if "Parameters" in frames:
            export_data["error_parameters"] = frames["Parameters"]
        if "pKa" in frames:
            export_data["error_pka_table"] = frames["pKa"]
        if "log_beta" in frames:
            export_data["error_log_beta_table"] = frames["log_beta"]
        if "Derived constants" in frames:
            export_data["derived_constants"] = frames["Derived constants"]
        if "Covariance" in frames:
            export_data["error_covariance_matrix"] = frames["Covariance"]
        if "Correlation" in frames:
            export_data["error_correlation_matrix"] = frames["Correlation"]
        if "Error diagnostics" in frames:
            export_data["error_diagnostics"] = frames["Error diagnostics"]
        if "Bootstrap samples" in frames:
            export_data["bootstrap_samples"] = frames["Bootstrap samples"]
        export_data["diagnostics_summary"] = summary
        core_metrics = dict(output.get("core_metrics") or {})
        stability_diag = dict(core_metrics.get("stability_diag") or {})
        export_data["diagnostics_full"] = str(stability_diag.get("diag_full") or "")
        export_data["stability_status"] = stability_diag.get("status")
        export_data["condition_number"] = stability_diag.get("cond_jjt")
        export_data["stability_indicator"] = stability_diag.get("stability_indicator")

    def _render_current_plot(self) -> None:
        if not self._plot_pages or not self._last_result:
            self.canvas_main.show_message("No plot available")
            self._update_plot_nav()
            return
        key, title = self._plot_pages[self._plot_index]
        graphs = self._last_result.get("legacy_graphs") or self._last_result.get("graphs") or {}
        png = graphs.get(key)
        if png:
            self.canvas_main.show_image_base64(str(png), title=title)
        else:
            self.canvas_main.show_message("No data for this plot")
        self._update_plot_nav()

    def _navigate_plot(self, delta: int) -> None:
        if not self._plot_pages:
            return
        self._plot_index = (self._plot_index + int(delta)) % len(self._plot_pages)
        self._render_current_plot()

    def _update_plot_nav(self) -> None:
        can_nav = (not self._is_running) and len(self._plot_pages) > 1
        self.btn_plot_prev.setEnabled(can_nav)
        self.btn_plot_next.setEnabled(can_nav)
        if self._plot_pages:
            _key, title = self._plot_pages[self._plot_index]
            self.lbl_plot_title.setText(str(title))
        else:
            self.lbl_plot_title.setText("Plots")

    def _on_save_results_clicked(self) -> None:
        if not self._last_result:
            QMessageBox.information(self, "No results", "Run a fit first.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save acid-base results",
            "acid_base_results.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv") or "CSV" in selected_filter:
                export_data = self._last_result.get("export_data") or {}
                table = export_data.get("experimental_vs_calculated") or export_data.get("species_vs_pH") or {}
                pd.DataFrame(table).to_csv(path, index=False)
            else:
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

    def _on_render_graphs_clicked(self) -> None:
        if not self._last_result:
            QMessageBox.information(self, "No results", "Run a fit first.")
            return
        self._build_plot_pages(self._last_result)
        self._plot_index = 0
        self._render_current_plot()
        self.log.append_text("Graphs rendered from the latest acid-base result.")

    def _on_export_config_clicked(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export config", "acid_base_config.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.log.append_text(f"Config exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _on_import_config_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import config", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.load_config(config)
            self.log.append_text(f"Config imported from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))

    def load_config(self, config: dict[str, Any]) -> None:
        if self._worker is not None:
            raise RuntimeError("Cancel the running fit before importing config.")
        file_path = str(config.get("file_path") or "")
        if file_path and Path(file_path).exists():
            self._set_file_path(file_path)
        data_type = str(config.get("data_type") or "potentiometry")
        ix = self.combo_data_type.findData(data_type)
        if ix >= 0:
            self.combo_data_type.setCurrentIndex(ix)
        sheet_name = str(config.get("sheet_name") or "")
        if sheet_name:
            ix = self.combo_sheet.findText(sheet_name)
            if ix >= 0:
                self.combo_sheet.setCurrentIndex(ix)
        model = config.get("acid_base_model") or {}
        if model:
            model_type = str(model.get("model_type") or "simple_monoprotic")
            ix = self.combo_model_type.findData(model_type)
            if ix >= 0:
                self.combo_model_type.setCurrentIndex(ix)
            self._load_components_from_model(model)
            self._refresh_species_table()
            if str(self.combo_model_type.currentData() or "") == "custom_species":
                self._load_species_from_model(model)
        pkw = config.get("pkw")
        if pkw is None and config.get("kw") not in (None, ""):
            pkw = -np.log10(float(config["kw"]))
        if pkw is not None:
            self._set_parameter_value("pKw", f"{float(pkw):.4f}")

    def _load_components_from_model(self, model: dict[str, Any]) -> None:
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            self.components_table.setRowCount(0)
            for comp in list(model.get("components") or []):
                row = self.components_table.rowCount()
                self.components_table.insertRow(row)
                use_log_beta = bool(comp.get("use_log_beta", False))
                values = list((comp.get("log_beta") or []) if use_log_beta else (comp.get("pka") or []))
                defaults = [
                    comp.get("name", f"L{row + 1}"),
                    comp.get("analytical_concentration", 0.0),
                    comp.get("base_charge", 0),
                    comp.get("n_steps", len(values)),
                    ", ".join(str(v) for v in values),
                ]
                for col, value in enumerate(defaults):
                    _set_table_text(self.components_table, row, col, value)
                use_combo = self._make_combo(["pKa", "log_beta"], "log_beta" if use_log_beta else "pKa")
                use_combo.currentIndexChanged.connect(self._on_component_table_changed)
                self.components_table.setCellWidget(row, 5, use_combo)
                fixed = self._make_checkbox(bool(comp.get("fixed_concentration", False)))
                fixed.stateChanged.connect(self._on_component_table_changed)
                self.components_table.setCellWidget(row, 6, fixed)
                role = self._make_combo(
                    ["analyte", "titrant", "background", "spectator", "imposed pH dataset"],
                    str(comp.get("role") or "analyte"),
                )
                role.currentIndexChanged.connect(self._on_component_table_changed)
                self.components_table.setCellWidget(row, 7, role)
        finally:
            self._updating_tables = was_updating

    def _load_species_from_model(self, model: dict[str, Any]) -> None:
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            self.species_table.setRowCount(0)
            for sp in list(model.get("species") or []):
                row = self.species_table.rowCount()
                self.species_table.insertRow(row)
                for col, value in enumerate(
                    [
                        sp.get("name", ""),
                        sp.get("component", ""),
                        sp.get("h_count", 0),
                        sp.get("charge", 0),
                        sp.get("log_beta", 0.0),
                        "",
                    ]
                ):
                    _set_table_text(self.species_table, row, col, value)
                self.species_table.setCellWidget(row, 6, self._make_checkbox(bool(sp.get("include", True))))
                self.species_table.setCellWidget(row, 7, self._make_checkbox(bool(sp.get("fixed", False))))
        finally:
            self._updating_tables = was_updating

    def _set_parameter_value(self, name: str, value: str, *, update_spin: bool = True) -> None:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                _set_table_text(self.parameters_table, row, 1, value)
                if update_spin and name.strip().lower() == "pkw" and hasattr(self, "spin_pkw"):
                    try:
                        pkw_for_spin = float(value)
                    except (TypeError, ValueError):
                        pkw_for_spin = self.spin_pkw.value()
                    pkw_for_spin = min(30.0, max(0.0, pkw_for_spin))
                    self.spin_pkw.blockSignals(True)
                    self.spin_pkw.setValue(pkw_for_spin)
                    self.spin_pkw.blockSignals(False)
                return

    def reset_tab(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "Cancel the running fit before resetting.")
            return
        self._file_path = ""
        self.lbl_file_status.setText("No file selected")
        self.combo_sheet.clear()
        self.combo_sheet.setEnabled(False)
        self.preview_text.clear()
        self.combo_data_type.setCurrentIndex(0)
        self.combo_model_type.setCurrentIndex(0)
        self.components_table.setRowCount(0)
        self._add_default_component()
        self.custom_titrant_table.setRowCount(0)
        self.spin_initial_volume.setValue(10.0)
        self.spin_titrant_conc.setValue(1.0e-3)
        self.combo_titrant_type.setCurrentIndex(0)
        self.combo_strong_ion.setCurrentIndex(0)
        self.spin_initial_strong_charge.setValue(0.0)
        self.spin_titrant_strong_charge.setValue(0.0)
        self.spin_volume_offset.setValue(0.0)
        self.spin_ph_min.setValue(-2.0)
        self.spin_ph_max.setValue(16.0)
        self.edit_e0.clear()
        self.edit_slope.setText("-59.16")
        self.chk_fit_electrode.setChecked(False)
        self.chk_baseline.setChecked(False)
        self.spin_pkw.setValue(14.0)
        self.spin_sigma_ph.setValue(1.0)
        self.spin_sigma_emf.setValue(1.0)
        self._refresh_species_table()
        self._refresh_parameter_table()
        self._last_config = None
        self._last_result = None
        self._plot_pages = []
        self._plot_index = 0
        self.results_text.clear()
        if self.errors_panel is not None:
            self.errors_panel.set_errors_context(None)
        self.canvas_main.show_message("Import a CSV or Excel file to begin")
        self._set_running(False)
