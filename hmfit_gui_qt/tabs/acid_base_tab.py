from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PySide6.QtCore import QItemSelectionModel, QThread, Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
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
from hmfit_core.acid_base_model_utils import (
    acid_base_constant_blocks,
    acid_base_model_from_equations,
    build_acid_base_template,
    canonicalize_acid_base_model,
    normalize_constant_mode,
    parameter_rows_from_model,
    proton_component_name,
    validate_acid_base_model,
)
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


def _parse_float_list(raw: Any, *, default: list[float] | None = None) -> list[float]:
    if raw is None:
        return list(default or [])
    if isinstance(raw, np.ndarray):
        return [float(v) for v in raw.reshape(-1)]
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    text = str(raw).replace(";", ",").strip()
    if not text:
        return list(default or [])
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _table_text(table: QTableWidget, row: int, col: int, default: str = "") -> str:
    item = table.item(row, col)
    if item is None:
        return default
    text = str(item.text() or "").strip()
    return text if text else default


def _set_table_text(table: QTableWidget, row: int, col: int, value: Any) -> None:
    table.setItem(row, col, QTableWidgetItem(str(value)))


def _volume_unit_factor_to_mL(unit: str) -> float:
    key = str(unit or "").strip().lower()
    if key in {"ul", "µl"}:
        return 1.0e-3
    if key == "l":
        return 1.0e3
    return 1.0


@dataclass
class AcidBaseUiState:
    data_type: str = "potentiometry"
    file_path: str = ""
    sheet_name: str = ""
    volume_column: str = ""
    volume_unit: str = "mL"
    signal_type: str = "pH"
    signal_column: str = ""
    analyte_name: str = "L"
    analyte_concentration: float = 1.0e-3
    initial_volume: float = 10.0
    titrant_type: str = "base"
    titrant_concentration: float = 1.0e-3
    charge_L: int = -1
    species_rows: list[dict[str, Any]] = field(default_factory=list)
    pka_rows: list[dict[str, Any]] = field(default_factory=list)
    pkw: float = 14.0
    pkw_bounds: tuple[float, float] = (0.0, 30.0)
    fit_pkw: bool = False
    fit_analyte_concentration: bool = False
    fit_titrant_concentration: bool = False
    fit_volume_offset: bool = False
    fit_electrode: bool = False
    advanced_options: dict[str, Any] = field(default_factory=dict)


def _normalise_column_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("µ", "u")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _guess_volume_unit_from_name(name: str) -> str | None:
    key = _normalise_column_name(name)
    if "ul" in key or "microl" in key:
        return "µL"
    if "ml" in key:
        return "mL"
    if re.search(r"(?:^|_)l(?:$|_)", key) and "ml" not in key and "ul" not in key:
        return "L"
    return None


def _volume_column_score(name: str) -> int:
    key = _normalise_column_name(name)
    score = 0
    if any(token in key for token in ("volume", "vol", "vadd", "v_added", "added")):
        score += 20
    if key in {"v", "volume", "vol", "volume_ml", "v_ml", "volume_added", "vadd"}:
        score += 30
    if _guess_volume_unit_from_name(name):
        score += 10
    return score


def _signal_column_score(name: str) -> tuple[int, str | None]:
    key = _normalise_column_name(name)
    if "ph" in key:
        return 40, "pH"
    if any(token in key for token in ("emf", "mv", "e_mv", "potential")) or key in {"e", "e_m_v"}:
        return 40, "mV"
    if key == "signal":
        return 10, None
    return 0, None


class AcidBaseTab(QWidget):
    COMPONENT_COLUMNS = [
        "Role",
        "Component name",
        "Analytical concentration",
        "Charge",
        "Is proton",
        "Is titrant",
        "Is background/spectator",
        "Fixed concentration",
    ]
    SPECIES_COLUMNS = [
        "Species name",
        "Charge",
        "h_count",
        "Include",
        "Observable",
        "Fixed",
        "Non-observable / non-absorbing",
        "Parent component or group",
    ]
    PARAM_COLUMNS = [
        "Parameter",
        "Type",
        "Initial value",
        "Min",
        "Max",
        "Fit / Fixed",
        "Linked species",
        "Description",
    ]
    BASIC_SPECIES_COLUMNS = ["Species name", "h_count", "Charge", "Include", "Observable", "Fixed"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FitWorker | None = None
        self._thread: QThread | None = None
        self._file_path = ""
        self._current_data_frame: pd.DataFrame | None = None
        self._potentiometry_warning_messages: list[str] = []
        self._last_result: dict[str, Any] | None = None
        self._last_config: dict[str, Any] | None = None
        self._plot_pages: list[tuple[str, str]] = []
        self._plot_index = 0
        self._is_running = False
        self._updating_tables = False
        self._updating_basic_fields = False
        self._advanced_tables_dirty = True
        self._advanced_table_rebuild_count = 0
        self._pot_preview_restore_mask: list[bool] | None = None
        self.ui_state = AcidBaseUiState()
        self.errors_panel: ErrorAnalysisWidget | None = None

        self._build_ui()
        self._load_template("simple_monoprotic")
        self._refresh_parameter_table()
        self._on_titrant_type_changed()
        self._on_signal_type_changed()
        self._sync_pkw_ui()
        self._sync_plot_options_for_analysis()
        self._sync_advanced_ui(False)
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

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Advanced mode", left_container))
        self.chk_advanced_mode = QCheckBox("OFF / ON", left_container)
        self.chk_advanced_mode.setChecked(False)
        self.chk_advanced_mode.toggled.connect(self._sync_advanced_ui)
        mode_row.addWidget(self.chk_advanced_mode)
        self.btn_suggest_setup = QPushButton("Suggest setup from data", left_container)
        self.btn_suggest_setup.clicked.connect(self._suggest_setup_from_data)
        mode_row.addWidget(self.btn_suggest_setup)
        mode_row.addStretch(1)
        left_layout.addLayout(mode_row)

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
        self._data_group = QGroupBox("Experimental data", parent)
        data_layout = QVBoxLayout(self._data_group)
        form = QFormLayout()

        self.lbl_data_type = QLabel("Data type", self._data_group)
        self.combo_data_type = QComboBox(self._data_group)
        self.combo_data_type.addItem("Potentiometry pH/EMF", "potentiometry")
        self.combo_data_type.addItem("Spectroscopy signal vs pH", "spectroscopy")
        self.combo_data_type.addItem("1H NMR shifts vs pH", "nmr")
        self.combo_data_type.currentIndexChanged.connect(self._on_data_type_changed)
        form.addRow(self.lbl_data_type, self.combo_data_type)
        self.lbl_data_type.setVisible(False)
        self.combo_data_type.setVisible(False)

        self.lbl_sheet = QLabel("Data sheet", self._data_group)
        self.combo_sheet = QComboBox(self._data_group)
        self.combo_sheet.setEnabled(False)
        self.combo_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        form.addRow(self.lbl_sheet, self.combo_sheet)
        self.lbl_sheet.setVisible(False)
        self.combo_sheet.setVisible(False)
        data_layout.addLayout(form)

        file_row = QHBoxLayout()
        self.btn_choose_file = QPushButton("Choose file...", self._data_group)
        self.btn_choose_file.clicked.connect(self._on_choose_file_clicked)
        file_row.addWidget(self.btn_choose_file)
        self.lbl_file_status = QLabel("No file selected", self._data_group)
        self.lbl_file_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_row.addWidget(self.lbl_file_status, 1)
        data_layout.addLayout(file_row)

        self._data_import_stack = QStackedWidget(self._data_group)
        data_layout.addWidget(self._data_import_stack, 1)

        pot_page = QWidget(self._data_import_stack)
        pot_layout = QVBoxLayout(pot_page)
        pot_form = QFormLayout()

        self.combo_volume_column = QComboBox(pot_page)
        self.combo_volume_column.currentIndexChanged.connect(self._refresh_potentiometry_preview)
        pot_form.addRow("Volume column", self.combo_volume_column)

        self.combo_volume_unit = QComboBox(pot_page)
        self.combo_volume_unit.addItem("µL", "µL")
        self.combo_volume_unit.addItem("mL", "mL")
        self.combo_volume_unit.addItem("L", "L")
        self.combo_volume_unit.setCurrentIndex(self.combo_volume_unit.findData("mL"))
        self.combo_volume_unit.currentIndexChanged.connect(self._refresh_potentiometry_preview)
        pot_form.addRow("Volume unit", self.combo_volume_unit)

        self.combo_signal_type = QComboBox(pot_page)
        self.combo_signal_type.addItem("pH", "pH")
        self.combo_signal_type.addItem("EMF", "mV")
        self.combo_signal_type.currentIndexChanged.connect(self._on_signal_type_changed)
        pot_form.addRow("Signal type", self.combo_signal_type)

        self.combo_signal_column = QComboBox(pot_page)
        self.combo_signal_column.currentIndexChanged.connect(self._refresh_potentiometry_preview)
        pot_form.addRow("Signal column", self.combo_signal_column)

        self.chk_ideal_nernst = QCheckBox("Use ideal Nernst slope", pot_page)
        self.chk_ideal_nernst.setChecked(True)
        self.chk_ideal_nernst.toggled.connect(self._on_ideal_nernst_toggled)
        pot_form.addRow("", self.chk_ideal_nernst)
        pot_layout.addLayout(pot_form)

        self.lbl_pot_validation = QLabel("", pot_page)
        self.lbl_pot_validation.setWordWrap(True)
        self.lbl_pot_validation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pot_layout.addWidget(self.lbl_pot_validation)

        preview_buttons = QHBoxLayout()
        self.btn_include_all_points = QPushButton("Include all", pot_page)
        self.btn_include_all_points.clicked.connect(lambda: self._set_preview_inclusion("all"))
        self.btn_exclude_all_points = QPushButton("Exclude all", pot_page)
        self.btn_exclude_all_points.clicked.connect(lambda: self._set_preview_inclusion("none"))
        self.btn_exclude_first_point = QPushButton("Exclude first point", pot_page)
        self.btn_exclude_first_point.clicked.connect(lambda: self._set_preview_inclusion("exclude_first"))
        self.btn_exclude_selected_points = QPushButton("Exclude selected", pot_page)
        self.btn_exclude_selected_points.clicked.connect(lambda: self._set_preview_inclusion("exclude_selected"))
        self.btn_restore_point_selection = QPushButton("Restore selection", pot_page)
        self.btn_restore_point_selection.clicked.connect(lambda: self._set_preview_inclusion("restore"))
        for btn in (
            self.btn_include_all_points,
            self.btn_exclude_all_points,
            self.btn_exclude_first_point,
            self.btn_exclude_selected_points,
            self.btn_restore_point_selection,
        ):
            preview_buttons.addWidget(btn)
        preview_buttons.addStretch(1)
        pot_layout.addLayout(preview_buttons)

        self.table_pot_preview = QTableWidget(0, 5, pot_page)
        self.table_pot_preview.setHorizontalHeaderLabels(
            [
                "Include",
                "Original row",
                "Selected volume",
                "Volume (mL)",
                "Observed signal",
            ]
        )
        self.table_pot_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_pot_preview.horizontalHeader().setStretchLastSection(True)
        self.table_pot_preview.setMinimumHeight(180)
        pot_layout.addWidget(self.table_pot_preview, 1)
        self._data_import_stack.addWidget(pot_page)

        generic_page = QWidget(self._data_import_stack)
        generic_layout = QVBoxLayout(generic_page)
        self.preview_text = QPlainTextEdit(self._data_group)
        self.preview_text.setReadOnly(True)
        self.preview_text.setMinimumHeight(130)
        generic_layout.addWidget(self.preview_text, 1)
        self._data_import_stack.addWidget(generic_page)

        layout.addWidget(self._data_group)

    def _build_model_tab(self) -> None:
        tab = QWidget(self.model_opt_tabs)
        layout = QVBoxLayout(tab)

        setup_group = QGroupBox("Titration setup", tab)
        setup_form = QFormLayout(setup_group)
        self.edit_analyte_name = QLineEdit("L", setup_group)
        self.edit_analyte_name.textChanged.connect(self._on_basic_species_model_changed)
        self.spin_initial_volume = QDoubleSpinBox(setup_group)
        self.spin_initial_volume.setDecimals(6)
        self.spin_initial_volume.setRange(1.0e-12, 1.0e9)
        self.spin_initial_volume.setValue(10.0)
        self.spin_analyte_conc = QDoubleSpinBox(setup_group)
        self.spin_analyte_conc.setDecimals(8)
        self.spin_analyte_conc.setRange(1.0e-12, 1.0e6)
        self.spin_analyte_conc.setValue(1.0e-3)
        self.spin_analyte_conc.setSingleStep(1.0e-4)
        self.spin_analyte_conc.valueChanged.connect(self._on_analyte_concentration_changed)
        self.spin_titrant_conc = QDoubleSpinBox(setup_group)
        self.spin_titrant_conc.setDecimals(8)
        self.spin_titrant_conc.setRange(1.0e-12, 1.0e6)
        self.spin_titrant_conc.setValue(1.0e-3)
        self.spin_titrant_conc.setSingleStep(1.0e-4)
        self.spin_titrant_conc.valueChanged.connect(self._on_titrant_concentration_changed)
        self.combo_titrant_type = QComboBox(setup_group)
        self.combo_titrant_type.addItem("strong base", "base")
        self.combo_titrant_type.addItem("strong acid", "acid")
        self.combo_titrant_type.addItem("custom titrant", "custom")
        self.combo_titrant_type.currentIndexChanged.connect(self._on_titrant_type_changed)
        self.spin_base_charge = QSpinBox(setup_group)
        self.spin_base_charge.setRange(-20, 20)
        self.spin_base_charge.setValue(-1)
        self.spin_base_charge.valueChanged.connect(self._on_base_charge_changed)
        self.combo_common_model = QComboBox(setup_group)
        self.combo_common_model.addItem("Monoprotic acid: HA / A-", "monoprotic_acid")
        self.combo_common_model.addItem("Monoprotic base: B / BH+", "monoprotic_base")
        self.combo_common_model.addItem("Diprotic acid: H2A / HA- / A2-", "diprotic_acid")
        self.combo_common_model.addItem("Diprotic base: B / BH+ / BH2 2+", "diprotic_base")
        self.combo_common_model.addItem("Custom", "custom")
        self.combo_common_model.currentIndexChanged.connect(self._on_common_model_changed)
        setup_form.addRow("Common model", self.combo_common_model)
        setup_form.addRow("Analyte name", self.edit_analyte_name)
        setup_form.addRow("Initial volume", self.spin_initial_volume)
        setup_form.addRow("Analytical concentration", self.spin_analyte_conc)
        setup_form.addRow("Titrant type", self.combo_titrant_type)
        setup_form.addRow("Titrant concentration", self.spin_titrant_conc)
        setup_form.addRow("Charge of L", self.spin_base_charge)
        layout.addWidget(setup_group)

        chemical_group = QGroupBox("Acid-base species model", tab)
        chemical_layout = QVBoxLayout(chemical_group)
        template_row = QHBoxLayout()
        self.lbl_model_preset = QLabel("Model preset", chemical_group)
        template_row.addWidget(self.lbl_model_preset)
        self.combo_model_type = QComboBox(tab)
        self.combo_model_type.addItem("Simple monoprotic acid/base", "simple_monoprotic")
        self.combo_model_type.addItem("Diprotic acid/base", "diprotic_ligand")
        self.combo_model_type.addItem("Polyprotic acid/base", "polyprotic_acid_base")
        self.combo_model_type.addItem("Multiple acid-base components", "multiple_components")
        self.combo_model_type.addItem("Custom acid-base system", "custom_acid_base_system")
        self.combo_model_type.addItem(
            "Coupled acid-base / host-guest model (future)",
            "coupled_future",
        )
        self.combo_model_type.currentIndexChanged.connect(self._on_model_template_changed)
        future_idx = self.combo_model_type.findData("coupled_future")
        future_item = self.combo_model_type.model().item(future_idx)
        if future_item is not None:
            future_item.setEnabled(False)
        template_row.addWidget(self.combo_model_type, 1)
        self.btn_load_template = QPushButton("Load template", chemical_group)
        self.btn_load_template.clicked.connect(self._on_load_template_clicked)
        template_row.addWidget(self.btn_load_template)
        chemical_layout.addLayout(template_row)

        self.lbl_pka_count = QLabel("Number of labile protons", chemical_group)
        self.spin_pka_count = QSpinBox(chemical_group)
        self.spin_pka_count.setRange(1, 12)
        self.spin_pka_count.setValue(3)
        self.spin_pka_count.valueChanged.connect(self._on_pka_count_changed)
        pka_count_row = QHBoxLayout()
        pka_count_row.addWidget(self.lbl_pka_count)
        pka_count_row.addWidget(self.spin_pka_count)
        self.btn_generate_species_basic = QPushButton("Generate species", chemical_group)
        self.btn_generate_species_basic.clicked.connect(lambda: self._generate_basic_species(int(self.spin_pka_count.value())))
        pka_count_row.addWidget(self.btn_generate_species_basic)
        pka_count_row.addStretch(1)
        chemical_layout.addLayout(pka_count_row)

        self.table_basic_species = QTableWidget(0, len(self.BASIC_SPECIES_COLUMNS), chemical_group)
        self.table_basic_species.setHorizontalHeaderLabels(self.BASIC_SPECIES_COLUMNS)
        self.table_basic_species.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_basic_species.horizontalHeader().setStretchLastSection(True)
        self.table_basic_species.setMinimumHeight(130)
        self.table_basic_species.itemChanged.connect(self._on_basic_species_item_changed)
        chemical_layout.addWidget(self.table_basic_species)

        species_buttons = QHBoxLayout()
        self.btn_basic_monoprotic = QPushButton("Generate monoprotic", chemical_group)
        self.btn_basic_monoprotic.clicked.connect(lambda: self._generate_basic_species(1))
        species_buttons.addWidget(self.btn_basic_monoprotic)
        self.btn_basic_diprotic = QPushButton("Generate diprotic", chemical_group)
        self.btn_basic_diprotic.clicked.connect(lambda: self._generate_basic_species(2))
        species_buttons.addWidget(self.btn_basic_diprotic)
        self.btn_basic_triprotic = QPushButton("Generate triprotic", chemical_group)
        self.btn_basic_triprotic.clicked.connect(lambda: self._generate_basic_species(3))
        species_buttons.addWidget(self.btn_basic_triprotic)
        self.btn_basic_add_species = QPushButton("Add custom species", chemical_group)
        self.btn_basic_add_species.clicked.connect(self._add_basic_species_row)
        species_buttons.addWidget(self.btn_basic_add_species)
        self.btn_basic_remove_species = QPushButton("Remove selected", chemical_group)
        self.btn_basic_remove_species.clicked.connect(self._remove_selected_basic_species)
        species_buttons.addWidget(self.btn_basic_remove_species)
        self.btn_show_matrix = QPushButton("Show generated matrix", chemical_group)
        self.btn_show_matrix.setCheckable(True)
        self.btn_show_matrix.toggled.connect(self._refresh_basic_matrix)
        species_buttons.addWidget(self.btn_show_matrix)
        species_buttons.addStretch(1)
        chemical_layout.addLayout(species_buttons)

        self.basic_matrix_table = QTableWidget(0, 2, chemical_group)
        self.basic_matrix_table.setHorizontalHeaderLabels(["L", "H"])
        self.basic_matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.basic_matrix_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.basic_matrix_table.setVisible(False)
        chemical_layout.addWidget(self.basic_matrix_table)

        self.basic_pka_group = QGroupBox("Initial pKa guesses", tab)
        self.basic_pka_group.setVisible(False)
        pka_layout = QVBoxLayout(self.basic_pka_group)
        self.table_basic_pka = QTableWidget(0, 5, self.basic_pka_group)
        self.table_basic_pka.setHorizontalHeaderLabels(["Parameter", "Initial guess", "Min", "Max", "Fit"])
        self.table_basic_pka.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_basic_pka.horizontalHeader().setStretchLastSection(True)
        self.table_basic_pka.setMinimumHeight(90)
        self.table_basic_pka.itemChanged.connect(self._on_basic_pka_table_changed)
        pka_layout.addWidget(self.table_basic_pka)
        layout.addWidget(chemical_group)

        self.model_advanced_group = QGroupBox("Advanced options", tab)
        self.model_advanced_group.setCheckable(True)
        self.model_advanced_group.setChecked(False)
        self.model_advanced_group.toggled.connect(self._sync_advanced_ui)
        self.model_advanced_content = QWidget(self.model_advanced_group)
        advanced_outer = QVBoxLayout(self.model_advanced_group)
        advanced_outer.addWidget(self.model_advanced_content)
        advanced_layout = QVBoxLayout(self.model_advanced_content)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Model definition", tab))
        self.radio_model_matrix = QRadioButton("Matriz estequiometrica", tab)
        self.radio_model_equations = QRadioButton("Editor de ecuaciones", tab)
        self.radio_model_matrix.setChecked(True)
        self.radio_model_matrix.toggled.connect(self._sync_model_mode_ui)
        self.radio_model_equations.toggled.connect(self._sync_model_mode_ui)
        mode_row.addWidget(self.radio_model_matrix)
        mode_row.addWidget(self.radio_model_equations)
        mode_row.addStretch(1)
        advanced_layout.addLayout(mode_row)

        self.constants_group = QGroupBox("Constant convention", tab)
        constants_form = QFormLayout(self.constants_group)
        self.combo_constant_mode = QComboBox(self.constants_group)
        self.combo_constant_mode.addItem("pKa", "pKa")
        self.combo_constant_mode.addItem("log_beta", "log_beta")
        self.combo_constant_mode.currentIndexChanged.connect(self._refresh_parameter_table)
        constants_form.addRow("Primary input mode", self.combo_constant_mode)
        advanced_layout.addWidget(self.constants_group)

        self.model_definition_stack = QStackedWidget(tab)
        advanced_layout.addWidget(self.model_definition_stack, 1)

        matrix_page = QWidget(self.model_definition_stack)
        matrix_layout = QVBoxLayout(matrix_page)
        dims_group = QGroupBox("Model dimensions", matrix_page)
        dims_layout = QGridLayout(dims_group)
        self.spin_n_components_model = QSpinBox(dims_group)
        self.spin_n_components_model.setRange(0, 20)
        self.spin_n_components_model.setValue(2)
        self.spin_n_species_model = QSpinBox(dims_group)
        self.spin_n_species_model.setRange(0, 100)
        self.spin_n_species_model.setValue(2)
        self.btn_define_model_dimensions = QPushButton("Define Model Dimensions", dims_group)
        self.btn_define_model_dimensions.clicked.connect(self._define_model_dimensions)
        dims_layout.addWidget(self.btn_define_model_dimensions, 0, 0, 1, 2)
        dims_layout.addWidget(QLabel("Number of components"), 1, 0)
        dims_layout.addWidget(self.spin_n_components_model, 1, 1)
        dims_layout.addWidget(QLabel("Number of species"), 2, 0)
        dims_layout.addWidget(self.spin_n_species_model, 2, 1)
        matrix_layout.addWidget(dims_group)

        self.stoich_group = QGroupBox("Stoichiometric matrix", matrix_page)
        stoich_layout = QVBoxLayout(self.stoich_group)
        self.lbl_stoich_selection_help = QLabel("Select non-observable / non-absorbing species", self.stoich_group)
        stoich_layout.addWidget(self.lbl_stoich_selection_help)
        self.stoich_table = QTableWidget(0, 0, self.stoich_group)
        self.stoich_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stoich_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.stoich_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.stoich_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.stoich_table.setAlternatingRowColors(True)
        self.stoich_table.itemChanged.connect(self._on_stoich_table_changed)
        self.stoich_table.itemSelectionChanged.connect(self._on_stoich_selection_changed)
        stoich_layout.addWidget(self.stoich_table, 1)
        matrix_layout.addWidget(self.stoich_group)

        self.components_group = QGroupBox("Components", matrix_page)
        comp_layout = QVBoxLayout(self.components_group)
        self.components_table = QTableWidget(0, len(self.COMPONENT_COLUMNS), self.components_group)
        self.components_table.setHorizontalHeaderLabels(self.COMPONENT_COLUMNS)
        self.components_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.components_table.horizontalHeader().setStretchLastSection(True)
        self.components_table.setMinimumHeight(150)
        self.components_table.itemChanged.connect(self._on_component_table_changed)
        comp_layout.addWidget(self.components_table)
        comp_buttons = QHBoxLayout()
        self.btn_add_component = QPushButton("Add component", self.components_group)
        self.btn_add_component.clicked.connect(lambda: self._insert_component_row())
        comp_buttons.addWidget(self.btn_add_component)
        self.btn_remove_component = QPushButton("Remove selected", self.components_group)
        self.btn_remove_component.clicked.connect(self._remove_selected_component)
        comp_buttons.addWidget(self.btn_remove_component)
        self.btn_sync_names = QPushButton("Sync headers", self.components_group)
        self.btn_sync_names.clicked.connect(self._sync_matrix_headers)
        comp_buttons.addWidget(self.btn_sync_names)
        comp_buttons.addStretch(1)
        comp_layout.addLayout(comp_buttons)
        matrix_layout.addWidget(self.components_group)

        self.species_group = QGroupBox("Species", matrix_page)
        species_layout = QVBoxLayout(self.species_group)
        self.species_table = QTableWidget(0, len(self.SPECIES_COLUMNS), self.species_group)
        self.species_table.setHorizontalHeaderLabels(self.SPECIES_COLUMNS)
        self.species_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.species_table.horizontalHeader().setStretchLastSection(True)
        self.species_table.setMinimumHeight(190)
        self.species_table.itemChanged.connect(self._on_species_table_changed)
        species_layout.addWidget(self.species_table)
        species_buttons = QHBoxLayout()
        self.btn_add_species = QPushButton("Add species", self.species_group)
        self.btn_add_species.clicked.connect(lambda: self._insert_species_row())
        species_buttons.addWidget(self.btn_add_species)
        self.btn_remove_species = QPushButton("Remove selected", self.species_group)
        self.btn_remove_species.clicked.connect(self._remove_selected_species)
        species_buttons.addWidget(self.btn_remove_species)
        self.btn_generate_species = QPushButton("Generate simple species", self.species_group)
        self.btn_generate_species.clicked.connect(self._generate_species_from_matrix)
        species_buttons.addWidget(self.btn_generate_species)
        species_buttons.addStretch(1)
        species_layout.addLayout(species_buttons)
        matrix_layout.addWidget(self.species_group)

        self.model_definition_stack.addWidget(matrix_page)

        equations_page = QWidget(self.model_definition_stack)
        equations_layout = QVBoxLayout(equations_page)
        equations_layout.addWidget(
            QLabel(
                "Examples:\n"
                "# Monoprotic\n"
                "L + H <=> HL ; pKa=5.20\n\n"
                "# Diprotic, stepwise\n"
                "L + H <=> HL ; pKa=4.50\n"
                "HL + H <=> H2L ; pKa=8.90\n\n"
                "# Diprotic, cumulative\n"
                "L + H <=> HL ; logB=4.50\n"
                "L + 2H <=> H2L ; logB=13.40",
                equations_page,
            )
        )
        self.equation_editor = QPlainTextEdit(equations_page)
        self.equation_editor.setPlaceholderText(
            "L + H <=> HL ; pKa=5.20\n"
            "HL + H <=> H2L ; pKa=8.90"
        )
        equations_layout.addWidget(self.equation_editor, 1)
        eq_buttons = QHBoxLayout()
        self.btn_apply_equations = QPushButton("Apply equations", equations_page)
        self.btn_apply_equations.clicked.connect(self._apply_equations_model)
        eq_buttons.addWidget(self.btn_apply_equations)
        eq_buttons.addStretch(1)
        equations_layout.addLayout(eq_buttons)
        self.model_definition_stack.addWidget(equations_page)
        self._sync_model_mode_ui()

        self.titration_group = QGroupBox("Potentiometric titration model", tab)
        titration_layout = QVBoxLayout(self.titration_group)
        titration_form = QFormLayout()
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
        self.spin_volume_offset.valueChanged.connect(self._on_volume_offset_changed)
        self.spin_ph_min = QDoubleSpinBox(self.titration_group)
        self.spin_ph_min.setDecimals(3)
        self.spin_ph_min.setRange(-10.0, 30.0)
        self.spin_ph_min.setValue(-2.0)
        self.spin_ph_max = QDoubleSpinBox(self.titration_group)
        self.spin_ph_max.setDecimals(3)
        self.spin_ph_max.setRange(-10.0, 30.0)
        self.spin_ph_max.setValue(16.0)

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
        advanced_layout.addWidget(self.titration_group)
        layout.addWidget(self.model_advanced_group)

        self.model_opt_tabs.addTab(tab, "Model")

    def _build_optimization_tab(self) -> None:
        tab = QWidget(self.model_opt_tabs)
        layout = QVBoxLayout(tab)

        self.parameters_table = QTableWidget(0, len(self.PARAM_COLUMNS), tab)
        self.parameters_table.setHorizontalHeaderLabels(self.PARAM_COLUMNS)
        self.parameters_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.parameters_table.horizontalHeader().setStretchLastSection(True)
        self.parameters_table.setMinimumHeight(220)
        self.parameters_table.itemChanged.connect(self._on_parameter_table_item_changed)
        self.parameters_table.setToolTip(
            "pKw is accepted for all acid-base datasets, but it only affects "
            "potentiometric electroneutrality calculations in v1. Spectroscopy "
            "and NMR fits use the measured pH directly."
        )
        layout.addWidget(self.parameters_table, 1)

        fit_options_group = QGroupBox("Fit options", tab)
        fit_options_layout = QVBoxLayout(fit_options_group)
        self.chk_fit_analyte_conc = QCheckBox("Fit analyte concentration", fit_options_group)
        self.chk_fit_titrant_conc = QCheckBox("Fit titrant concentration", fit_options_group)
        self.chk_fit_volume_offset = QCheckBox("Fit volume offset", fit_options_group)
        self.chk_fit_electrode_basic = QCheckBox("Fit electrode parameters", fit_options_group)
        self.chk_fit_electrode_basic.toggled.connect(self._on_fit_electrode_basic_toggled)
        for chk in (
            self.chk_fit_analyte_conc,
            self.chk_fit_titrant_conc,
            self.chk_fit_volume_offset,
            self.chk_fit_electrode_basic,
        ):
            chk.setChecked(False)
            chk.toggled.connect(self._refresh_parameter_table)
            fit_options_layout.addWidget(chk)
        layout.addWidget(fit_options_group)

        self.pkw_group = QGroupBox("Apparent pKw / medium autoprotolysis", tab)
        pkw_layout = QFormLayout(self.pkw_group)
        self.combo_medium = QComboBox(self.pkw_group)
        self.combo_medium.addItem("Aqueous, fixed pKw = 14", "aqueous")
        self.combo_medium.addItem("Mixed/non-aqueous, use apparent pKw", "mixed")
        self.combo_medium.addItem("Custom", "custom")
        self.combo_medium.currentIndexChanged.connect(self._on_medium_changed)
        pkw_layout.addRow("Medium / solvent", self.combo_medium)
        self.spin_pkw = QDoubleSpinBox(self.pkw_group)
        self.spin_pkw.setDecimals(4)
        self.spin_pkw.setRange(-50.0, 100.0)
        self.spin_pkw.setValue(14.0)
        self.spin_pkw.setToolTip(
            "Use fixed pKw for ordinary aqueous titrations. Enable fitting only "
            "for non-aqueous or mixed solvents, or when an apparent pH scale is being used."
        )
        self.spin_pkw.valueChanged.connect(self._on_pkw_spin_changed)
        self.spin_pkw_min = QDoubleSpinBox(self.pkw_group)
        self.spin_pkw_min.setDecimals(4)
        self.spin_pkw_min.setRange(-50.0, 100.0)
        self.spin_pkw_min.setValue(0.0)
        self.spin_pkw_min.valueChanged.connect(self._on_pkw_bounds_changed)
        self.spin_pkw_max = QDoubleSpinBox(self.pkw_group)
        self.spin_pkw_max.setDecimals(4)
        self.spin_pkw_max.setRange(-50.0, 100.0)
        self.spin_pkw_max.setValue(30.0)
        self.spin_pkw_max.valueChanged.connect(self._on_pkw_bounds_changed)
        pkw_bounds_row = QHBoxLayout()
        pkw_bounds_row.addWidget(self.spin_pkw_min)
        pkw_bounds_row.addWidget(QLabel("to", self.pkw_group))
        pkw_bounds_row.addWidget(self.spin_pkw_max)
        self.chk_fit_pkw = QCheckBox("Fit apparent pKw", self.pkw_group)
        self.chk_fit_pkw.setChecked(False)
        self.chk_fit_pkw.toggled.connect(self._on_fit_pkw_toggled)
        self.lbl_pkw_help = QLabel(
            "Use fixed pKw for ordinary aqueous titrations. Enable fitting only for "
            "non-aqueous or mixed solvents, or when an apparent pH scale is being used.",
            self.pkw_group,
        )
        self.lbl_pkw_help.setWordWrap(True)
        pkw_layout.addRow("Initial pKw_app", self.spin_pkw)
        pkw_layout.addRow("Min / Max", pkw_bounds_row)
        pkw_layout.addRow("", self.chk_fit_pkw)
        pkw_layout.addRow("", self.lbl_pkw_help)
        layout.addWidget(self.pkw_group)

        opt_buttons = QHBoxLayout()
        self.btn_refresh_params = QPushButton("Refresh parameter table", tab)
        self.btn_refresh_params.clicked.connect(self._refresh_parameter_table)
        opt_buttons.addWidget(self.btn_refresh_params)
        opt_buttons.addStretch(1)
        layout.addLayout(opt_buttons)

        self.electrode_group = QGroupBox("Electrode / spectroscopy options", tab)
        electrode_layout = QFormLayout(self.electrode_group)
        self.edit_e0 = QLineEdit("0.0", self.electrode_group)
        self.edit_e0.setPlaceholderText("fixed E0, mV")
        self.edit_e0.textChanged.connect(self._on_electrode_text_changed)
        self.edit_slope = QLineEdit("-59.16", self.electrode_group)
        self.edit_slope.setPlaceholderText("fixed slope, mV/pH")
        self.edit_slope.textChanged.connect(self._on_electrode_text_changed)
        self.chk_fit_electrode = QCheckBox("Fit E0 and slope for EMF data", self.electrode_group)
        self.chk_fit_electrode.toggled.connect(self._on_fit_electrode_advanced_toggled)
        self.chk_baseline = QCheckBox("Include linear baseline in spectroscopy", self.electrode_group)
        self.chk_baseline.toggled.connect(self._refresh_parameter_table)
        electrode_layout.addRow("Electrode E0", self.edit_e0)
        electrode_layout.addRow("Electrode slope", self.edit_slope)
        electrode_layout.addRow("", self.chk_fit_electrode)
        electrode_layout.addRow("", self.chk_baseline)
        layout.addWidget(self.electrode_group)

        self.weight_group = QGroupBox("Residual weights", tab)
        weight_layout = QFormLayout(self.weight_group)
        self.spin_sigma_ph = QDoubleSpinBox(self.weight_group)
        self.spin_sigma_ph.setDecimals(6)
        self.spin_sigma_ph.setRange(1.0e-12, 1.0e6)
        self.spin_sigma_ph.setValue(1.0)
        self.spin_sigma_emf = QDoubleSpinBox(self.weight_group)
        self.spin_sigma_emf.setDecimals(6)
        self.spin_sigma_emf.setRange(1.0e-12, 1.0e6)
        self.spin_sigma_emf.setValue(1.0)
        weight_layout.addRow("sigma pH", self.spin_sigma_ph)
        weight_layout.addRow("sigma EMF", self.spin_sigma_emf)
        layout.addWidget(self.weight_group)

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

    def _advanced_visible(self) -> bool:
        if self._custom_model_template_selected():
            return True
        if hasattr(self, "chk_advanced_mode"):
            return bool(self.chk_advanced_mode.isChecked())
        return bool(hasattr(self, "model_advanced_group") and self.model_advanced_group.isChecked())

    def _custom_model_template_selected(self) -> bool:
        if not hasattr(self, "combo_model_type"):
            return False
        return str(self.combo_model_type.currentData() or "") in {
            "multiple_components",
            "custom_acid_base_system",
        }

    def _model_definition_is_custom(self, model_def: dict[str, Any] | None) -> bool:
        return str((model_def or {}).get("template_id") or (model_def or {}).get("model_type") or "") == "custom_acid_base_system"

    def _free_custom_model_selected(self) -> bool:
        if not hasattr(self, "combo_model_type"):
            return False
        return str(self.combo_model_type.currentData() or "") == "custom_acid_base_system"

    def _set_combo_item_enabled(self, combo: QComboBox, data: str, enabled: bool) -> None:
        idx = combo.findData(data)
        item = combo.model().item(idx) if idx >= 0 else None
        if item is not None:
            item.setEnabled(bool(enabled))

    def _sync_advanced_ui(self, checked: bool | None = None) -> None:
        advanced = (self._advanced_visible() if checked is None else bool(checked)) or self._custom_model_template_selected()
        if hasattr(self, "chk_advanced_mode") and self.chk_advanced_mode.isChecked() != advanced:
            self.chk_advanced_mode.blockSignals(True)
            self.chk_advanced_mode.setChecked(advanced)
            self.chk_advanced_mode.blockSignals(False)
        if hasattr(self, "model_advanced_group") and self.model_advanced_group.isChecked() != advanced:
            self.model_advanced_group.blockSignals(True)
            self.model_advanced_group.setChecked(advanced)
            self.model_advanced_group.blockSignals(False)
        if hasattr(self, "model_advanced_content"):
            self.model_advanced_content.setVisible(advanced)
        if hasattr(self, "model_advanced_group"):
            self.model_advanced_group.setVisible(advanced)
        if hasattr(self, "pkw_group"):
            self._sync_pkw_ui()
        if hasattr(self, "electrode_group"):
            is_emf = str(self.combo_signal_type.currentData() or "").lower() == "mv" if hasattr(self, "combo_signal_type") else False
            self.electrode_group.setVisible((advanced and is_emf) or (is_emf and self.chk_fit_electrode_basic.isChecked()))
        if hasattr(self, "weight_group"):
            self.weight_group.setVisible(advanced)
        free_custom = self._free_custom_model_selected()
        if hasattr(self, "constants_group"):
            self.constants_group.setVisible(advanced and not free_custom)
        if hasattr(self, "components_group"):
            self.components_group.setVisible(advanced and not free_custom)
        if hasattr(self, "species_group"):
            self.species_group.setVisible(advanced and not free_custom)
        for widget_name in (
            "lbl_model_preset",
            "combo_model_type",
            "btn_load_template",
        ):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setVisible(True)
        for widget_name in (
            "btn_basic_monoprotic",
            "btn_basic_diprotic",
            "btn_basic_triprotic",
        ):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setVisible(advanced)
        if hasattr(self, "combo_model_type"):
            self._set_combo_item_enabled(self.combo_model_type, "multiple_components", True)
            self._set_combo_item_enabled(self.combo_model_type, "custom_acid_base_system", True)
        if hasattr(self, "combo_titrant_type"):
            self._sync_titrant_options(advanced)
        if advanced and self._advanced_tables_dirty:
            try:
                self._populate_advanced_tables_from_basic_model(self._basic_model_from_species_table())
            except Exception:
                pass
        self._sync_template_controls()
        self._apply_parameter_table_visibility()
        self._refresh_basic_matrix()
        if hasattr(self, "titration_group") and hasattr(self, "combo_data_type"):
            self.titration_group.setVisible(
                advanced and str(self.combo_data_type.currentData() or "") == "potentiometry"
            )
        if hasattr(self, "lbl_stoich_selection_help"):
            self.lbl_stoich_selection_help.setVisible(free_custom)
        self._on_titrant_type_changed()
        self._on_signal_type_changed()
        self._sync_plot_options_for_analysis()

    def _sync_titrant_options(self, advanced: bool) -> None:
        current = str(self.combo_titrant_type.currentData() or "base")
        self.combo_titrant_type.blockSignals(True)
        try:
            self.combo_titrant_type.clear()
            self.combo_titrant_type.addItem("strong base", "base")
            self.combo_titrant_type.addItem("strong acid", "acid")
            if advanced:
                self.combo_titrant_type.addItem("custom titrant", "custom")
            idx = self.combo_titrant_type.findData(current if advanced or current != "custom" else "base")
            self.combo_titrant_type.setCurrentIndex(max(0, idx))
        finally:
            self.combo_titrant_type.blockSignals(False)

    def _sync_template_controls(self) -> None:
        if not hasattr(self, "combo_model_type"):
            return
        template_id = str(self.combo_model_type.currentData() or "")
        needs_count = (not self._advanced_visible()) or template_id == "polyprotic_acid_base"
        if hasattr(self, "lbl_pka_count"):
            self.lbl_pka_count.setVisible(needs_count)
        if hasattr(self, "spin_pka_count"):
            self.spin_pka_count.setVisible(needs_count)
        if hasattr(self, "btn_generate_species_basic"):
            self.btn_generate_species_basic.setVisible(needs_count)

    def _sync_pkw_ui(self) -> None:
        if not hasattr(self, "pkw_group"):
            return
        advanced = self._advanced_visible()
        medium = str(self.combo_medium.currentData() or "aqueous") if hasattr(self, "combo_medium") else "aqueous"
        fit_pkw = bool(self.chk_fit_pkw.isChecked()) if hasattr(self, "chk_fit_pkw") else False
        is_pot = str(self.combo_data_type.currentData() or "potentiometry") == "potentiometry" if hasattr(self, "combo_data_type") else True
        show = is_pot or advanced
        show_fields = advanced or medium in {"mixed", "custom"} or fit_pkw
        self.pkw_group.setVisible(show)
        if hasattr(self, "spin_pkw"):
            self.spin_pkw.setVisible(show_fields)
            self.spin_pkw_min.setVisible(show_fields)
            self.spin_pkw_max.setVisible(show_fields)
            self.chk_fit_pkw.setVisible(show_fields)
            self.lbl_pkw_help.setVisible(show_fields and medium in {"mixed", "custom"})

    def _on_model_template_changed(self) -> None:
        if self._updating_tables:
            return
        self._sync_template_controls()
        template_id = str(self.combo_model_type.currentData() or "simple_monoprotic")
        if template_id == "coupled_future":
            QMessageBox.information(
                self,
                "Template unavailable",
                "The coupled acid-base / host-guest template is reserved for a future extension.",
            )
            return
        if template_id in {"multiple_components", "custom_acid_base_system"}:
            self._sync_advanced_ui(True)
        if template_id == "simple_monoprotic":
            self._generate_basic_species(1)
            return
        if template_id == "diprotic_ligand":
            self._generate_basic_species(2)
            return
        self._load_template(template_id)

    def _on_pka_count_changed(self) -> None:
        if self._updating_tables:
            return
        if not self._advanced_visible() or str(self.combo_model_type.currentData() or "") == "polyprotic_acid_base":
            self._generate_basic_species(int(self.spin_pka_count.value()))

    def _on_common_model_changed(self) -> None:
        if self._updating_tables or self._updating_basic_fields:
            return
        model = str(self.combo_common_model.currentData() or "custom")
        if model == "custom":
            return
        settings = {
            "monoprotic_acid": ("A", -1, 1),
            "monoprotic_base": ("B", 0, 1),
            "diprotic_acid": ("A", -2, 2),
            "diprotic_base": ("B", 0, 2),
        }.get(model)
        if settings is None:
            return
        name, charge, n_pka = settings
        was = self._updating_basic_fields
        self._updating_basic_fields = True
        try:
            self.edit_analyte_name.setText(name)
            self.spin_base_charge.setValue(charge)
            self.spin_pka_count.setValue(n_pka)
        finally:
            self._updating_basic_fields = was
        self._generate_basic_species(n_pka)

    def _default_species_name(self, h_count: int) -> str:
        if h_count <= 0:
            return str(self.edit_analyte_name.text() or "L").strip() or "L"
        return f"H{h_count}L" if h_count > 1 else "HL"

    def _insert_basic_species_row(
        self,
        *,
        name: str = "",
        h_count: int = 0,
        charge: int | None = None,
        include: bool = True,
        observable: bool = True,
        fixed: bool = False,
    ) -> None:
        row = self.table_basic_species.rowCount()
        self.table_basic_species.insertRow(row)
        h_count = int(h_count)
        if charge is None:
            charge = int(self.spin_base_charge.value()) + h_count
        _set_table_text(self.table_basic_species, row, 0, name or self._default_species_name(h_count))
        _set_table_text(self.table_basic_species, row, 1, h_count)
        _set_table_text(self.table_basic_species, row, 2, int(charge))
        for col, checked in ((3, include), (4, observable), (5, fixed)):
            chk = self._make_checkbox(bool(checked))
            chk.stateChanged.connect(self._on_basic_species_model_changed)
            self.table_basic_species.setCellWidget(row, col, chk)

    def _basic_species_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.table_basic_species.rowCount()):
            include_widget = self.table_basic_species.cellWidget(row, 3)
            observable_widget = self.table_basic_species.cellWidget(row, 4)
            fixed_widget = self.table_basic_species.cellWidget(row, 5)
            h_count = int(float(_table_text(self.table_basic_species, row, 1, "0")))
            rows.append(
                {
                    "name": _table_text(self.table_basic_species, row, 0, self._default_species_name(h_count)),
                    "h_count": h_count,
                    "charge": int(float(_table_text(self.table_basic_species, row, 2, str(int(self.spin_base_charge.value()) + h_count)))),
                    "include": True if not isinstance(include_widget, QCheckBox) else include_widget.isChecked(),
                    "observable": True if not isinstance(observable_widget, QCheckBox) else observable_widget.isChecked(),
                    "fixed": False if not isinstance(fixed_widget, QCheckBox) else fixed_widget.isChecked(),
                }
            )
        return rows

    def _generate_basic_species(self, n_pka: int) -> None:
        if self._updating_basic_fields:
            return
        self._updating_basic_fields = True
        self.table_basic_species.blockSignals(True)
        try:
            self.table_basic_species.setRowCount(0)
            base_charge = int(self.spin_base_charge.value())
            if hasattr(self, "spin_pka_count") and self.spin_pka_count.value() != int(n_pka):
                self.spin_pka_count.blockSignals(True)
                self.spin_pka_count.setValue(int(n_pka))
                self.spin_pka_count.blockSignals(False)
            for h_count in range(int(n_pka) + 1):
                self._insert_basic_species_row(
                    h_count=h_count,
                    charge=base_charge + h_count,
                    include=True,
                    observable=True,
                    fixed=(h_count == 0),
                )
        finally:
            self.table_basic_species.blockSignals(False)
            self._updating_basic_fields = False
        self._on_basic_species_model_changed()

    def _add_basic_species_row(self) -> None:
        rows = self._basic_species_rows()
        next_h = (max([int(row.get("h_count", 0)) for row in rows] or [0]) + 1)
        self._insert_basic_species_row(h_count=next_h)
        self._on_basic_species_model_changed()

    def _remove_selected_basic_species(self) -> None:
        rows = sorted({idx.row() for idx in self.table_basic_species.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table_basic_species.removeRow(row)
        if rows:
            self._on_basic_species_model_changed()

    def _on_base_charge_changed(self, value: int) -> None:
        if self._updating_basic_fields:
            return
        self.table_basic_species.blockSignals(True)
        try:
            for row in range(self.table_basic_species.rowCount()):
                h_count = int(float(_table_text(self.table_basic_species, row, 1, "0")))
                _set_table_text(self.table_basic_species, row, 2, int(value) + h_count)
        finally:
            self.table_basic_species.blockSignals(False)
        self._on_basic_species_model_changed()

    def _on_basic_species_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_basic_fields or self._updating_tables:
            return
        if item.column() == 1:
            try:
                h_count = int(float(item.text()))
            except Exception:
                item.setToolTip("h_count must be an integer greater than or equal to 0.")
                return
            if h_count < 0:
                item.setToolTip("h_count must be greater than or equal to 0.")
                return
            item.setText(str(h_count))
            _set_table_text(self.table_basic_species, item.row(), 2, int(self.spin_base_charge.value()) + h_count)
            if not _table_text(self.table_basic_species, item.row(), 0, ""):
                _set_table_text(self.table_basic_species, item.row(), 0, self._default_species_name(h_count))
        self._on_basic_species_model_changed()

    def _on_basic_species_model_changed(self) -> None:
        if self._updating_basic_fields or self._updating_tables:
            return
        try:
            model = self._basic_model_from_species_table()
        except Exception:
            self._refresh_basic_matrix()
            return
        self._advanced_tables_dirty = True
        if self._advanced_visible():
            self._populate_advanced_tables_from_basic_model(model)
        self._refresh_basic_matrix()
        self._refresh_parameter_table()

    def _basic_model_from_species_table(self) -> dict[str, Any]:
        analyte_name = str(self.edit_analyte_name.text() or "L").strip() or "L"
        rows = self._basic_species_rows()
        species_rows: list[dict[str, Any]] = []
        included_h = sorted({int(row["h_count"]) for row in rows if bool(row.get("include", True)) and int(row["h_count"]) > 0})
        pka_defaults = self._basic_pka_values()
        while len(pka_defaults) < len(included_h):
            pka_defaults.append(5.0 + 2.0 * len(pka_defaults))
        log_beta_by_h: dict[int, float] = {}
        cumulative = pka_to_log_beta(pka_defaults[: len(included_h)]) if included_h else []
        for h_count, log_beta in zip(included_h, cumulative):
            log_beta_by_h[int(h_count)] = float(log_beta)
        for row in rows:
            h_count = int(row["h_count"])
            species_rows.append(
                {
                    "name": str(row["name"] or self._default_species_name(h_count)),
                    "charge": int(row["charge"]),
                    "h_count": h_count,
                    "include": bool(row.get("include", True)),
                    "observable": bool(row.get("observable", True)),
                    "fixed": bool(row.get("fixed", False)) or h_count == 0,
                    "non_observable": not bool(row.get("observable", True)),
                    "parent_component": analyte_name,
                    "log_beta": 0.0 if h_count == 0 else float(log_beta_by_h.get(h_count, 0.0)),
                }
            )
        components = [
            {
                "name": analyte_name,
                "role": "analyte",
                "analytical_concentration": float(self.spin_analyte_conc.value()),
                "charge": int(self.spin_base_charge.value()),
                "is_proton": False,
                "is_titrant": False,
                "is_background": False,
                "fixed_concentration": False,
                "implicit": False,
            },
            {
                "name": proton_component_name(),
                "role": "proton",
                "analytical_concentration": None,
                "charge": 1,
                "is_proton": True,
                "is_titrant": False,
                "is_background": False,
                "fixed_concentration": True,
                "implicit": True,
            },
        ]
        return self._sanitize_internal_proton_model(
            {
                "template_id": str(self.combo_model_type.currentData() or "custom_acid_base_system"),
                "definition_mode": "matrix",
                "constant_mode": "pKa",
                "components": components,
                "species": species_rows,
                "stoichiometric_matrix": [
                    [1 for _row in species_rows],
                    [int(row.get("h_count", 0)) for row in species_rows],
                ],
                "equations_text": "",
            }
        )

    def _populate_advanced_tables_from_basic_model(self, model: dict[str, Any]) -> None:
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            self.components_table.setRowCount(0)
            for comp in list(model.get("components") or []):
                comp = dict(comp)
                if bool(comp.get("is_proton")) or str(comp.get("name") or "").strip().upper() == "H":
                    comp.update({"name": "H", "role": "proton", "charge": 1, "is_proton": True, "is_titrant": False})
                self._insert_component_row(comp)
            self.species_table.setRowCount(0)
            for sp in list(model.get("species") or []):
                self._insert_species_row(sp)
            self._sync_component_role_flags()
            self.spin_n_components_model.setValue(max(1, self.components_table.rowCount()))
            self.spin_n_species_model.setValue(max(1, self.species_table.rowCount()))
            self._sync_matrix_headers()
            self._set_stoich_matrix_rows(list(model.get("stoichiometric_matrix") or []))
            self._advanced_tables_dirty = False
            self._advanced_table_rebuild_count += 1
        finally:
            self._updating_tables = was_updating

    def _populate_basic_species_from_model(self, model: dict[str, Any]) -> None:
        blocks = acid_base_constant_blocks(model)
        block = blocks[0] if blocks else None
        analyte_name = str(block.get("component_name") if block else "L")
        species = list(block.get("species") if block else model.get("species") or [])
        analyte = next(
            (
                comp
                for comp in list(model.get("components") or [])
                if str(comp.get("name") or "") == analyte_name
            ),
            {},
        )
        was_updating = self._updating_basic_fields
        self._updating_basic_fields = True
        self.table_basic_species.blockSignals(True)
        try:
            self.edit_analyte_name.setText(analyte_name or "L")
            if analyte.get("charge") not in (None, ""):
                self.spin_base_charge.setValue(int(analyte.get("charge") or 0))
            self.table_basic_species.setRowCount(0)
            for sp in sorted(species, key=lambda row: int(row.get("h_count") or 0)):
                self._insert_basic_species_row(
                    name=str(sp.get("name") or ""),
                    h_count=int(sp.get("h_count") or 0),
                    charge=int(sp.get("charge") or 0),
                    include=bool(sp.get("include", True)),
                    observable=bool(sp.get("observable", True)),
                    fixed=bool(sp.get("fixed", False)),
                )
        finally:
            self.table_basic_species.blockSignals(False)
            self._updating_basic_fields = was_updating
        self._refresh_basic_matrix()

    def _refresh_basic_matrix(self) -> None:
        if not hasattr(self, "basic_matrix_table"):
            return
        rows = self._basic_species_rows() if hasattr(self, "table_basic_species") else []
        self.basic_matrix_table.setVisible(bool(rows) and bool(self.btn_show_matrix.isChecked()))
        self.basic_matrix_table.setRowCount(len(rows))
        self.basic_matrix_table.setColumnCount(2)
        self.basic_matrix_table.setHorizontalHeaderLabels(["L", "H"])
        self.basic_matrix_table.setVerticalHeaderLabels([str(row.get("name") or f"sp{idx + 1}") for idx, row in enumerate(rows)])
        for row_idx, row in enumerate(rows):
            _set_table_text(self.basic_matrix_table, row_idx, 0, 1)
            _set_table_text(self.basic_matrix_table, row_idx, 1, int(row.get("h_count", 0)))

    def _basic_pka_values(self) -> list[float]:
        values: list[float] = []
        for row in range(self.table_basic_pka.rowCount()):
            raw = _table_text(self.table_basic_pka, row, 1, "")
            if raw:
                values.append(float(raw))
        return values

    def _basic_pka_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.table_basic_pka.rowCount()):
            fit_widget = self.table_basic_pka.cellWidget(row, 4)
            rows.append(
                {
                    "parameter": _table_text(self.table_basic_pka, row, 0, f"pKa{row + 1}"),
                    "initial": float(_table_text(self.table_basic_pka, row, 1, "5.0")),
                    "min": float(_table_text(self.table_basic_pka, row, 2, "-5.0")),
                    "max": float(_table_text(self.table_basic_pka, row, 3, "25.0")),
                    "fit": True if not isinstance(fit_widget, QCheckBox) else fit_widget.isChecked(),
                }
            )
        return rows

    def _set_basic_pka_table_from_model(self, model: dict[str, Any]) -> None:
        if not hasattr(self, "table_basic_pka"):
            return
        blocks = acid_base_constant_blocks(model)
        pka_values = list(blocks[0].get("pka") or []) if blocks else []
        was_updating = self._updating_basic_fields
        self._updating_basic_fields = True
        self.table_basic_pka.blockSignals(True)
        try:
            previous = {
                _table_text(self.table_basic_pka, row, 0, f"pKa{row + 1}"): {
                    "min": _table_text(self.table_basic_pka, row, 2, "-5.0"),
                    "max": _table_text(self.table_basic_pka, row, 3, "25.0"),
                    "fit": (
                        True
                        if not isinstance(self.table_basic_pka.cellWidget(row, 4), QCheckBox)
                        else self.table_basic_pka.cellWidget(row, 4).isChecked()
                    ),
                }
                for row in range(self.table_basic_pka.rowCount())
            }
            self.table_basic_pka.setRowCount(len(pka_values))
            for idx, value in enumerate(pka_values, start=1):
                key = f"pKa{idx}"
                label = QTableWidgetItem(f"pKa{idx}")
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table_basic_pka.setItem(idx - 1, 0, label)
                _set_table_text(self.table_basic_pka, idx - 1, 1, f"{float(value):.6g}")
                _set_table_text(self.table_basic_pka, idx - 1, 2, previous.get(key, {}).get("min", "-5.0"))
                _set_table_text(self.table_basic_pka, idx - 1, 3, previous.get(key, {}).get("max", "25.0"))
                fit = self._make_checkbox(bool(previous.get(key, {}).get("fit", True)))
                fit.stateChanged.connect(self._on_basic_pka_fit_changed)
                self.table_basic_pka.setCellWidget(idx - 1, 4, fit)
        finally:
            self.table_basic_pka.blockSignals(False)
            self._updating_basic_fields = was_updating

    def _on_basic_pka_table_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_basic_fields or self._updating_tables or item.column() not in {1, 2, 3}:
            return
        try:
            value = float(item.text())
        except Exception:
            item.setToolTip("Use a numeric value.")
            return
        if item.column() == 1 and not -5.0 <= value <= 25.0:
            item.setToolTip("Recommended pKa guesses are between -5 and 25.")
        else:
            item.setToolTip("")
        self._sync_parameter_row_from_basic_pka(item.row())

    def _on_basic_pka_fit_changed(self) -> None:
        if self._updating_basic_fields or self._updating_tables:
            return
        for row in range(self.table_basic_pka.rowCount()):
            self._sync_parameter_row_from_basic_pka(row)
        self._refresh_parameter_table()

    def _sync_parameter_row_from_basic_pka(self, row: int) -> None:
        name = f"pKa{int(row) + 1}"
        self._set_parameter_value(name, _table_text(self.table_basic_pka, row, 1, "5.0"), update_spin=False)
        self._set_parameter_bounds(
            name,
            _table_text(self.table_basic_pka, row, 2, "-5.0"),
            _table_text(self.table_basic_pka, row, 3, "25.0"),
        )
        fit_widget = self.table_basic_pka.cellWidget(row, 4)
        fit = True if not isinstance(fit_widget, QCheckBox) else fit_widget.isChecked()
        self._set_parameter_fit(name, fit)

    def _set_first_analyte_concentration(self, value: float) -> None:
        if not hasattr(self, "components_table"):
            return
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            for row in range(self.components_table.rowCount()):
                is_proton_widget = self.components_table.cellWidget(row, 4)
                name = _table_text(self.components_table, row, 1, "")
                if bool(isinstance(is_proton_widget, QCheckBox) and is_proton_widget.isChecked()):
                    continue
                if str(name).strip().lower() == proton_component_name().lower():
                    continue
                _set_table_text(self.components_table, row, 2, f"{float(value):.8g}")
                break
        finally:
            self._updating_tables = was_updating

    def _sync_basic_fields_from_model(self) -> None:
        if self._updating_basic_fields or not hasattr(self, "spin_analyte_conc"):
            return
        try:
            model = self._current_model_from_ui()
            analyte = self._first_analyte_component(model)
        except Exception:
            return
        was_updating = self._updating_basic_fields
        self._updating_basic_fields = True
        try:
            conc = analyte.get("analytical_concentration")
            if conc not in (None, ""):
                self.spin_analyte_conc.blockSignals(True)
                self.spin_analyte_conc.setValue(float(conc))
                self.spin_analyte_conc.blockSignals(False)
            if analyte.get("name") not in (None, "") and hasattr(self, "edit_analyte_name"):
                self.edit_analyte_name.blockSignals(True)
                self.edit_analyte_name.setText(str(analyte.get("name") or "L"))
                self.edit_analyte_name.blockSignals(False)
            if analyte.get("charge") not in (None, "") and hasattr(self, "spin_base_charge"):
                self.spin_base_charge.blockSignals(True)
                self.spin_base_charge.setValue(int(analyte.get("charge") or 0))
                self.spin_base_charge.blockSignals(False)
            self._populate_basic_species_from_model(model)
            self._set_basic_pka_table_from_model(model)
        finally:
            self._updating_basic_fields = was_updating

    def _on_analyte_concentration_changed(self, value: float) -> None:
        if self._updating_basic_fields:
            return
        self._set_first_analyte_concentration(float(value))
        self._set_parameter_value("analyte concentration", f"{float(value):.8g}", update_spin=False)
        self._on_basic_species_model_changed()
        self._refresh_parameter_table()

    def _on_titrant_concentration_changed(self, value: float) -> None:
        if self._updating_basic_fields:
            return
        self._set_parameter_value("titrant concentration", f"{float(value):.8g}", update_spin=False)
        self._refresh_parameter_table()

    def _on_volume_offset_changed(self, value: float) -> None:
        if self._updating_basic_fields:
            return
        self._set_parameter_value("volume offset", f"{float(value):.8g}", update_spin=False)
        self._refresh_parameter_table()

    def _on_electrode_text_changed(self) -> None:
        if self._updating_basic_fields:
            return
        self._set_parameter_value("electrode_e0", self.edit_e0.text(), update_spin=False)
        self._set_parameter_value("electrode_slope", self.edit_slope.text(), update_spin=False)
        self._refresh_parameter_table()

    def _on_fit_electrode_basic_toggled(self, checked: bool) -> None:
        if hasattr(self, "chk_fit_electrode"):
            self.chk_fit_electrode.blockSignals(True)
            self.chk_fit_electrode.setChecked(bool(checked))
            self.chk_fit_electrode.blockSignals(False)
        if hasattr(self, "electrode_group"):
            is_emf = str(self.combo_signal_type.currentData() or "").lower() == "mv"
            self.electrode_group.setVisible(is_emf and (self._advanced_visible() or bool(checked)))
        self._refresh_parameter_table()

    def _on_fit_electrode_advanced_toggled(self, checked: bool) -> None:
        if hasattr(self, "chk_fit_electrode_basic"):
            self.chk_fit_electrode_basic.blockSignals(True)
            self.chk_fit_electrode_basic.setChecked(bool(checked))
            self.chk_fit_electrode_basic.blockSignals(False)

    def _on_parameter_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_tables or self._updating_basic_fields or item.column() not in {2, 3, 4}:
            return
        name = _table_text(self.parameters_table, item.row(), 0, "").strip().lower()
        value = str(item.text() or "").strip()
        was_updating = self._updating_basic_fields
        self._updating_basic_fields = True
        try:
            if name == "pkw" and item.column() in {3, 4}:
                if hasattr(self, "spin_pkw_min") and item.column() == 3 and value:
                    self.spin_pkw_min.setValue(float(value))
                if hasattr(self, "spin_pkw_max") and item.column() == 4 and value:
                    self.spin_pkw_max.setValue(float(value))
                return
            pka_match = re.search(r"pka[_\-\s]*(\d+)$", name)
            if pka_match is not None and hasattr(self, "table_basic_pka"):
                idx = int(pka_match.group(1)) - 1
                target_col = {2: 1, 3: 2, 4: 3}.get(int(item.column()))
                if target_col is not None and 0 <= idx < self.table_basic_pka.rowCount():
                    _set_table_text(self.table_basic_pka, idx, target_col, value)
                return
            if item.column() != 2:
                return
            if name == "analyte concentration" and value:
                numeric = float(value)
                self.spin_analyte_conc.setValue(numeric)
                self._set_first_analyte_concentration(numeric)
            elif name == "titrant concentration" and value:
                self.spin_titrant_conc.setValue(float(value))
            elif name == "volume offset" and value:
                self.spin_volume_offset.setValue(float(value))
            elif name == "pkw" and value:
                self.spin_pkw.setValue(float(value))
            elif name == "electrode_e0":
                self.edit_e0.setText(value)
            elif name == "electrode_slope":
                self.edit_slope.setText(value)
        except Exception:
            return
        finally:
            self._updating_basic_fields = was_updating

    def _apply_parameter_table_visibility(self) -> None:
        if not hasattr(self, "parameters_table"):
            return
        advanced = self._advanced_visible()
        self.parameters_table.setHorizontalHeaderItem(5, QTableWidgetItem("Fit"))
        for col in range(self.parameters_table.columnCount()):
            self.parameters_table.setColumnHidden(col, (not advanced) and col in {1, 6, 7})
        hidden_basic_names = {
            "analyte concentration",
            "titrant concentration",
            "volume offset",
            "electrode_e0",
            "electrode_slope",
            "pkw",
            "baseline",
        }
        for row in range(self.parameters_table.rowCount()):
            name = _table_text(self.parameters_table, row, 0, "").strip().lower()
            fit_widget = self.parameters_table.cellWidget(row, 5)
            is_fit = bool(isinstance(fit_widget, QCheckBox) and fit_widget.isChecked())
            show_optional = (
                (name == "analyte concentration" and is_fit)
                or (name == "titrant concentration" and is_fit)
                or (name == "volume offset" and is_fit)
                or (name == "pkw" and is_fit)
                or (name in {"electrode_e0", "electrode_slope"} and is_fit)
            )
            self.parameters_table.setRowHidden(row, (not advanced) and name in hidden_basic_names and not show_optional)

    def _on_signal_type_changed(self) -> None:
        is_emf = str(self.combo_signal_type.currentData() or "").lower() == "mv" if hasattr(self, "combo_signal_type") else False
        if hasattr(self, "chk_ideal_nernst"):
            self.chk_ideal_nernst.setVisible(is_emf)
        if not is_emf and hasattr(self, "chk_fit_electrode_basic"):
            self.chk_fit_electrode_basic.setChecked(False)
        if is_emf and hasattr(self, "chk_ideal_nernst") and self.chk_ideal_nernst.isChecked():
            self._on_ideal_nernst_toggled(True)
        if hasattr(self, "chk_fit_electrode_basic"):
            self.chk_fit_electrode_basic.setVisible(is_emf)
        if hasattr(self, "electrode_group"):
            self.electrode_group.setVisible(is_emf and (self._advanced_visible() or self.chk_fit_electrode_basic.isChecked()))
        self._refresh_potentiometry_preview()
        self._sync_plot_options_for_analysis()

    def _on_ideal_nernst_toggled(self, checked: bool) -> None:
        if not checked:
            return
        if hasattr(self, "edit_e0"):
            self.edit_e0.setText("0.0")
        if hasattr(self, "edit_slope"):
            self.edit_slope.setText("-59.16")
        if hasattr(self, "chk_fit_electrode"):
            self.chk_fit_electrode.setChecked(False)

    def _sync_model_mode_ui(self) -> None:
        if not hasattr(self, "model_definition_stack"):
            return
        self.model_definition_stack.setCurrentIndex(0 if self.radio_model_matrix.isChecked() else 1)

    def _on_load_template_clicked(self) -> None:
        template_id = str(self.combo_model_type.currentData() or "simple_monoprotic")
        if template_id == "coupled_future":
            QMessageBox.information(
                self,
                "Template unavailable",
                "The coupled acid-base / host-guest template is reserved for a future extension.",
            )
            return
        self._load_template(template_id)

    def _load_template(self, template_id: str) -> None:
        if str(template_id) == "custom_acid_base_system":
            self._populate_empty_custom_model()
            return
        concentration = float(self.spin_analyte_conc.value()) if hasattr(self, "spin_analyte_conc") else 1.0e-3
        n_pka = int(self.spin_pka_count.value()) if hasattr(self, "spin_pka_count") else None
        pka_values = self._basic_pka_values() if hasattr(self, "table_basic_pka") else []
        pka = pka_values if str(template_id) == "polyprotic_acid_base" and pka_values else None
        model = build_acid_base_template(
            template_id,
            analytical_concentration=concentration,
            n_pka=n_pka,
            pka=pka,
        )
        self._populate_model_from_definition(model, update_template=True)

    def _populate_empty_custom_model(self) -> None:
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            idx = self.combo_model_type.findData("custom_acid_base_system")
            if idx >= 0:
                self.combo_model_type.setCurrentIndex(idx)
            mode_idx = self.combo_constant_mode.findData("pKa")
            if mode_idx >= 0:
                self.combo_constant_mode.setCurrentIndex(mode_idx)
            self.radio_model_matrix.setChecked(True)
            self.radio_model_equations.setChecked(False)
            if hasattr(self, "equation_editor"):
                self.equation_editor.clear()
            self.spin_n_components_model.setValue(0)
            self.spin_n_species_model.setValue(0)
            self.components_table.setRowCount(0)
            self.species_table.setRowCount(0)
            self.stoich_table.setRowCount(0)
            self.stoich_table.setColumnCount(0)
            self.parameters_table.setRowCount(0)
            self._advanced_tables_dirty = False
        finally:
            self._updating_tables = was_updating
        self._sync_advanced_ui(True)
        self._refresh_basic_matrix()
        self._refresh_parameter_table()

    def _define_model_dimensions(self) -> None:
        n_components = int(self.spin_n_components_model.value())
        n_species = int(self.spin_n_species_model.value())
        if n_components <= 0 or n_species <= 0:
            QMessageBox.warning(self, "Invalid model", "Enter Number of Components and Species (>0).")
            return
        custom_idx = self.combo_model_type.findData("custom_acid_base_system")
        if custom_idx >= 0 and self.combo_model_type.currentIndex() != custom_idx:
            self.combo_model_type.blockSignals(True)
            self.combo_model_type.setCurrentIndex(custom_idx)
            self.combo_model_type.blockSignals(False)
        model = {
            "template_id": "custom_acid_base_system",
            "definition_mode": "matrix",
            "constant_mode": str(self.combo_constant_mode.currentData() or "pKa"),
            "components": [],
            "species": [],
            "stoichiometric_matrix": [[0 for _ in range(n_species)] for _ in range(n_components)],
            "equations_text": str(self.equation_editor.toPlainText() if hasattr(self, "equation_editor") else ""),
        }
        for idx in range(n_components):
            model["components"].append(
                {
                    "name": f"C{idx + 1}",
                    "role": "",
                    "analytical_concentration": "",
                    "charge": 0,
                    "is_proton": False,
                    "is_titrant": False,
                    "is_background": False,
                    "fixed_concentration": False,
                    "implicit": False,
                }
            )
        for idx in range(n_species):
            model["species"].append(
                {
                    "name": f"S{idx + 1}",
                    "charge": 0,
                    "h_count": 0,
                    "include": True,
                    "observable": True,
                    "fixed": False,
                    "non_observable": False,
                    "parent_component": "",
                    "log_beta": 0.0,
                }
            )
        self._populate_model_from_definition(model, update_template=False)
        self._sync_advanced_ui(True)

    def _insert_component_row(self, row_data: dict[str, Any] | None = None) -> None:
        row_data = dict(row_data or {})
        row = self.components_table.rowCount()
        self.components_table.insertRow(row)
        role_value = str(row_data.get("role") or "")
        if not role_value and bool(row_data.get("is_proton", False)):
            role_value = "proton"
        elif not role_value and bool(row_data.get("is_titrant", False)):
            role_value = "titrant"
        elif not role_value and bool(row_data.get("is_background", False)):
            role_value = "background"
        role = self._make_combo(
            ["", "analyte", "proton", "titrant", "background", "spectator", "imposed pH"],
            role_value,
        )
        role.currentIndexChanged.connect(self._on_component_table_changed)
        self.components_table.setCellWidget(row, 0, role)
        _set_table_text(self.components_table, row, 1, row_data.get("name", ""))
        _set_table_text(self.components_table, row, 2, "" if row_data.get("analytical_concentration") in (None, "") else row_data.get("analytical_concentration"))
        _set_table_text(self.components_table, row, 3, row_data.get("charge", 0))
        is_proton = self._make_checkbox(bool(row_data.get("is_proton", False)))
        is_proton.stateChanged.connect(self._on_component_table_changed)
        self.components_table.setCellWidget(row, 4, is_proton)
        is_titrant = self._make_checkbox(bool(row_data.get("is_titrant", False)))
        is_titrant.stateChanged.connect(self._on_component_table_changed)
        self.components_table.setCellWidget(row, 5, is_titrant)
        is_background = self._make_checkbox(bool(row_data.get("is_background", False)))
        is_background.stateChanged.connect(self._on_component_table_changed)
        self.components_table.setCellWidget(row, 6, is_background)
        fixed = self._make_checkbox(bool(row_data.get("fixed_concentration", False)))
        fixed.stateChanged.connect(self._on_component_table_changed)
        self.components_table.setCellWidget(row, 7, fixed)

    def _remove_selected_component(self) -> None:
        rows = sorted({idx.row() for idx in self.components_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.components_table.removeRow(row)
        if rows:
            self.spin_n_components_model.setValue(max(1, self.components_table.rowCount()))
            self._sync_matrix_headers()
            self._refresh_parameter_table()

    def _insert_species_row(self, row_data: dict[str, Any] | None = None) -> None:
        row_data = dict(row_data or {})
        row = self.species_table.rowCount()
        self.species_table.insertRow(row)
        _set_table_text(self.species_table, row, 0, row_data.get("name", f"S{row + 1}"))
        _set_table_text(self.species_table, row, 1, row_data.get("charge", 0))
        _set_table_text(self.species_table, row, 2, row_data.get("h_count", 0))
        include = self._make_checkbox(bool(row_data.get("include", True)))
        include.stateChanged.connect(self._on_species_table_changed)
        self.species_table.setCellWidget(row, 3, include)
        observable = self._make_checkbox(bool(row_data.get("observable", True)))
        observable.stateChanged.connect(self._on_species_table_changed)
        self.species_table.setCellWidget(row, 4, observable)
        fixed = self._make_checkbox(bool(row_data.get("fixed", False)))
        fixed.stateChanged.connect(self._on_species_table_changed)
        self.species_table.setCellWidget(row, 5, fixed)
        non_observable = self._make_checkbox(bool(row_data.get("non_observable", False)))
        non_observable.stateChanged.connect(self._on_species_table_changed)
        self.species_table.setCellWidget(row, 6, non_observable)
        _set_table_text(self.species_table, row, 7, row_data.get("parent_component", "L"))

    def _remove_selected_species(self) -> None:
        rows = sorted({idx.row() for idx in self.species_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.species_table.removeRow(row)
        if rows:
            self.spin_n_species_model.setValue(max(1, self.species_table.rowCount()))
            self._sync_matrix_headers()
            self._refresh_parameter_table()

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
        self.titration_group.setVisible(is_pot and self._advanced_visible())
        if hasattr(self, "_data_import_stack"):
            self._data_import_stack.setCurrentIndex(0 if is_pot else 1)
        self._preview_selected_sheet()
        self._refresh_parameter_table()
        self._sync_plot_options_for_analysis()
        self._sync_pkw_ui()

    def _on_titrant_type_changed(self) -> None:
        advanced = self._advanced_visible()
        self._set_combo_item_enabled(self.combo_titrant_type, "custom", advanced)
        if not advanced and str(self.combo_titrant_type.currentData() or "") == "custom":
            idx = self.combo_titrant_type.findData("base")
            if idx >= 0:
                self.combo_titrant_type.setCurrentIndex(idx)
        custom = str(self.combo_titrant_type.currentData() or "") == "custom" and advanced
        self.custom_titrant_table.setEnabled(custom)
        self.btn_add_titrant_row.setEnabled(custom)
        self.btn_remove_titrant_row.setEnabled(custom)

    def _on_component_table_changed(self) -> None:
        if self._updating_tables:
            return
        self._sync_component_role_flags()
        self._sync_matrix_headers()
        self._sync_basic_fields_from_model()
        self._refresh_parameter_table()

    def _on_species_table_changed(self) -> None:
        if self._updating_tables:
            return
        self._sync_matrix_headers()
        self._sync_basic_fields_from_model()
        self._refresh_parameter_table()

    def _on_stoich_table_changed(self) -> None:
        if self._updating_tables:
            return
        self._sync_basic_fields_from_model()
        self._refresh_parameter_table()

    def _on_stoich_selection_changed(self) -> None:
        if self._updating_tables:
            return
        selected = set(self._selected_stoich_species_rows())
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            for row in range(self.species_table.rowCount()):
                observable_widget = self.species_table.cellWidget(row, 4)
                non_observable_widget = self.species_table.cellWidget(row, 6)
                non_observable = row in selected
                if isinstance(observable_widget, QCheckBox):
                    observable_widget.setChecked(not non_observable)
                if isinstance(non_observable_widget, QCheckBox):
                    non_observable_widget.setChecked(non_observable)
        finally:
            self._updating_tables = was_updating
        self._refresh_parameter_table()

    def _sync_matrix_headers(self) -> None:
        if not hasattr(self, "stoich_table"):
            return
        component_names = self._component_names_for_matrix_headers()
        species_names = self._species_names_for_matrix_headers()
        previous = self._stoich_matrix_rows()
        self.stoich_table.blockSignals(True)
        self.stoich_table.setRowCount(len(species_names))
        self.stoich_table.setColumnCount(len(component_names))
        self.stoich_table.setVerticalHeaderLabels(species_names)
        self.stoich_table.setHorizontalHeaderLabels(component_names)
        for row in range(self.stoich_table.rowCount()):
            for col in range(self.stoich_table.columnCount()):
                if self.stoich_table.item(row, col) is None:
                    _set_table_text(self.stoich_table, row, col, "0")
        self.stoich_table.blockSignals(False)
        self._set_stoich_matrix_rows(previous)
        self._sync_stoich_selection_from_species()

    def _component_names_for_matrix_headers(self) -> list[str]:
        if hasattr(self, "components_table") and self.components_table.rowCount() > 0:
            return [
                _table_text(self.components_table, row, 1, f"C{row + 1}") or f"C{row + 1}"
                for row in range(self.components_table.rowCount())
            ]
        if hasattr(self, "stoich_table"):
            return [
                str(self.stoich_table.horizontalHeaderItem(col).text() if self.stoich_table.horizontalHeaderItem(col) else f"C{col + 1}")
                for col in range(self.stoich_table.columnCount())
            ]
        return []

    def _species_names_for_matrix_headers(self) -> list[str]:
        if hasattr(self, "species_table") and self.species_table.rowCount() > 0:
            return [
                _table_text(self.species_table, row, 0, f"S{row + 1}") or f"S{row + 1}"
                for row in range(self.species_table.rowCount())
            ]
        if hasattr(self, "stoich_table"):
            return [
                str(self.stoich_table.verticalHeaderItem(row).text() if self.stoich_table.verticalHeaderItem(row) else f"S{row + 1}")
                for row in range(self.stoich_table.rowCount())
            ]
        return []

    def _sync_component_role_flags(self) -> None:
        if not hasattr(self, "components_table"):
            return
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            for row in range(self.components_table.rowCount()):
                role_widget = self.components_table.cellWidget(row, 0)
                if not isinstance(role_widget, QComboBox):
                    continue
                role = str(role_widget.currentText() or "").strip().lower()
                is_proton_widget = self.components_table.cellWidget(row, 4)
                is_titrant_widget = self.components_table.cellWidget(row, 5)
                is_background_widget = self.components_table.cellWidget(row, 6)
                if role == "proton":
                    if isinstance(is_proton_widget, QCheckBox):
                        is_proton_widget.setChecked(True)
                    if isinstance(is_titrant_widget, QCheckBox):
                        is_titrant_widget.setChecked(False)
                elif role == "titrant":
                    if isinstance(is_proton_widget, QCheckBox):
                        is_proton_widget.setChecked(False)
                    if isinstance(is_titrant_widget, QCheckBox):
                        is_titrant_widget.setChecked(True)
                elif role in {"background", "spectator"}:
                    if isinstance(is_background_widget, QCheckBox):
                        is_background_widget.setChecked(True)
                elif role == "analyte":
                    if isinstance(is_proton_widget, QCheckBox):
                        is_proton_widget.setChecked(False)
                    if isinstance(is_titrant_widget, QCheckBox):
                        is_titrant_widget.setChecked(False)
                    if isinstance(is_background_widget, QCheckBox):
                        is_background_widget.setChecked(False)
        finally:
            self._updating_tables = was_updating

    def _component_rows(self) -> list[dict[str, Any]]:
        if self._free_custom_model_selected() and hasattr(self, "stoich_table"):
            return [
                {
                    "name": str(self.stoich_table.horizontalHeaderItem(col).text() if self.stoich_table.horizontalHeaderItem(col) else f"C{col + 1}"),
                    "role": "",
                    "analytical_concentration": None,
                    "charge": 0,
                    "is_proton": False,
                    "is_titrant": False,
                    "is_background": False,
                    "fixed_concentration": False,
                    "implicit": False,
                }
                for col in range(self.stoich_table.columnCount())
            ]
        rows: list[dict[str, Any]] = []
        for row in range(self.components_table.rowCount()):
            role_widget = self.components_table.cellWidget(row, 0)
            role = role_widget.currentText() if isinstance(role_widget, QComboBox) else "analyte"
            is_proton_widget = self.components_table.cellWidget(row, 4)
            is_titrant_widget = self.components_table.cellWidget(row, 5)
            is_background_widget = self.components_table.cellWidget(row, 6)
            fixed_widget = self.components_table.cellWidget(row, 7)
            name = _table_text(self.components_table, row, 1, f"L{row + 1}")
            role_key = str(role or "").strip().lower()
            is_proton = (
                bool(isinstance(is_proton_widget, QCheckBox) and is_proton_widget.isChecked())
                or role_key == "proton"
            )
            is_titrant = (
                bool(isinstance(is_titrant_widget, QCheckBox) and is_titrant_widget.isChecked())
                or role_key == "titrant"
            )
            is_background = (
                bool(isinstance(is_background_widget, QCheckBox) and is_background_widget.isChecked())
                or role_key in {"background", "spectator"}
            )
            output_role = "proton" if is_proton else ("titrant" if is_titrant else (role_key or ""))
            rows.append(
                {
                    "name": name,
                    "role": output_role,
                    "analytical_concentration": _optional_float(_table_text(self.components_table, row, 2, "")),
                    "charge": int(float(_table_text(self.components_table, row, 3, "1" if is_proton else "0"))),
                    "is_proton": is_proton,
                    "is_titrant": is_titrant,
                    "is_background": is_background,
                    "fixed_concentration": bool(isinstance(fixed_widget, QCheckBox) and fixed_widget.isChecked()),
                    "implicit": bool(is_proton),
                }
            )
        return rows

    def _species_rows(self) -> list[dict[str, Any]]:
        if self._free_custom_model_selected() and hasattr(self, "stoich_table"):
            selected_non_observable = set(self._selected_stoich_species_rows())
            rows: list[dict[str, Any]] = []
            for row in range(self.stoich_table.rowCount()):
                header = self.stoich_table.verticalHeaderItem(row)
                name = str(header.text() if header is not None else f"sp{row + 1}")
                non_observable = row in selected_non_observable
                rows.append(
                    {
                        "name": name,
                        "charge": 0,
                        "h_count": 0,
                        "include": True,
                        "observable": not non_observable,
                        "fixed": False,
                        "non_observable": non_observable,
                        "parent_component": "",
                    }
                )
            return rows
        rows: list[dict[str, Any]] = []
        selected_non_observable = set(self._selected_stoich_species_rows())
        for row in range(self.species_table.rowCount()):
            include_widget = self.species_table.cellWidget(row, 3)
            observable_widget = self.species_table.cellWidget(row, 4)
            fixed_widget = self.species_table.cellWidget(row, 5)
            non_observable_widget = self.species_table.cellWidget(row, 6)
            non_observable = False if not isinstance(non_observable_widget, QCheckBox) else non_observable_widget.isChecked()
            observable = True if not isinstance(observable_widget, QCheckBox) else observable_widget.isChecked()
            if row in selected_non_observable:
                non_observable = True
                observable = False
            rows.append(
                {
                    "name": _table_text(self.species_table, row, 0, ""),
                    "charge": _optional_float(_table_text(self.species_table, row, 1, "")),
                    "h_count": int(float(_table_text(self.species_table, row, 2, "0"))),
                    "include": True if not isinstance(include_widget, QCheckBox) else include_widget.isChecked(),
                    "observable": observable,
                    "fixed": False if not isinstance(fixed_widget, QCheckBox) else fixed_widget.isChecked(),
                    "non_observable": non_observable,
                    "parent_component": _table_text(self.species_table, row, 7, ""),
                }
            )
        return rows

    def _stoich_matrix_rows(self) -> list[list[int]]:
        n_components = self.stoich_table.columnCount()
        n_species = self.stoich_table.rowCount()
        matrix: list[list[int]] = [[0 for _ in range(n_species)] for _ in range(n_components)]
        for species_row in range(n_species):
            for comp_col in range(n_components):
                matrix[comp_col][species_row] = int(float(_table_text(self.stoich_table, species_row, comp_col, "0")))
        return matrix

    def _set_stoich_matrix_rows(self, matrix: list[list[Any]]) -> None:
        was_updating = self._updating_tables
        self._updating_tables = True
        self.stoich_table.blockSignals(True)
        try:
            for row in range(self.stoich_table.rowCount()):
                for col in range(self.stoich_table.columnCount()):
                    value = matrix[col][row] if col < len(matrix) and row < len(list(matrix[col] or [])) else 0
                    _set_table_text(self.stoich_table, row, col, value)
        finally:
            self.stoich_table.blockSignals(False)
            self._updating_tables = was_updating
        self._sync_stoich_selection_from_species()

    def _selected_stoich_species_rows(self) -> list[int]:
        selection = self.stoich_table.selectionModel() if hasattr(self, "stoich_table") else None
        if selection is None:
            return []
        return sorted({idx.row() for idx in selection.selectedRows()})

    def _sync_stoich_selection_from_species(self) -> None:
        if not hasattr(self, "stoich_table"):
            return
        selection = self.stoich_table.selectionModel()
        if selection is None:
            return
        if self.species_table.rowCount() == 0:
            return
        wanted: set[int] = set()
        for row in range(self.species_table.rowCount()):
            observable_widget = self.species_table.cellWidget(row, 4)
            non_observable_widget = self.species_table.cellWidget(row, 6)
            non_observable = bool(isinstance(non_observable_widget, QCheckBox) and non_observable_widget.isChecked())
            observable = True if not isinstance(observable_widget, QCheckBox) else observable_widget.isChecked()
            if non_observable or not observable:
                wanted.add(row)
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            selection.clearSelection()
            for row in sorted(wanted):
                if 0 <= row < self.stoich_table.rowCount():
                    index = self.stoich_table.model().index(row, 0)
                    selection.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        finally:
            self._updating_tables = was_updating

    def _generate_species_from_matrix(self) -> None:
        components = self._component_rows()
        matrix = self._stoich_matrix_rows()
        proton_row = next((idx for idx, comp in enumerate(components) if bool(comp.get("is_proton"))), None)
        non_proton_names = [str(comp.get("name") or "") for comp in components if not bool(comp.get("is_proton"))]
        n_species = max((len(row) for row in matrix), default=0)
        species_rows: list[dict[str, Any]] = []
        for col in range(n_species):
            parent = ""
            for row_idx, comp in enumerate(components):
                if bool(comp.get("is_proton")):
                    continue
                coef = matrix[row_idx][col] if row_idx < len(matrix) and col < len(matrix[row_idx]) else 0
                if coef > 0:
                    parent = str(comp.get("name") or "")
                    break
            if not parent and non_proton_names:
                parent = non_proton_names[0]
            h_count = 0 if proton_row is None else int(matrix[proton_row][col] if col < len(matrix[proton_row]) else 0)
            existing_name = _table_text(self.species_table, col, 0, "") if col < self.species_table.rowCount() else ""
            name = existing_name or (parent if h_count == 0 else (f"H{parent}" if h_count == 1 else f"H{h_count}{parent}"))
            base_charge = next((int(comp.get("charge") or 0) for comp in components if str(comp.get("name") or "") == parent), 0)
            species_rows.append(
                {
                    "name": name,
                    "charge": base_charge + h_count,
                    "h_count": h_count,
                    "include": True,
                    "observable": True,
                    "fixed": h_count == 0,
                    "non_observable": False,
                    "parent_component": parent,
                }
            )
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            self.species_table.setRowCount(0)
            for row_data in species_rows:
                self._insert_species_row(row_data)
        finally:
            self._updating_tables = was_updating
        self._sync_matrix_headers()
        self._refresh_parameter_table()

    def _apply_equations_model(self) -> None:
        try:
            concentration = _optional_float(self._parameter_value("analyte concentration", default="0.001"))
            model = acid_base_model_from_equations(
                self.equation_editor.toPlainText(),
                analytical_concentration=float(concentration or 1.0e-3),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Equation editor", str(exc))
            return
        self._populate_model_from_definition(model, update_equations_text=False)
        self.log.append_text("Acid-base equations parsed into matrix/species tables.")

    def _populate_model_from_definition(
        self,
        model_def: dict[str, Any],
        *,
        update_template: bool = False,
        update_equations_text: bool = True,
    ) -> None:
        auto_add_proton = not self._model_definition_is_custom(model_def)
        model = self._sanitize_internal_proton_model(
            canonicalize_acid_base_model(
                model_def,
                auto_add_proton=auto_add_proton,
                infer_proton_from_name=auto_add_proton,
            ),
            auto_add_proton=auto_add_proton,
        )
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            if update_template and hasattr(self, "parameters_table"):
                self.parameters_table.setRowCount(0)
            if update_template:
                idx = self.combo_model_type.findData(str(model.get("template_id") or "custom_acid_base_system"))
                if idx >= 0:
                    self.combo_model_type.setCurrentIndex(idx)
            mode_idx = self.combo_constant_mode.findData(str(model.get("constant_mode") or "pKa"))
            if mode_idx >= 0:
                self.combo_constant_mode.setCurrentIndex(mode_idx)
            if update_equations_text and hasattr(self, "equation_editor"):
                self.equation_editor.setPlainText(str(model.get("equations_text") or ""))
            definition_mode = str(model.get("definition_mode") or "matrix")
            self.radio_model_matrix.setChecked(definition_mode != "equations")
            self.radio_model_equations.setChecked(definition_mode == "equations")
            if self._advanced_visible():
                self.components_table.setRowCount(0)
                for comp in list(model.get("components") or []):
                    self._insert_component_row(comp)
                self.species_table.setRowCount(0)
                for sp in list(model.get("species") or []):
                    self._insert_species_row(sp)
                self.spin_n_components_model.setValue(max(1, self.components_table.rowCount()))
                self.spin_n_species_model.setValue(max(1, self.species_table.rowCount()))
                self._sync_matrix_headers()
                self._set_stoich_matrix_rows(list(model.get("stoichiometric_matrix") or []))
                self._advanced_tables_dirty = False
            else:
                self._advanced_tables_dirty = True
        finally:
            self._updating_tables = was_updating
        self._sync_model_mode_ui()
        self._populate_basic_species_from_model(model)
        self._set_basic_pka_table_from_model(model)
        self._refresh_parameter_table()

    def _current_model_from_ui(self) -> dict[str, Any]:
        if not self._advanced_visible() and hasattr(self, "table_basic_species"):
            return self._basic_model_from_species_table()
        equations_text = str(self.equation_editor.toPlainText() or "")
        parsed_equations_model: dict[str, Any] | None = None
        if self.radio_model_equations.isChecked():
            parsed_equations_model = acid_base_model_from_equations(
                equations_text,
                analytical_concentration=float(_optional_float(self._parameter_value("analyte concentration", default="0.001")) or 1.0e-3),
            )
        model = {
            "template_id": str(self.combo_model_type.currentData() or "custom_acid_base_system"),
            "definition_mode": "equations" if self.radio_model_equations.isChecked() else "matrix",
            "constant_mode": normalize_constant_mode(self.combo_constant_mode.currentData()),
            "equations_text": equations_text,
            "components": self._component_rows(),
            "species": self._species_rows(),
            "stoichiometric_matrix": self._stoich_matrix_rows(),
        }
        if parsed_equations_model is not None and (not model["components"] or not model["species"]):
            model = parsed_equations_model
        return self._sanitize_internal_proton_model(
            canonicalize_acid_base_model(
                model,
                auto_add_proton=not self._free_custom_model_selected(),
                infer_proton_from_name=not self._free_custom_model_selected(),
            )
        )

    def _sanitize_internal_proton_model(self, model: dict[str, Any], *, auto_add_proton: bool | None = None) -> dict[str, Any]:
        if auto_add_proton is None:
            auto_add_proton = not self._free_custom_model_selected()
        sanitized = canonicalize_acid_base_model(
            model,
            auto_add_proton=auto_add_proton,
            infer_proton_from_name=auto_add_proton,
        )
        has_h = False
        for comp in sanitized["components"]:
            if bool(comp.get("is_proton")) or (
                auto_add_proton
                and str(comp.get("name") or "").strip().lower() == proton_component_name().lower()
            ):
                comp["name"] = proton_component_name()
                comp["role"] = "proton"
                comp["analytical_concentration"] = None
                comp["charge"] = 1
                comp["is_proton"] = True
                comp["is_titrant"] = False
                comp["is_background"] = False
                comp["fixed_concentration"] = True
                comp["implicit"] = True
                has_h = True
        if auto_add_proton and not has_h:
            sanitized["components"].append(
                {
                    "name": proton_component_name(),
                    "role": "proton",
                    "analytical_concentration": None,
                    "charge": 1,
                    "is_proton": True,
                    "is_titrant": False,
                    "is_background": False,
                    "fixed_concentration": True,
                    "implicit": True,
                }
            )
        component_names = [str(comp["name"]) for comp in sanitized["components"]]
        species = list(sanitized.get("species") or [])
        sanitized["component_names"] = component_names
        if auto_add_proton:
            sanitized["stoichiometric_matrix"] = [
                [
                    (int(sp.get("h_count") or 0) if comp_name == proton_component_name() else (1 if str(sp.get("parent_component") or "") == comp_name else 0))
                    for sp in species
                ]
                for comp_name in component_names
            ]
        return canonicalize_acid_base_model(
            sanitized,
            auto_add_proton=auto_add_proton,
            infer_proton_from_name=auto_add_proton,
        )

    def _apply_parameter_table_to_model(self, model: dict[str, Any]) -> dict[str, Any]:
        auto_add_proton = not self._free_custom_model_selected()
        updated = canonicalize_acid_base_model(
            model,
            auto_add_proton=auto_add_proton,
            infer_proton_from_name=auto_add_proton,
        )
        constant_mode = normalize_constant_mode(updated.get("constant_mode"))
        blocks = acid_base_constant_blocks(updated)
        component_count = len(blocks)
        for block in blocks:
            prefix = "" if component_count <= 1 else f"{block['component_name']}_"
            values: list[float] = []
            h_counts: list[int] = []
            parameter_prefix = "pKa" if constant_mode == "pKa" else "log_beta"
            protonated_species = [sp for sp in block["species"] if int(sp.get("h_count") or 0) > 0]
            for idx, species in enumerate(protonated_species, start=1):
                raw = self._parameter_value(f"{prefix}{parameter_prefix}{idx}", default=None)
                if raw in (None, ""):
                    continue
                values.append(float(raw))
                h_counts.append(int(species.get("h_count") or idx))
            if values:
                log_beta_values = pka_to_log_beta(values) if constant_mode == "pKa" else values
                log_by_h = {int(h_count): float(value) for h_count, value in zip(h_counts, log_beta_values)}
                for sp in updated["species"]:
                    if str(sp.get("parent_component") or "") != block["component_name"]:
                        continue
                    h_count = int(sp.get("h_count") or 0)
                    if h_count == 0:
                        sp["log_beta"] = 0.0
                    elif h_count in log_by_h:
                        sp["log_beta"] = float(log_by_h[h_count])
        blocks_after = acid_base_constant_blocks(updated)
        if blocks_after:
            updated["log_beta"] = [float(v) for v in blocks_after[0]["log_beta"]]
            updated["pka"] = [float(v) for v in blocks_after[0]["pka"]]
        return updated

    def _refresh_parameter_table(self) -> None:
        if self._updating_tables:
            return
        previous_values: dict[str, str] = {}
        previous_min: dict[str, str] = {}
        previous_max: dict[str, str] = {}
        previous_fit: dict[str, bool] = {}
        for row in range(self.parameters_table.rowCount()):
            name = _table_text(self.parameters_table, row, 0, "")
            if not name:
                continue
            previous_values[name] = _table_text(self.parameters_table, row, 2, "")
            previous_min[name] = _table_text(self.parameters_table, row, 3, "")
            previous_max[name] = _table_text(self.parameters_table, row, 4, "")
            fit_widget = self.parameters_table.cellWidget(row, 5)
            previous_fit[name] = bool(isinstance(fit_widget, QCheckBox) and fit_widget.isChecked())

        try:
            model = self._current_model_from_ui()
            constant_rows = parameter_rows_from_model(model)
        except Exception:
            return

        blocks = acid_base_constant_blocks(model)
        if self._free_custom_model_selected() and not blocks:
            was_updating = self._updating_tables
            self._updating_tables = True
            try:
                self.parameters_table.setRowCount(0)
                if hasattr(self, "table_basic_pka"):
                    self.table_basic_pka.setRowCount(0)
            finally:
                self._updating_tables = was_updating
            self._apply_parameter_table_visibility()
            return
        first_block = blocks[0] if blocks else {"component_name": "L"}
        rows = list(constant_rows)
        rows.extend(
            [
                {
                    "parameter": "analyte concentration",
                    "type": "local",
                    "initial_value": next(
                        (
                            comp.get("analytical_concentration")
                            for comp in list(model.get("components") or [])
                            if str(comp.get("name") or "") == str(first_block.get("component_name") or "")
                        ),
                        1.0e-3,
                    ),
                    "min": 0.0,
                    "max": "",
                    "fixed": True,
                    "linked_species": str(first_block.get("component_name") or ""),
                    "description": "Analytical concentration for the primary analyte.",
                },
                {
                    "parameter": "titrant concentration",
                    "type": "local",
                    "initial_value": float(self.spin_titrant_conc.value()) if hasattr(self, "spin_titrant_conc") else 1.0e-3,
                    "min": 0.0,
                    "max": "",
                    "fixed": True,
                    "linked_species": "",
                    "description": "Nominal titrant concentration.",
                },
                {
                    "parameter": "volume offset",
                    "type": "local",
                    "initial_value": float(self.spin_volume_offset.value()) if hasattr(self, "spin_volume_offset") else 0.0,
                    "min": "",
                    "max": "",
                    "fixed": True,
                    "linked_species": "",
                    "description": "Blank correction / effective volume offset.",
                },
                {
                    "parameter": "electrode_e0",
                    "type": "local",
                    "initial_value": _optional_float(self.edit_e0.text()) if hasattr(self, "edit_e0") else "",
                    "min": "",
                    "max": "",
                    "fixed": True,
                    "linked_species": "",
                    "description": "Electrode intercept, mV.",
                },
                {
                    "parameter": "electrode_slope",
                    "type": "local",
                    "initial_value": self.edit_slope.text() if hasattr(self, "edit_slope") else "-59.16",
                    "min": -120.0,
                    "max": 120.0,
                    "fixed": True,
                    "linked_species": "",
                    "description": "Electrode slope, mV/pH.",
                },
                {
                    "parameter": "pKw",
                    "type": "local",
                    "initial_value": f"{float(self.spin_pkw.value()):.4f}" if hasattr(self, "spin_pkw") else "14.0000",
                    "min": f"{float(self.spin_pkw_min.value()):.4f}" if hasattr(self, "spin_pkw_min") else 0.0,
                    "max": f"{float(self.spin_pkw_max.value()):.4f}" if hasattr(self, "spin_pkw_max") else 30.0,
                    "fixed": not bool(self.chk_fit_pkw.isChecked()) if hasattr(self, "chk_fit_pkw") else True,
                    "linked_species": "",
                    "description": "Apparent medium autoprotolysis; Kw = 10^(-pKw).",
                },
                {
                    "parameter": "baseline",
                    "type": "local",
                    "initial_value": 1 if bool(self.chk_baseline.isChecked()) else 0,
                    "min": "",
                    "max": "",
                    "fixed": True,
                    "linked_species": "",
                    "description": "Optional spectroscopy baseline flag.",
                },
            ]
        )

        pkw_tooltip = (
            "Kw = 10^(-pKw). This value affects potentiometric electroneutrality "
            "calculations only; imposed-pH spectroscopy and NMR fits do not use "
            "Kw to calculate fractions in v1."
        )
        was_updating = self._updating_tables
        self._updating_tables = True
        try:
            self.parameters_table.setRowCount(0)
            for row_data in rows:
                row = self.parameters_table.rowCount()
                self.parameters_table.insertRow(row)
                name = str(row_data["parameter"])
                initial = previous_values.get(name, row_data.get("initial_value", ""))
                if initial is None:
                    initial = ""
                row_min = previous_min.get(name, row_data.get("min", ""))
                row_max = previous_max.get(name, row_data.get("max", ""))
                for col, value in enumerate(
                    [
                        name,
                        row_data.get("type", ""),
                        initial,
                        row_min,
                        row_max,
                    ]
                ):
                    _set_table_text(self.parameters_table, row, col, value)
                    if name == "pKw":
                        item = self.parameters_table.item(row, col)
                        if item is not None:
                            item.setToolTip(pkw_tooltip)
                checked = previous_fit.get(name, not bool(row_data.get("fixed", False)))
                lowered_name = name.strip().lower()
                if lowered_name == "analyte concentration":
                    checked = bool(self.chk_fit_analyte_conc.isChecked())
                elif lowered_name == "titrant concentration":
                    checked = bool(self.chk_fit_titrant_conc.isChecked())
                elif lowered_name == "volume offset":
                    checked = bool(self.chk_fit_volume_offset.isChecked())
                elif lowered_name in {"electrode_e0", "electrode_slope"}:
                    checked = bool(self.chk_fit_electrode_basic.isChecked())
                elif lowered_name == "pkw":
                    checked = bool(self.chk_fit_pkw.isChecked())
                elif lowered_name.startswith("pka") and name not in previous_fit:
                    try:
                        pka_row = int(lowered_name.replace("pka", "") or "0") - 1
                        pka_fit = self.table_basic_pka.cellWidget(pka_row, 4)
                        if isinstance(pka_fit, QCheckBox):
                            checked = bool(pka_fit.isChecked())
                    except Exception:
                        pass
                fit = self._make_checkbox(checked)
                fit.stateChanged.connect(
                    lambda state, pname=name: self._on_parameter_fit_changed(
                        pname,
                        state == Qt.CheckState.Checked.value,
                    )
                )
                if name == "pKw":
                    fit.setToolTip(pkw_tooltip)
                self.parameters_table.setCellWidget(row, 5, fit)
                _set_table_text(self.parameters_table, row, 6, row_data.get("linked_species", ""))
                _set_table_text(self.parameters_table, row, 7, row_data.get("description", ""))
                if name == "pKw":
                    try:
                        pkw_value = float(initial)
                    except Exception:
                        pkw_value = 14.0
                    self.spin_pkw.blockSignals(True)
                    self.spin_pkw.setValue(pkw_value)
                    self.spin_pkw.blockSignals(False)
                    try:
                        pkw_min = float(row_min)
                    except Exception:
                        pkw_min = 0.0
                    try:
                        pkw_max = float(row_max)
                    except Exception:
                        pkw_max = 30.0
                    self.spin_pkw_min.blockSignals(True)
                    self.spin_pkw_max.blockSignals(True)
                    self.spin_pkw_min.setValue(pkw_min)
                    self.spin_pkw_max.setValue(pkw_max)
                    self.spin_pkw_min.blockSignals(False)
                    self.spin_pkw_max.blockSignals(False)
                    self.chk_fit_pkw.blockSignals(True)
                    self.chk_fit_pkw.setChecked(checked)
                    self.chk_fit_pkw.blockSignals(False)
        finally:
            self._updating_tables = was_updating
        self._set_basic_pka_table_from_model(self._apply_parameter_table_to_model(model))
        self._apply_parameter_table_visibility()

    def _on_pkw_spin_changed(self, value: float) -> None:
        if self._updating_tables:
            return
        self._set_parameter_value("pKw", f"{float(value):.4f}", update_spin=False)

    def _on_medium_changed(self) -> None:
        if self._updating_tables:
            return
        medium = str(self.combo_medium.currentData() or "aqueous")
        if medium == "aqueous":
            self.spin_pkw.setValue(14.0)
            self.spin_pkw_min.setValue(0.0)
            self.spin_pkw_max.setValue(30.0)
            self.chk_fit_pkw.setChecked(False)
        elif medium == "mixed":
            self.spin_pkw.setValue(18.0)
            self.spin_pkw_min.setValue(10.0)
            self.spin_pkw_max.setValue(30.0)
        self._sync_pkw_ui()
        self._refresh_parameter_table()

    def _on_pkw_bounds_changed(self, _value: float) -> None:
        if self._updating_tables:
            return
        self._set_parameter_bounds(
            "pKw",
            f"{float(self.spin_pkw_min.value()):.4f}",
            f"{float(self.spin_pkw_max.value()):.4f}",
        )

    def _on_fit_pkw_toggled(self, checked: bool) -> None:
        if self._updating_tables:
            return
        self._set_parameter_fit("pKw", bool(checked))
        self._sync_pkw_ui()

    def _on_parameter_fit_changed(self, name: str, checked: bool) -> None:
        if self._updating_tables:
            return
        lowered = name.strip().lower()
        checkbox_map = {
            "analyte concentration": getattr(self, "chk_fit_analyte_conc", None),
            "titrant concentration": getattr(self, "chk_fit_titrant_conc", None),
            "volume offset": getattr(self, "chk_fit_volume_offset", None),
        }
        target = checkbox_map.get(lowered)
        if isinstance(target, QCheckBox):
            target.blockSignals(True)
            target.setChecked(bool(checked))
            target.blockSignals(False)
        if lowered in {"electrode_e0", "electrode_slope"} and hasattr(self, "chk_fit_electrode_basic"):
            self.chk_fit_electrode_basic.blockSignals(True)
            self.chk_fit_electrode_basic.setChecked(bool(checked))
            self.chk_fit_electrode_basic.blockSignals(False)
        if lowered == "pkw" and hasattr(self, "chk_fit_pkw"):
            self.chk_fit_pkw.blockSignals(True)
            self.chk_fit_pkw.setChecked(bool(checked))
            self.chk_fit_pkw.blockSignals(False)
            self._sync_pkw_ui()
        pka_match = re.search(r"pka[_\-\s]*(\d+)$", lowered)
        if pka_match is not None and hasattr(self, "table_basic_pka"):
            idx = int(pka_match.group(1)) - 1
            if 0 <= idx < self.table_basic_pka.rowCount():
                pka_fit = self.table_basic_pka.cellWidget(idx, 4)
                if isinstance(pka_fit, QCheckBox):
                    pka_fit.blockSignals(True)
                    pka_fit.setChecked(bool(checked))
                    pka_fit.blockSignals(False)
        if not self._advanced_visible():
            self._apply_parameter_table_visibility()

    def _parameter_value(self, name: str, default: Any = "") -> Any:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                value = _table_text(self.parameters_table, row, 2, "")
                return default if value == "" else value
        return default

    def _parameter_min_max(self, name: str, default_min: Any = "", default_max: Any = "") -> tuple[Any, Any]:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                lo = _table_text(self.parameters_table, row, 3, "")
                hi = _table_text(self.parameters_table, row, 4, "")
                return (default_min if lo == "" else lo, default_max if hi == "" else hi)
        return default_min, default_max

    def _parameter_fixed(self, name: str, default: bool = True) -> bool:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                fit_widget = self.parameters_table.cellWidget(row, 5)
                if isinstance(fit_widget, QCheckBox):
                    return not bool(fit_widget.isChecked())
                return bool(default)
        return bool(default)

    def _acid_base_model_config(self) -> dict[str, Any]:
        model = self._current_model_from_ui()
        model = self._apply_parameter_table_to_model(model)
        model["definition_mode"] = "equations" if self.radio_model_equations.isChecked() else "matrix"
        model["constant_mode"] = normalize_constant_mode(self.combo_constant_mode.currentData())
        model["equations_text"] = str(self.equation_editor.toPlainText() or "")
        return model

    def _first_analyte_component(self, model: dict[str, Any]) -> dict[str, Any]:
        components = list(model.get("components") or [])
        for comp in components:
            if bool(comp.get("is_proton", False)):
                continue
            if str(comp.get("role") or "").lower() == "analyte":
                return comp
        for comp in components:
            if not bool(comp.get("is_proton", False)):
                return comp
        raise ValueError("Define at least one non-proton acid-base component.")

    def _friendly_model_message(self, message: str) -> str:
        text = str(message or "")
        lowered = text.lower()
        if "missing constants" in lowered or "non-consecutive" in lowered or "stoichiometric" in lowered:
            return (
                "The selected acid-base model could not be generated correctly. "
                "Please check the number of pKa values or switch to Advanced mode."
            )
        if "proton" in lowered:
            return "The model needs a proton component. This is created automatically in Basic mode."
        if "charge" in lowered:
            return "The model needs valid charges for potentiometry. Check Advanced mode if you are using a custom model."
        return text

    def _validate_basic_species_table(self) -> list[str]:
        if self._advanced_visible():
            return []
        errors: list[str] = []
        rows: list[dict[str, Any]] = []
        for row in range(self.table_basic_species.rowCount()):
            try:
                h_count = int(float(_table_text(self.table_basic_species, row, 1, "")))
                charge = int(float(_table_text(self.table_basic_species, row, 2, "")))
            except Exception:
                errors.append("Each acid-base species needs numeric h_count and charge values.")
                continue
            if h_count < 0:
                errors.append("h_count must be an integer greater than or equal to 0.")
            rows.append({"h_count": h_count, "charge": charge})
        if not rows:
            errors.append("Add at least one acid-base species.")
            return errors
        h_counts = [int(row["h_count"]) for row in rows]
        if 0 not in h_counts:
            errors.append("At least one species must have h_count = 0.")
        if len(h_counts) != len(set(h_counts)):
            errors.append("Each included acid-base species should have a unique h_count in Basic mode.")
        if sorted(h_counts) != list(range(max(h_counts) + 1)):
            errors.append("Basic mode expects consecutive h_count values: 0, 1, 2, ...")
        return errors

    def _validate_internal_proton(self, model: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        proton_rows = [
            comp
            for comp in list(model.get("components") or [])
            if bool(comp.get("is_proton")) or str(comp.get("name") or "").strip().lower() == proton_component_name().lower()
        ]
        if not proton_rows:
            errors.append("The proton component H+ is created automatically. Please check the species h_count values.")
            return errors
        for comp in proton_rows:
            if int(comp.get("charge") or 0) != 1 or bool(comp.get("is_titrant", False)):
                errors.append("The internal H component must be H+ and cannot be used as the titrant.")
                break
        return errors

    def _snapshot_ui_state(self) -> AcidBaseUiState:
        pka_rows = self._basic_pka_rows() if hasattr(self, "table_basic_pka") else []
        pkw_min, pkw_max = self._parameter_min_max("pKw", default_min="0.0", default_max="30.0")
        return AcidBaseUiState(
            data_type=str(self.combo_data_type.currentData() or "potentiometry"),
            file_path=str(self._file_path or ""),
            sheet_name=str(self.combo_sheet.currentData() or ""),
            volume_column=str(self.combo_volume_column.currentData() or ""),
            volume_unit=str(self.combo_volume_unit.currentData() or "mL"),
            signal_type=str(self.combo_signal_type.currentData() or "pH"),
            signal_column=str(self.combo_signal_column.currentData() or ""),
            analyte_name=str(self.edit_analyte_name.text() or "L"),
            analyte_concentration=float(self.spin_analyte_conc.value()),
            initial_volume=float(self.spin_initial_volume.value()),
            titrant_type=str(self.combo_titrant_type.currentData() or "base"),
            titrant_concentration=float(self.spin_titrant_conc.value()),
            charge_L=int(self.spin_base_charge.value()),
            species_rows=self._basic_species_rows() if hasattr(self, "table_basic_species") else [],
            pka_rows=pka_rows,
            pkw=float(self._parameter_value("pKw", default="14.0000")),
            pkw_bounds=(float(pkw_min), float(pkw_max)),
            fit_pkw=not self._parameter_fixed("pKw", default=True),
            fit_analyte_concentration=not self._parameter_fixed("analyte concentration", default=True),
            fit_titrant_concentration=not self._parameter_fixed("titrant concentration", default=True),
            fit_volume_offset=not self._parameter_fixed("volume offset", default=True),
            fit_electrode=(
                (not self._parameter_fixed("electrode_e0", default=True))
                or (not self._parameter_fixed("electrode_slope", default=True))
            ),
            advanced_options={
                "advanced_mode": self._advanced_visible(),
                "medium": str(self.combo_medium.currentData() or "aqueous") if hasattr(self, "combo_medium") else "aqueous",
                "pH_bounds": [float(self.spin_ph_min.value()), float(self.spin_ph_max.value())],
                "sigma_pH": float(self.spin_sigma_ph.value()),
                "sigma_E": float(self.spin_sigma_emf.value()),
            },
        )

    def _experimental_ph_values(self) -> list[float]:
        if str(self.combo_data_type.currentData() or "") == "potentiometry":
            try:
                payload = self._build_potentiometry_import_payload()
            except Exception:
                return []
            signal_type = str(payload.get("signal_type") or "")
            if signal_type.lower() not in {"ph", "ph*"}:
                return []
            observed = np.asarray(payload.get("observed_signal") or [], dtype=float)
            mask = np.asarray(payload.get("included_mask") or [], dtype=bool)
            if observed.size != mask.size:
                return []
            return [float(v) for v in observed[mask]]
        path = Path(str(self._file_path or ""))
        if not path.exists():
            return []
        try:
            if path.suffix.lower() == ".xlsx":
                sheet = self.combo_sheet.currentData()
                if sheet in (None, ""):
                    sheet = 0
                df = pd.read_excel(path, sheet_name=sheet)
            else:
                df = pd.read_csv(path)
        except Exception:
            return []
        columns = {str(col).strip().lower(): str(col) for col in df.columns}
        ph_col = columns.get("ph")
        if ph_col is None:
            return []
        series = pd.to_numeric(df[ph_col], errors="coerce").dropna()
        return [float(value) for value in series.to_numpy(dtype=float)]

    def _collect_config(self) -> dict[str, Any]:
        if not self._file_path:
            raise ValueError("No data file selected.")
        basic_errors = self._validate_basic_species_table()
        if basic_errors:
            raise ValueError("\n".join(basic_errors))
        model = self._acid_base_model_config()
        self.ui_state = self._snapshot_ui_state()
        analysis_kind = str(self.combo_data_type.currentData() or "potentiometry")
        if float(self.spin_initial_volume.value()) <= 0.0:
            raise ValueError("Initial volume must be greater than zero.")
        if float(self.spin_analyte_conc.value()) <= 0.0:
            raise ValueError("Analyte concentration must be greater than zero.")
        if float(self.spin_titrant_conc.value()) <= 0.0:
            raise ValueError("Titrant concentration must be greater than zero.")
        for value in self._basic_pka_values():
            if not -5.0 <= float(value) <= 25.0:
                raise ValueError("pKa initial guesses should be between -5 and 25.")
        errors, warnings = validate_acid_base_model(
            model,
            analysis_kind=analysis_kind,
            experimental_pH=self._experimental_ph_values(),
            require_charges=analysis_kind == "potentiometry",
        )
        if errors:
            raise ValueError("\n".join(self._friendly_model_message(msg) for msg in errors))
        proton_errors = self._validate_internal_proton(model)
        if proton_errors:
            raise ValueError("\n".join(proton_errors))
        for warning in warnings:
            self.log.append_text(f"Warning: {warning}")
        analyte = self._first_analyte_component(model)
        blocks = acid_base_constant_blocks(model)
        primary_block = next(
            (block for block in blocks if block["component_name"] == str(analyte.get("name") or "")),
            blocks[0] if blocks else {"pka": [], "log_beta": []},
        )
        try:
            pkw = float(self._parameter_value("pKw", default="14.0000"))
            pkw_min_raw, pkw_max_raw = self._parameter_min_max("pKw", default_min="0.0", default_max="30.0")
            pkw_min = float(pkw_min_raw)
            pkw_max = float(pkw_max_raw)
        except Exception as exc:
            raise ValueError("pKw and pKw bounds must be numeric.") from exc
        if pkw_min >= pkw_max:
            raise ValueError("pKw min must be lower than pKw max.")
        if not pkw_min <= pkw <= pkw_max:
            raise ValueError("Initial pKw must be within the configured pKw bounds.")
        kw = 10.0 ** (-pkw)
        fit_pkw = not self._parameter_fixed("pKw", default=True)
        first_pka = [float(v) for v in list(primary_block.get("pka") or [])]
        titrant_concentrations, custom_strong_charge = self._custom_titrant_config()
        strong_mode = str(self.combo_strong_ion.currentData() or "automatic")
        titrant_type = str(self.combo_titrant_type.currentData() or "base")

        cfg = {
            "file_path": self._file_path,
            "sheet_name": str(self.combo_sheet.currentData() or ""),
            "data_type": analysis_kind,
            "acid_base_model": model,
            "component_name": str(analyte.get("name") or "L"),
            "pka_initial": ", ".join(str(v) for v in first_pka) if first_pka else "5.0",
            "analyte_concentration": float(analyte.get("analytical_concentration", 0.0) or 0.0),
            "base_charge": int(analyte.get("charge", -1) or -1),
            "initial_volume": float(self.spin_initial_volume.value()),
            "titrant_concentration": float(self.spin_titrant_conc.value()),
            "titrant_type": titrant_type,
            "strong_ion_mode": strong_mode,
            "volume_offset": float(self.spin_volume_offset.value()),
            "pH_bounds": [float(self.spin_ph_min.value()), float(self.spin_ph_max.value())],
            "electrode_e0": _optional_float(str(self._parameter_value("electrode_e0", default=self.edit_e0.text()))),
            "electrode_slope": _optional_float(str(self._parameter_value("electrode_slope", default=self.edit_slope.text()))),
            "fit_electrode": (
                (not self._parameter_fixed("electrode_e0", default=True))
                or (not self._parameter_fixed("electrode_slope", default=True))
            ),
            "fit_analyte_concentration": not self._parameter_fixed("analyte concentration", default=True),
            "fit_titrant_concentration": not self._parameter_fixed("titrant concentration", default=True),
            "fit_volume_offset": not self._parameter_fixed("volume offset", default=True),
            "pka_fit_mask": [bool(row["fit"]) for row in self._basic_pka_rows()],
            "pka_bounds": [[float(row["min"]), float(row["max"])] for row in self._basic_pka_rows()],
            "sigma_pH": float(self.spin_sigma_ph.value()),
            "sigma_E": float(self.spin_sigma_emf.value()),
            "pkw": pkw,
            "pkw_bounds": [pkw_min, pkw_max],
            "fit_pkw": bool(fit_pkw),
            "kw": kw,
            "baseline": bool(int(float(self._parameter_value("baseline", default="1" if self.chk_baseline.isChecked() else "0")))),
        }
        if analysis_kind == "potentiometry":
            cfg.update(self._build_potentiometry_import_payload())
            if fit_pkw and (
                bool(cfg["fit_analyte_concentration"])
                or bool(cfg["fit_titrant_concentration"])
                or bool(cfg["fit_volume_offset"])
                or bool(cfg["fit_electrode"])
            ):
                self.log.append_text(
                    "Warning: Fitting pKw together with concentration, electrode and volume "
                    "offset parameters may lead to strong parameter correlation. For reliable "
                    "results, fit pKw only when the titration data contain enough information "
                    "and the solvent system justifies it."
                )
        if strong_mode == "manual":
            cfg["initial_strong_charge"] = float(self.spin_initial_strong_charge.value())
            cfg["titrant_strong_charge"] = float(self.spin_titrant_strong_charge.value())
        if titrant_type == "custom":
            cfg["titrant_concentrations"] = titrant_concentrations
            cfg["titrant_strong_charge"] = custom_strong_charge
        return cfg

    def _selected_plot_ids(self) -> set[str]:
        selected: set[str] = set()
        for row in range(self.plot_options.count()):
            item = self.plot_options.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected.add(str(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def _sync_plot_options_for_analysis(self) -> None:
        if not hasattr(self, "plot_options"):
            return
        kind = str(self.combo_data_type.currentData() or "potentiometry")
        visible_by_kind = {
            "potentiometry": {"fit", "residuals", "species_pH", "species_volume"},
            "spectroscopy": {"fit", "residuals", "species_pH", "spectroscopy_pure"},
            "nmr": {"fit", "residuals", "species_pH", "nmr_limiting"},
        }.get(kind, {"fit", "residuals"})
        for row in range(self.plot_options.count()):
            item = self.plot_options.item(row)
            plot_id = str(item.data(Qt.ItemDataRole.UserRole))
            item.setHidden(plot_id not in visible_by_kind)
            if plot_id in visible_by_kind and item.checkState() != Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Checked)

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
                self.combo_sheet.setVisible(False)
                self.lbl_sheet.setVisible(False)
                self.preview_text.setPlainText(f"Could not read Excel workbook: {exc}")
                self.combo_sheet.blockSignals(False)
                return
            for sheet in sheets:
                self.combo_sheet.addItem(str(sheet), str(sheet))
            show_sheet = len(sheets) > 1
            self.combo_sheet.setEnabled(show_sheet)
            self.combo_sheet.setVisible(show_sheet)
            self.lbl_sheet.setVisible(show_sheet)
        else:
            self.combo_sheet.addItem("CSV/TXT", "")
            self.combo_sheet.setEnabled(False)
            self.combo_sheet.setVisible(False)
            self.lbl_sheet.setVisible(False)
        self.combo_sheet.blockSignals(False)
        self._preview_selected_sheet()

    def _on_sheet_changed(self) -> None:
        self._preview_selected_sheet()

    def _read_selected_dataframe(self, *, nrows: int | None = None) -> pd.DataFrame:
        path = Path(str(self._file_path or ""))
        if not path.exists():
            raise FileNotFoundError("No data file selected.")
        if path.suffix.lower() == ".xlsx":
            sheet = self.combo_sheet.currentData()
            if sheet in (None, ""):
                sheet = 0
            return pd.read_excel(path, sheet_name=sheet, nrows=nrows)
        return pd.read_csv(path, nrows=nrows)

    def _populate_potentiometry_columns(self, columns: list[str]) -> None:
        previous_volume = str(self.combo_volume_column.currentData() or self.combo_volume_column.currentText() or "")
        previous_signal = str(self.combo_signal_column.currentData() or self.combo_signal_column.currentText() or "")
        for combo in (self.combo_volume_column, self.combo_signal_column):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select column...", "")
            for column in columns:
                combo.addItem(column, column)
            combo.blockSignals(False)
        for combo, wanted in (
            (self.combo_volume_column, previous_volume),
            (self.combo_signal_column, previous_signal),
        ):
            index = combo.findData(wanted)
            if index < 0 and not wanted:
                index = 0
            if index >= 0:
                combo.setCurrentIndex(index)
        if self.combo_volume_column.currentIndex() <= 0:
            scored = [(_volume_column_score(column), idx, column) for idx, column in enumerate(columns, start=1)]
            scored = [item for item in scored if item[0] > 0]
            if scored:
                _score, idx, column = max(scored, key=lambda item: (item[0], -item[1]))
                self.combo_volume_column.setCurrentIndex(idx)
                guessed_unit = _guess_volume_unit_from_name(column)
                if guessed_unit:
                    unit_idx = self.combo_volume_unit.findData(guessed_unit)
                    if unit_idx >= 0:
                        self.combo_volume_unit.setCurrentIndex(unit_idx)
        if self.combo_signal_column.currentIndex() <= 0:
            scored_signal = [(*_signal_column_score(column), idx) for idx, column in enumerate(columns, start=1)]
            scored_signal = [item for item in scored_signal if item[0] > 0]
            if scored_signal:
                _score, kind, idx = max(scored_signal, key=lambda item: (item[0], -item[2]))
                self.combo_signal_column.setCurrentIndex(idx)
                if kind:
                    type_idx = self.combo_signal_type.findData(kind)
                    if type_idx >= 0:
                        self.combo_signal_type.setCurrentIndex(type_idx)
        if self.combo_signal_column.currentIndex() > 0:
            signal_name = str(self.combo_signal_column.currentData() or "")
            _score, kind = _signal_column_score(signal_name)
            if kind:
                idx = self.combo_signal_type.findData(kind)
                if idx >= 0:
                    self.combo_signal_type.setCurrentIndex(idx)

    def _set_pot_validation_messages(self, errors: list[str], warnings: list[str]) -> None:
        self._potentiometry_warning_messages = list(warnings)
        lines: list[str] = []
        if errors:
            lines.extend([f"Error: {msg}" for msg in errors])
        if warnings:
            lines.extend([f"Warning: {msg}" for msg in warnings])
        self.lbl_pot_validation.setText("\n".join(lines) if lines else "Pre-fit checks: no issues detected yet.")
        color = "#a40000" if errors else ("#8f5c00" if warnings else "#444444")
        self.lbl_pot_validation.setStyleSheet(f"color: {color};")

    def _refresh_potentiometry_preview(self) -> None:
        existing_mask = self._current_preview_mask() if hasattr(self, "table_pot_preview") else []
        self.table_pot_preview.setRowCount(0)
        if str(self.combo_data_type.currentData() or "") != "potentiometry":
            self._set_pot_validation_messages([], [])
            return
        if self._current_data_frame is None or self._current_data_frame.empty:
            self._set_pot_validation_messages([], [])
            return
        volume_col = str(self.combo_volume_column.currentData() or "")
        signal_col = str(self.combo_signal_column.currentData() or "")
        if not volume_col or not signal_col:
            self._set_pot_validation_messages(["Select both the volume and signal columns."], [])
            return
        errors, warnings, preview_df = self._build_potentiometry_preview_dataframe(self._current_data_frame)
        if existing_mask and len(existing_mask) == len(preview_df.index):
            preview_df["include"] = existing_mask
        self._set_pot_validation_messages(errors, warnings)

        signal_type = str(self.combo_signal_type.currentData() or "signal")
        self.table_pot_preview.setHorizontalHeaderLabels(
            [
                "Include",
                "Original row",
                f"{volume_col}",
                "Volume (mL)",
                f"{signal_col} ({signal_type})",
            ]
        )
        self.table_pot_preview.blockSignals(True)
        try:
            self.table_pot_preview.setRowCount(len(preview_df.index))
            for row_idx, row in enumerate(preview_df.to_dict(orient="records")):
                include_item = QTableWidgetItem()
                include_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                include_item.setCheckState(
                    Qt.CheckState.Checked if bool(row["include"]) else Qt.CheckState.Unchecked
                )
                self.table_pot_preview.setItem(row_idx, 0, include_item)
                _set_table_text(self.table_pot_preview, row_idx, 1, int(row["original_row"]))
                _set_table_text(self.table_pot_preview, row_idx, 2, row["volume_raw"])
                _set_table_text(self.table_pot_preview, row_idx, 3, row["volume_mL"])
                _set_table_text(self.table_pot_preview, row_idx, 4, row["signal_raw"])
        finally:
            self.table_pot_preview.blockSignals(False)

    def _current_preview_mask(self) -> list[bool]:
        mask: list[bool] = []
        for row in range(self.table_pot_preview.rowCount()):
            item = self.table_pot_preview.item(row, 0)
            mask.append(bool(item is not None and item.checkState() == Qt.CheckState.Checked))
        return mask

    def _apply_preview_mask(self, mask: list[bool]) -> None:
        self.table_pot_preview.blockSignals(True)
        try:
            for row, include in enumerate(mask[: self.table_pot_preview.rowCount()]):
                item = self.table_pot_preview.item(row, 0)
                if item is not None:
                    item.setCheckState(Qt.CheckState.Checked if include else Qt.CheckState.Unchecked)
        finally:
            self.table_pot_preview.blockSignals(False)
        try:
            errors, warnings, _preview_df = self._build_potentiometry_preview_dataframe(
                self._current_data_frame,
                include_from_table=True,
            )
            self._set_pot_validation_messages(errors, warnings)
        except Exception:
            pass

    def _set_preview_inclusion(self, action: str) -> None:
        n = self.table_pot_preview.rowCount()
        if n <= 0:
            return
        current = self._current_preview_mask()
        action = str(action)
        if action != "restore":
            self._pot_preview_restore_mask = list(current)
        if action == "all":
            mask = [True] * n
        elif action == "none":
            mask = [False] * n
        elif action == "exclude_first":
            mask = list(current)
            if mask:
                mask[0] = False
        elif action == "exclude_selected":
            selected_rows = {idx.row() for idx in self.table_pot_preview.selectedIndexes()}
            mask = [include and row not in selected_rows for row, include in enumerate(current)]
        elif action == "restore" and self._pot_preview_restore_mask is not None:
            mask = list(self._pot_preview_restore_mask)
        else:
            return
        self._apply_preview_mask(mask)

    def _build_potentiometry_preview_dataframe(
        self,
        df: pd.DataFrame,
        *,
        include_from_table: bool = True,
    ) -> tuple[list[str], list[str], pd.DataFrame]:
        volume_col = str(self.combo_volume_column.currentData() or "")
        signal_col = str(self.combo_signal_column.currentData() or "")
        errors: list[str] = []
        warnings: list[str] = []
        if not volume_col or volume_col not in df.columns:
            errors.append("Select a valid volume column.")
        if not signal_col or signal_col not in df.columns:
            errors.append("Select a valid signal column.")
        if errors:
            return errors, warnings, pd.DataFrame()

        volume_numeric = pd.to_numeric(df[volume_col], errors="coerce")
        signal_numeric = pd.to_numeric(df[signal_col], errors="coerce")
        invalid_volume_rows = [int(idx) + 2 for idx, value in enumerate(volume_numeric) if pd.isna(value)]
        invalid_signal_rows = [int(idx) + 2 for idx, value in enumerate(signal_numeric) if pd.isna(value)]
        if invalid_volume_rows:
            errors.append(
                "Volume column must be numeric. Invalid rows: "
                + ", ".join(str(v) for v in invalid_volume_rows[:8])
            )
        if invalid_signal_rows:
            errors.append(
                "Signal column must be numeric. Invalid rows: "
                + ", ".join(str(v) for v in invalid_signal_rows[:8])
            )
        factor = _volume_unit_factor_to_mL(str(self.combo_volume_unit.currentData() or "mL"))
        converted_volume = volume_numeric.astype(float) * factor
        finite_volume = converted_volume.dropna().to_numpy(dtype=float)
        if finite_volume.size >= 2 and np.any(np.diff(finite_volume) < 0.0):
            errors.append("Volume should be non-decreasing.")
        if finite_volume.size and abs(float(finite_volume[0])) <= 1.0e-15:
            warnings.append("Initial volume is zero.")
        if finite_volume.size:
            vmax = float(np.nanmax(np.abs(finite_volume)))
            if vmax > 10000.0:
                warnings.append("Converted volumes are very large in mL. Check whether the source column is in µL.")
            elif 0.0 < vmax < 1.0e-6:
                warnings.append("Converted volumes are extremely small in mL. Check the selected volume unit.")
        if len(finite_volume) < 6:
            warnings.append("Very few points are included. Fits are usually more reliable with more titration points.")
        signal_type = str(self.combo_signal_type.currentData() or "")
        finite_signal = signal_numeric.dropna().to_numpy(dtype=float)
        if signal_type.lower() in {"ph", "ph*"} and finite_signal.size:
            signal_min = float(np.nanmin(finite_signal))
            signal_max = float(np.nanmax(finite_signal))
            if signal_min < -2.0 or signal_max > 16.0:
                warnings.append("Some pH values are outside the usual -2 to 16 range; the fit can still run.")
            if signal_max > 14.0:
                warnings.append("pH values above 14 detected. For mixed solvents, consider using apparent pKw and wider pH bounds.")

        preview_df = pd.DataFrame(
            {
                "include": [True] * len(df.index),
                "original_row": np.arange(len(df.index), dtype=int) + 2,
                "volume_raw": df[volume_col].tolist(),
                "volume_mL": converted_volume.tolist(),
                "signal_raw": signal_numeric.astype(float).tolist(),
            }
        )
        if include_from_table and self.table_pot_preview.rowCount() == len(df.index):
            include_mask: list[bool] = []
            for row in range(self.table_pot_preview.rowCount()):
                item = self.table_pot_preview.item(row, 0)
                include_mask.append(bool(item is not None and item.checkState() == Qt.CheckState.Checked))
            preview_df["include"] = include_mask
        return errors, warnings, preview_df

    def _build_potentiometry_import_payload(self) -> dict[str, Any]:
        if self._current_data_frame is None:
            raise ValueError("Load a data file before processing potentiometry data.")
        errors, warnings, preview_df = self._build_potentiometry_preview_dataframe(
            self._current_data_frame,
            include_from_table=True,
        )
        self._set_pot_validation_messages(errors, warnings)
        if errors:
            raise ValueError("\n".join(errors))
        included_mask = preview_df["include"].to_numpy(dtype=bool)
        if not np.any(included_mask):
            raise ValueError("Select at least one potentiometry row to include.")
        return {
            "titrant_volume": preview_df["volume_mL"].to_numpy(dtype=float).tolist(),
            "observed_signal": preview_df["signal_raw"].to_numpy(dtype=float).tolist(),
            "included_mask": included_mask.tolist(),
            "signal_type": str(self.combo_signal_type.currentData() or "pH"),
            "volume_unit": str(self.combo_volume_unit.currentData() or "mL"),
            "volume_column": str(self.combo_volume_column.currentData() or ""),
            "signal_column": str(self.combo_signal_column.currentData() or ""),
            "potentiometry_warnings": list(warnings),
        }

    def _preview_selected_sheet(self) -> None:
        path = Path(str(self._file_path or ""))
        if not path.exists():
            return
        try:
            self._current_data_frame = self._read_selected_dataframe()
            preview_df = self._current_data_frame.head(8)
            self.preview_text.setPlainText(preview_df.to_string(index=False))
            self._populate_potentiometry_columns([str(col) for col in self._current_data_frame.columns])
            self._refresh_potentiometry_preview()
        except Exception as exc:
            self._current_data_frame = None
            self.preview_text.setPlainText(f"Could not preview data file: {exc}")
            self.combo_volume_column.clear()
            self.combo_signal_column.clear()
            self.table_pot_preview.setRowCount(0)
            self._set_pot_validation_messages([f"Could not preview data file: {exc}"], [])

    def _suggest_setup_from_data(self) -> None:
        if self._current_data_frame is None or self._current_data_frame.empty:
            self.log.append_text("Suggestion: load a potentiometry file first.")
            return
        columns = [str(col) for col in self._current_data_frame.columns]
        self._populate_potentiometry_columns(columns)
        suggestions: list[str] = []
        volume_col = str(self.combo_volume_column.currentData() or "")
        signal_col = str(self.combo_signal_column.currentData() or "")
        if volume_col:
            unit = _guess_volume_unit_from_name(volume_col)
            raw_volume = pd.to_numeric(self._current_data_frame[volume_col], errors="coerce").dropna()
            if unit is None and not raw_volume.empty:
                vmax = float(raw_volume.max())
                unit = "µL" if vmax > 1000.0 else "mL"
            if unit:
                idx = self.combo_volume_unit.findData(unit)
                if idx >= 0:
                    self.combo_volume_unit.setCurrentIndex(idx)
                    suggestions.append(f"Volume unit: {unit}")
        if signal_col:
            _score, kind = _signal_column_score(signal_col)
            if kind:
                idx = self.combo_signal_type.findData(kind)
                if idx >= 0:
                    self.combo_signal_type.setCurrentIndex(idx)
                    suggestions.append(f"Signal type: {'EMF' if kind == 'mV' else kind}")
        if str(self.combo_signal_type.currentData() or "").lower() in {"ph", "ph*"} and signal_col:
            signal = pd.to_numeric(self._current_data_frame[signal_col], errors="coerce").dropna()
            if not signal.empty:
                ph_min = float(signal.min())
                ph_max = float(signal.max())
                self.spin_ph_min.setValue(max(-10.0, math.floor(ph_min) - 1.0))
                self.spin_ph_max.setValue(min(30.0, math.ceil(ph_max) + 1.0))
                suggestions.append(f"pH bounds: {self.spin_ph_min.value():.0f} to {self.spin_ph_max.value():.0f}")
                if ph_max > 14.0:
                    idx = self.combo_medium.findData("mixed")
                    if idx >= 0:
                        self.combo_medium.setCurrentIndex(idx)
                    suggestions.append("pH > 14: apparent pKw_app = 18 suggested for mixed/non-aqueous media.")
        self._refresh_potentiometry_preview()
        self.log.append_text("Suggestions: " + ("; ".join(suggestions) if suggestions else "no changes suggested."))

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
        volume_unit = str(config.get("volume_unit") or "")
        if volume_unit:
            idx = self.combo_volume_unit.findData(volume_unit)
            if idx >= 0:
                self.combo_volume_unit.setCurrentIndex(idx)
        signal_type = str(config.get("signal_type") or "")
        if signal_type:
            idx = self.combo_signal_type.findData(signal_type)
            if idx >= 0:
                self.combo_signal_type.setCurrentIndex(idx)
        volume_column = str(config.get("volume_column") or "")
        if volume_column:
            idx = self.combo_volume_column.findData(volume_column)
            if idx >= 0:
                self.combo_volume_column.setCurrentIndex(idx)
        signal_column = str(config.get("signal_column") or "")
        if signal_column:
            idx = self.combo_signal_column.findData(signal_column)
            if idx >= 0:
                self.combo_signal_column.setCurrentIndex(idx)
        model = config.get("acid_base_model")
        if model:
            is_custom_model = self._model_definition_is_custom(dict(model))
            self._populate_model_from_definition(
                canonicalize_acid_base_model(
                    model,
                    auto_add_proton=not is_custom_model,
                    infer_proton_from_name=not is_custom_model,
                ),
                update_template=True,
            )
        else:
            fallback_model = canonicalize_acid_base_model(
                None,
                fallback_component_name=str(config.get("component_name") or "L"),
                fallback_pka=_parse_float_list(config.get("pka_initial") or config.get("initial_pka"), default=[5.0]),
                fallback_concentration=float(config.get("analyte_concentration", 1.0e-3) or 1.0e-3),
                fallback_base_charge=int(config.get("base_charge", -1) or -1),
            )
            self._populate_model_from_definition(fallback_model, update_template=False)
        if bool(config.get("fit_pkw", False)) and hasattr(self, "model_advanced_group"):
            self.model_advanced_group.setChecked(True)
        pkw = config.get("pkw")
        if pkw is None and config.get("kw") not in (None, ""):
            pkw = -np.log10(float(config["kw"]))
        if pkw is not None:
            self._set_parameter_value("pKw", f"{float(pkw):.4f}")
        pkw_bounds = config.get("pkw_bounds")
        if isinstance(pkw_bounds, (list, tuple)) and len(pkw_bounds) >= 2:
            self._set_parameter_bounds("pKw", f"{float(pkw_bounds[0]):.4f}", f"{float(pkw_bounds[1]):.4f}")
            if hasattr(self, "spin_pkw_min"):
                self.spin_pkw_min.setValue(float(pkw_bounds[0]))
            if hasattr(self, "spin_pkw_max"):
                self.spin_pkw_max.setValue(float(pkw_bounds[1]))
        if hasattr(self, "chk_fit_pkw"):
            self.chk_fit_pkw.setChecked(bool(config.get("fit_pkw", False)))
            self._set_parameter_fixed("pKw", not bool(config.get("fit_pkw", False)))
        if config.get("initial_volume") not in (None, ""):
            self.spin_initial_volume.setValue(float(config["initial_volume"]))
        if config.get("titrant_concentration") not in (None, ""):
            self.spin_titrant_conc.setValue(float(config["titrant_concentration"]))
        titrant_type = str(config.get("titrant_type") or "")
        if titrant_type:
            idx = self.combo_titrant_type.findData(titrant_type)
            if idx >= 0:
                self.combo_titrant_type.setCurrentIndex(idx)
        strong_mode = str(config.get("strong_ion_mode") or "")
        if strong_mode:
            idx = self.combo_strong_ion.findData(strong_mode)
            if idx >= 0:
                self.combo_strong_ion.setCurrentIndex(idx)
        if config.get("initial_strong_charge") not in (None, ""):
            self.spin_initial_strong_charge.setValue(float(config["initial_strong_charge"]))
        if config.get("titrant_strong_charge") not in (None, ""):
            self.spin_titrant_strong_charge.setValue(float(config["titrant_strong_charge"]))
        if config.get("volume_offset") not in (None, ""):
            self.spin_volume_offset.setValue(float(config["volume_offset"]))
        if isinstance(config.get("pH_bounds"), (list, tuple)) and len(config["pH_bounds"]) >= 2:
            self.spin_ph_min.setValue(float(config["pH_bounds"][0]))
            self.spin_ph_max.setValue(float(config["pH_bounds"][1]))
        if config.get("electrode_e0") not in (None, ""):
            self.edit_e0.setText(str(config["electrode_e0"]))
        if config.get("electrode_slope") not in (None, ""):
            self.edit_slope.setText(str(config["electrode_slope"]))
        self.chk_fit_electrode.setChecked(bool(config.get("fit_electrode", False)))
        self.chk_baseline.setChecked(bool(config.get("baseline", False)))
        if config.get("sigma_pH") not in (None, ""):
            self.spin_sigma_ph.setValue(float(config["sigma_pH"]))
        if config.get("sigma_E") not in (None, ""):
            self.spin_sigma_emf.setValue(float(config["sigma_E"]))
        self._refresh_parameter_table()

    def _set_parameter_value(self, name: str, value: str, *, update_spin: bool = True) -> None:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                _set_table_text(self.parameters_table, row, 2, value)
                self._sync_basic_pka_parameter(name, value=value)
                if update_spin and name.strip().lower() == "pkw" and hasattr(self, "spin_pkw"):
                    try:
                        pkw_for_spin = float(value)
                    except (TypeError, ValueError):
                        pkw_for_spin = self.spin_pkw.value()
                    pkw_for_spin = min(self.spin_pkw.maximum(), max(self.spin_pkw.minimum(), pkw_for_spin))
                    self.spin_pkw.blockSignals(True)
                    self.spin_pkw.setValue(pkw_for_spin)
                    self.spin_pkw.blockSignals(False)
                return

    def _set_parameter_bounds(self, name: str, min_value: str, max_value: str) -> None:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                _set_table_text(self.parameters_table, row, 3, min_value)
                _set_table_text(self.parameters_table, row, 4, max_value)
                self._sync_basic_pka_parameter(name, min_value=min_value, max_value=max_value)
                return

    def _set_parameter_fit(self, name: str, fit: bool) -> None:
        for row in range(self.parameters_table.rowCount()):
            if _table_text(self.parameters_table, row, 0, "").strip().lower() == name.strip().lower():
                fit_widget = self.parameters_table.cellWidget(row, 5)
                if isinstance(fit_widget, QCheckBox):
                    fit_widget.blockSignals(True)
                    fit_widget.setChecked(bool(fit))
                    fit_widget.blockSignals(False)
                self._sync_basic_pka_parameter(name, fit=fit)
                return

    def _set_parameter_fixed(self, name: str, fixed: bool) -> None:
        self._set_parameter_fit(name, not bool(fixed))

    def _sync_basic_pka_parameter(
        self,
        name: str,
        *,
        value: str | None = None,
        min_value: str | None = None,
        max_value: str | None = None,
        fit: bool | None = None,
    ) -> None:
        if not hasattr(self, "table_basic_pka"):
            return
        match = re.search(r"pka[_\-\s]*(\d+)$", str(name or "").strip().lower())
        if match is None:
            return
        idx = int(match.group(1)) - 1
        if not 0 <= idx < self.table_basic_pka.rowCount():
            return
        self.table_basic_pka.blockSignals(True)
        try:
            if value is not None:
                _set_table_text(self.table_basic_pka, idx, 1, value)
            if min_value is not None:
                _set_table_text(self.table_basic_pka, idx, 2, min_value)
            if max_value is not None:
                _set_table_text(self.table_basic_pka, idx, 3, max_value)
            if fit is not None:
                fit_widget = self.table_basic_pka.cellWidget(idx, 4)
                if isinstance(fit_widget, QCheckBox):
                    fit_widget.blockSignals(True)
                    fit_widget.setChecked(bool(fit))
                    fit_widget.blockSignals(False)
        finally:
            self.table_basic_pka.blockSignals(False)

    def reset_tab(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "Cancel the running fit before resetting.")
            return
        self._file_path = ""
        self.lbl_file_status.setText("No file selected")
        self.combo_sheet.clear()
        self.combo_sheet.setEnabled(False)
        self.combo_sheet.setVisible(False)
        self.lbl_sheet.setVisible(False)
        self._current_data_frame = None
        self.preview_text.clear()
        self.combo_volume_column.clear()
        self.combo_signal_column.clear()
        self.combo_volume_unit.setCurrentIndex(self.combo_volume_unit.findData("mL"))
        self.combo_signal_type.setCurrentIndex(self.combo_signal_type.findData("pH"))
        self.table_pot_preview.setRowCount(0)
        self._set_pot_validation_messages([], [])
        self.combo_data_type.setCurrentIndex(0)
        self.combo_model_type.setCurrentIndex(0)
        self.edit_analyte_name.setText("L")
        self.spin_base_charge.setValue(-1)
        self.spin_pka_count.setValue(3)
        self._load_template("simple_monoprotic")
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
        self.edit_e0.setText("0.0")
        self.edit_slope.setText("-59.16")
        self.chk_fit_electrode.setChecked(False)
        self.chk_fit_electrode_basic.setChecked(False)
        self.chk_ideal_nernst.setChecked(True)
        self.chk_baseline.setChecked(False)
        self.spin_pkw.setValue(14.0)
        self.spin_sigma_ph.setValue(1.0)
        self.spin_sigma_emf.setValue(1.0)
        self.model_advanced_group.setChecked(False)
        self._sync_advanced_ui(False)
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
        self.log.append_text("Acid-base tab reset to the default monoprotic template.")
