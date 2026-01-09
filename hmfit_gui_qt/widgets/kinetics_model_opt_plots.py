from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from PySide6.QtCore import Qt, Signal
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
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QHeaderView,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hmfit_gui_qt.workers.errors_worker import ErrorsWorker


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _coerce_float_or_none(value: object) -> float | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


@dataclass(frozen=True)
class KineticsParamState:
    param_names: list[str]
    values: list[float]
    bounds: list[tuple[float | None, float | None]]
    fixed_mask: list[bool]
    log_mask: list[bool]


class KineticsModelOptPlotsWidget(QWidget):
    validate_requested = Signal()
    import_requested = Signal()
    export_requested = Signal()
    reset_requested = Signal()
    process_requested = Signal()
    cancel_requested = Signal()
    save_requested = Signal()
    metadata_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._in_table_update = False
        self._errors_context: dict[str, Any] | None = None
        self._errors_last_output: dict[str, Any] | None = None
        self._errors_worker: ErrorsWorker | None = None
        self._build_ui()

    # ---- UI ----
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)

        # --- Model tab ---
        model_tab = QWidget(self.tabs)
        self.tabs.addTab(model_tab, "Model")
        model_layout = QVBoxLayout(model_tab)

        self.mechanism_edit = QPlainTextEdit(model_tab)
        self.mechanism_edit.setPlaceholderText("Enter kinetic mechanism...")
        model_layout.addWidget(self.mechanism_edit, 2)

        validate_row = QHBoxLayout()
        self.btn_validate = QPushButton("Validate", model_tab)
        self.btn_validate.clicked.connect(self.validate_requested.emit)
        validate_row.addWidget(self.btn_validate)
        validate_row.addStretch(1)
        model_layout.addLayout(validate_row)

        self.mechanism_summary = QPlainTextEdit(model_tab)
        self.mechanism_summary.setReadOnly(True)
        self.mechanism_summary.setMinimumHeight(90)
        model_layout.addWidget(self.mechanism_summary, 1)

        y0_group = QGroupBox("Initial concentrations (y0)", model_tab)
        y0_layout = QVBoxLayout(y0_group)
        self.y0_table = QTableWidget(y0_group)
        self.y0_table.setColumnCount(2)
        self.y0_table.setHorizontalHeaderLabels(["Species", "Initial conc."])
        self.y0_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.y0_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.y0_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.y0_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.y0_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        y0_layout.addWidget(self.y0_table)
        model_layout.addWidget(y0_group)

        fixed_group = QGroupBox("Fixed concentrations", model_tab)
        fixed_layout = QVBoxLayout(fixed_group)
        self.fixed_table = QTableWidget(fixed_group)
        self.fixed_table.setColumnCount(2)
        self.fixed_table.setHorizontalHeaderLabels(["Species", "Initial conc."])
        self.fixed_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.fixed_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.fixed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.fixed_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.fixed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        fixed_layout.addWidget(self.fixed_table)
        model_layout.addWidget(fixed_group)

        # --- Optimization tab ---
        opt_tab = QWidget(self.tabs)
        self.tabs.addTab(opt_tab, "Optimization")
        opt_layout = QVBoxLayout(opt_tab)

        opt_form = QFormLayout()
        self.combo_algorithm = QComboBox(opt_tab)
        self.combo_algorithm.addItems(["least_squares", "trust-region", "multistart"])
        opt_form.addRow("Algorithm", self.combo_algorithm)

        self.combo_backend = QComboBox(opt_tab)
        self.combo_backend.addItems(["SciPy", "JAX"])
        opt_form.addRow("Backend", self.combo_backend)

        self.chk_multistart = QCheckBox("Multi-start", opt_tab)
        self.edit_multistart = QLineEdit(opt_tab)
        self.edit_multistart.setPlaceholderText("runs or seed range (e.g. 10 or 1-10)")
        self.edit_multistart.setEnabled(False)
        self.chk_multistart.toggled.connect(self.edit_multistart.setEnabled)
        multi_widget = QWidget(opt_tab)
        multi_layout = QHBoxLayout(multi_widget)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.addWidget(self.chk_multistart)
        multi_layout.addWidget(self.edit_multistart, 1)
        opt_form.addRow(multi_widget)

        self.chk_nnls = QCheckBox("NNLS (A >= 0)", opt_tab)
        opt_form.addRow(self.chk_nnls)

        opt_layout.addLayout(opt_form)

        opt_layout.addWidget(QLabel("Parameters (Initial Estimates)", opt_tab))
        self.params_table = QTableWidget(opt_tab)
        self.params_table.setColumnCount(6)
        self.params_table.setHorizontalHeaderLabels(["Name", "Value", "Min", "Max", "Fixed", "Log10?"])
        self.params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.params_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.params_table.setAlternatingRowColors(True)
        self.params_table.itemChanged.connect(self._on_params_item_changed)
        opt_layout.addWidget(self.params_table, 1)

        # --- Plots tab ---
        plots_tab = QWidget(self.tabs)
        self.tabs.addTab(plots_tab, "Plots")
        plots_layout = QVBoxLayout(plots_tab)

        plot_group = QGroupBox("Plots", plots_tab)
        plot_form = QGridLayout(plot_group)
        self.combo_preset = QComboBox(plot_group)
        self.combo_preset.addItem("Select a preset...")
        self.combo_x_axis = QComboBox(plot_group)
        self.combo_x_axis.addItem("Select X axis...")
        self.list_y_series = QListWidget(plot_group)
        self.list_y_series.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.combo_vary_along = QComboBox(plot_group)
        self.combo_vary_along.addItem("Auto")

        plot_form.addWidget(QLabel("Preset"), 0, 0)
        plot_form.addWidget(self.combo_preset, 0, 1)
        plot_form.addWidget(QLabel("X axis"), 1, 0)
        plot_form.addWidget(self.combo_x_axis, 1, 1)
        plot_form.addWidget(QLabel("Y series"), 2, 0)
        plot_form.addWidget(self.list_y_series, 2, 1)
        plot_form.addWidget(QLabel("Vary along"), 3, 0)
        plot_form.addWidget(self.combo_vary_along, 3, 1)
        plots_layout.addWidget(plot_group)

        edit_group = QGroupBox("Edit plot", plots_tab)
        edit_grid = QGridLayout(edit_group)
        self.edit_title = QLineEdit(edit_group)
        self.edit_xlabel = QLineEdit(edit_group)
        self.edit_ylabel = QLineEdit(edit_group)
        self.combo_trace = QComboBox(edit_group)
        self.combo_trace.addItem("Select trace...")
        self.edit_trace_name = QLineEdit(edit_group)
        self.btn_apply_plot_edit = QPushButton("Apply", edit_group)
        self.btn_reset_plot_edit = QPushButton("Reset", edit_group)

        edit_grid.addWidget(QLabel("Title"), 0, 0)
        edit_grid.addWidget(self.edit_title, 0, 1)
        edit_grid.addWidget(QLabel("X axis label"), 1, 0)
        edit_grid.addWidget(self.edit_xlabel, 1, 1)
        edit_grid.addWidget(QLabel("Y axis label"), 2, 0)
        edit_grid.addWidget(self.edit_ylabel, 2, 1)
        edit_grid.addWidget(QLabel("Trace"), 3, 0)
        edit_grid.addWidget(self.combo_trace, 3, 1)
        edit_grid.addWidget(QLabel("New trace name"), 4, 0)
        edit_grid.addWidget(self.edit_trace_name, 4, 1)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_apply_plot_edit)
        btns.addWidget(self.btn_reset_plot_edit)
        edit_grid.addLayout(btns, 5, 0, 1, 2)
        plots_layout.addWidget(edit_group)

        export_row = QHBoxLayout()
        self.btn_export_png = QPushButton("Export PNG", plots_tab)
        self.btn_export_png.setEnabled(False)
        self.btn_export_png.setToolTip("Plot export not available yet.")
        self.btn_export_csv = QPushButton("Export CSV", plots_tab)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.setToolTip("Plot export not available yet.")
        export_row.addWidget(self.btn_export_png)
        export_row.addWidget(self.btn_export_csv)
        export_row.addStretch(1)
        plots_layout.addLayout(export_row)
        plots_layout.addStretch(1)

        # --- Errors tab ---
        errors_tab = QWidget(self.tabs)
        self.tabs.addTab(errors_tab, "Errors")
        self._build_errors_tab(errors_tab)

        # --- Actions row ---
        actions_group = QGroupBox("", self)
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_import = QPushButton("Import Config", actions_group)
        self.btn_import.clicked.connect(self.import_requested.emit)
        actions_layout.addWidget(self.btn_import)

        self.btn_export = QPushButton("Export Config", actions_group)
        self.btn_export.clicked.connect(self.export_requested.emit)
        actions_layout.addWidget(self.btn_export)

        actions_layout.addStretch(1)

        self.btn_reset = QPushButton("Reset Calculation", actions_group)
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        actions_layout.addWidget(self.btn_reset)

        self.btn_process = QPushButton("Process", actions_group)
        self.btn_process.clicked.connect(self.process_requested.emit)
        actions_layout.addWidget(self.btn_process)

        self.btn_cancel = QPushButton("Cancel", actions_group)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        self.btn_cancel.setEnabled(False)
        actions_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save results", actions_group)
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.btn_save.setEnabled(False)
        actions_layout.addWidget(self.btn_save)

        outer.addWidget(actions_group)

    # ---- Model helpers ----
    def set_mechanism_text(self, text: str) -> None:
        self.mechanism_edit.setPlainText(str(text or ""))

    def mechanism_text(self) -> str:
        return self.mechanism_edit.toPlainText()

    def set_summary_text(self, text: str) -> None:
        self.mechanism_summary.setPlainText(str(text or ""))

    def set_species_tables(self, dynamic_species: Iterable[str], fixed_species: Iterable[str]) -> None:
        dyn = [str(s) for s in (dynamic_species or [])]
        fix = [str(s) for s in (fixed_species or [])]
        self._populate_species_table(self.y0_table, dyn)
        self._populate_species_table(self.fixed_table, fix)

    def _populate_species_table(self, table: QTableWidget, species: list[str]) -> None:
        table.setRowCount(len(species))
        for row, name in enumerate(species):
            item = QTableWidgetItem(str(name))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 0, item)
            value_spin = QDoubleSpinBox(table)
            value_spin.setDecimals(6)
            value_spin.setRange(-1e12, 1e12)
            value_spin.setSingleStep(0.1)
            value_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_spin.setKeyboardTracking(False)
            value_spin.valueChanged.connect(lambda _value=None: self.metadata_changed.emit())
            table.setCellWidget(row, 1, value_spin)
        table.verticalHeader().setVisible(False)

    def set_y0_values(self, values: dict[str, float] | None) -> None:
        self._set_species_values(self.y0_table, values)

    def set_fixed_conc_values(self, values: dict[str, float] | None) -> None:
        self._set_species_values(self.fixed_table, values)

    def _set_species_values(self, table: QTableWidget, values: dict[str, float] | None) -> None:
        lookup = values or {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            value_widget = table.cellWidget(row, 1)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            value = float(lookup.get(name, 0.0))
            if isinstance(value_widget, QDoubleSpinBox):
                value_widget.blockSignals(True)
                value_widget.setValue(value)
                value_widget.blockSignals(False)
            else:
                value_item = table.item(row, 1)
                if value_item is None:
                    continue
                value_item.setText(f"{value:.6g}")

    def get_y0_values(self) -> dict[str, float]:
        return self._read_species_table(self.y0_table)

    def get_fixed_conc_values(self) -> dict[str, float]:
        return self._read_species_table(self.fixed_table)

    def _read_species_table(self, table: QTableWidget) -> dict[str, float]:
        output: dict[str, float] = {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            value_widget = table.cellWidget(row, 1)
            if isinstance(value_widget, QDoubleSpinBox):
                output[name] = float(value_widget.value())
            else:
                value_item = table.item(row, 1)
                if value_item is None:
                    output[name] = 0.0
                else:
                    output[name] = _coerce_float(value_item.text(), default=0.0)
        return output

    # ---- Parameters ----
    def set_params(self, params: Iterable[str]) -> None:
        names = [str(p) for p in (params or [])]
        self.params_table.setRowCount(len(names))
        for row, name in enumerate(names):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.params_table.setItem(row, 0, name_item)

            for col, default in ((1, "1.0"), (2, "0"), (3, "")):
                item = QTableWidgetItem(default)
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.params_table.setItem(row, col, item)

            fixed_item = QTableWidgetItem()
            fixed_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            fixed_item.setCheckState(Qt.CheckState.Unchecked)
            self.params_table.setItem(row, 4, fixed_item)

            log_item = QTableWidgetItem()
            log_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            log_item.setCheckState(Qt.CheckState.Unchecked)
            self.params_table.setItem(row, 5, log_item)

        self.params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, self.params_table.columnCount()):
            self.params_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

    def get_params_state(self) -> KineticsParamState:
        param_names: list[str] = []
        values: list[float] = []
        bounds: list[tuple[float | None, float | None]] = []
        fixed_mask: list[bool] = []
        log_mask: list[bool] = []

        for row in range(self.params_table.rowCount()):
            name_item = self.params_table.item(row, 0)
            value_item = self.params_table.item(row, 1)
            min_item = self.params_table.item(row, 2)
            max_item = self.params_table.item(row, 3)
            fixed_item = self.params_table.item(row, 4)
            log_item = self.params_table.item(row, 5)

            name = name_item.text().strip() if name_item is not None else ""
            if not name:
                continue
            value = _coerce_float(value_item.text() if value_item else "", default=0.0)
            min_val = _coerce_float_or_none(min_item.text() if min_item else "")
            max_val = _coerce_float_or_none(max_item.text() if max_item else "")
            fixed = bool(fixed_item and fixed_item.checkState() == Qt.CheckState.Checked)
            log_flag = bool(log_item and log_item.checkState() == Qt.CheckState.Checked)

            param_names.append(name)
            values.append(value)
            bounds.append((min_val, max_val))
            fixed_mask.append(fixed)
            log_mask.append(log_flag)

        return KineticsParamState(
            param_names=param_names,
            values=values,
            bounds=bounds,
            fixed_mask=fixed_mask,
            log_mask=log_mask,
        )

    def set_params_state(self, state: KineticsParamState) -> None:
        self.set_params(state.param_names)
        self._in_table_update = True
        try:
            for row, name in enumerate(state.param_names):
                value = state.values[row] if row < len(state.values) else 0.0
                min_val = state.bounds[row][0] if row < len(state.bounds) else None
                max_val = state.bounds[row][1] if row < len(state.bounds) else None
                fixed = state.fixed_mask[row] if row < len(state.fixed_mask) else False
                log_flag = state.log_mask[row] if row < len(state.log_mask) else False

                self.params_table.item(row, 1).setText(str(value))
                self.params_table.item(row, 2).setText("" if min_val is None else str(min_val))
                self.params_table.item(row, 3).setText("" if max_val is None else str(max_val))
                self.params_table.item(row, 4).setCheckState(
                    Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked
                )
                self.params_table.item(row, 5).setCheckState(
                    Qt.CheckState.Checked if log_flag else Qt.CheckState.Unchecked
                )
        finally:
            self._in_table_update = False

    def _on_params_item_changed(self, item: QTableWidgetItem) -> None:
        if self._in_table_update:
            return
        row = item.row()
        if item.column() != 4:
            return
        fixed = item.checkState() == Qt.CheckState.Checked
        if not fixed:
            return
        value_item = self.params_table.item(row, 1)
        min_item = self.params_table.item(row, 2)
        max_item = self.params_table.item(row, 3)
        if value_item is None or min_item is None or max_item is None:
            return
        value_text = value_item.text().strip()
        min_item.setText(value_text)
        max_item.setText(value_text)

    # ---- Multi-start ----
    def get_multi_start(self) -> tuple[int, list[int] | None]:
        if not self.chk_multistart.isChecked():
            return 1, None
        raw = (self.edit_multistart.text() or "").strip()
        if not raw:
            return 10, None
        if "-" in raw:
            try:
                a_str, b_str = raw.split("-", 1)
                a = int(a_str.strip())
                b = int(b_str.strip())
                if a > b:
                    a, b = b, a
                seeds = list(range(a, b + 1))
                return len(seeds), seeds
            except Exception:
                pass
        try:
            runs = int(raw)
            return max(1, runs), None
        except Exception:
            return 10, None

    def set_multi_start(self, runs: int | None, seeds: list[int] | None = None) -> None:
        runs_val = 1
        try:
            if runs is not None:
                runs_val = int(runs)
        except Exception:
            runs_val = 1

        seeds_list: list[int] | None = None
        if seeds is not None and isinstance(seeds, (list, tuple)):
            tmp: list[int] = []
            for s in seeds:
                try:
                    tmp.append(int(s))
                except Exception:
                    continue
            if tmp:
                seeds_list = tmp

        if seeds_list:
            seq = list(seeds_list)
            is_range = len(seq) >= 2 and all(seq[i] + 1 == seq[i + 1] for i in range(len(seq) - 1))
            if is_range:
                text = f"{seq[0]}-{seq[-1]}"
            elif len(seq) == 1:
                text = str(seq[0])
            else:
                text = str(max(runs_val, len(seq)))
            self.chk_multistart.setChecked(True)
            self.edit_multistart.setText(text)
            return

        if runs_val > 1:
            self.chk_multistart.setChecked(True)
            self.edit_multistart.setText(str(runs_val))
        else:
            self.chk_multistart.setChecked(False)
            self.edit_multistart.setText("")

    # ---- Errors tab ----
    def _build_errors_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        controls_grid = QGridLayout()
        controls_grid.setColumnStretch(1, 1)
        controls_grid.setColumnStretch(3, 1)
        controls_grid.setColumnStretch(5, 1)
        controls_grid.setColumnStretch(7, 1)

        controls_grid.addWidget(QLabel("Error method"), 0, 0)
        self.combo_error_method = QComboBox(parent)
        self.combo_error_method.addItem("Analytical (covariance)", "analytic")
        self.combo_error_method.addItem("Bootstrap (linearized)", "bootstrap_linear")
        self.combo_error_method.addItem("Bootstrap (one-step LM)", "bootstrap_onestep")
        self.combo_error_method.addItem("Bootstrap (full refit)", "bootstrap_full_refit")
        self.combo_error_method.currentIndexChanged.connect(self._on_error_method_changed)
        controls_grid.addWidget(self.combo_error_method, 0, 1, 1, 7)

        controls_grid.addWidget(QLabel("B (replicates)"), 1, 0)
        self.spin_error_b = QSpinBox(parent)
        self.spin_error_b.setRange(50, 5000)
        self.spin_error_b.setValue(500)
        controls_grid.addWidget(self.spin_error_b, 1, 1)

        controls_grid.addWidget(QLabel("Seed"), 1, 2)
        self.edit_error_seed = QLineEdit(parent)
        self.edit_error_seed.setPlaceholderText("optional")
        controls_grid.addWidget(self.edit_error_seed, 1, 3)

        controls_grid.addWidget(QLabel("Bootstrap type"), 1, 4)
        self.combo_error_bootstrap = QComboBox(parent)
        self.combo_error_bootstrap.addItem("Wild", "wild")
        self.combo_error_bootstrap.addItem("Residual", "residual")
        controls_grid.addWidget(self.combo_error_bootstrap, 1, 5)

        controls_grid.addWidget(QLabel("Wild type"), 1, 6)
        self.combo_error_wild = QComboBox(parent)
        self.combo_error_wild.addItem("Rademacher (+/-1)", "rademacher")
        self.combo_error_wild.addItem("Mammen", "mammen")
        controls_grid.addWidget(self.combo_error_wild, 1, 7)

        controls_grid.addWidget(QLabel("LM lambda"), 2, 0)
        self.spin_error_lambda = QDoubleSpinBox(parent)
        self.spin_error_lambda.setDecimals(6)
        self.spin_error_lambda.setRange(0.0, 1e6)
        self.spin_error_lambda.setSingleStep(1e-3)
        self.spin_error_lambda.setValue(1e-3)
        controls_grid.addWidget(self.spin_error_lambda, 2, 1)

        self.lbl_error_max_iter = QLabel("Max iter", parent)
        controls_grid.addWidget(self.lbl_error_max_iter, 2, 2)
        self.spin_error_max_iter = QSpinBox(parent)
        self.spin_error_max_iter.setRange(5, 200)
        self.spin_error_max_iter.setValue(30)
        controls_grid.addWidget(self.spin_error_max_iter, 2, 3)

        self.lbl_error_tol = QLabel("Tol", parent)
        controls_grid.addWidget(self.lbl_error_tol, 2, 4)
        self.spin_error_tol = QDoubleSpinBox(parent)
        self.spin_error_tol.setDecimals(12)
        self.spin_error_tol.setRange(1e-12, 1e-2)
        self.spin_error_tol.setSingleStep(1e-8)
        self.spin_error_tol.setValue(1e-8)
        controls_grid.addWidget(self.spin_error_tol, 2, 5)

        self.lbl_error_fail_policy = QLabel("Fail policy", parent)
        controls_grid.addWidget(self.lbl_error_fail_policy, 2, 6)
        self.combo_error_fail_policy = QComboBox(parent)
        self.combo_error_fail_policy.addItem("Skip failed replicates", "skip")
        self.combo_error_fail_policy.addItem("Stop on first failure", "stop")
        controls_grid.addWidget(self.combo_error_fail_policy, 2, 7)

        layout.addLayout(controls_grid)

        self.lbl_error_audit_warning = QLabel(
            "Warning: Bootstrap (full refit) can take several minutes. "
            "Use Cancel to stop the computation if needed.",
            parent,
        )
        self.lbl_error_audit_warning.setWordWrap(True)
        self.lbl_error_audit_warning.setVisible(False)
        layout.addWidget(self.lbl_error_audit_warning)

        flags_row = QHBoxLayout()
        self.chk_error_ci_16_84 = QCheckBox("Include 16/84 percentiles", parent)
        self.chk_error_ci_16_84.setChecked(False)
        self.chk_error_ci_16_84.toggled.connect(self._apply_errors_ci_visibility)
        flags_row.addWidget(self.chk_error_ci_16_84)

        self.chk_error_show_corr = QCheckBox("Show correlation matrix", parent)
        self.chk_error_show_corr.setChecked(True)
        self.chk_error_show_corr.toggled.connect(self._apply_errors_corr_visibility)
        flags_row.addWidget(self.chk_error_show_corr)

        flags_row.addStretch(1)
        self.btn_error_compute = QPushButton("Compute", parent)
        self.btn_error_compute.clicked.connect(self._on_error_compute_clicked)
        flags_row.addWidget(self.btn_error_compute)
        self.btn_error_export = QPushButton("Export XLSX", parent)
        self.btn_error_export.setEnabled(False)
        self.btn_error_export.clicked.connect(self._on_error_export_clicked)
        flags_row.addWidget(self.btn_error_export)
        layout.addLayout(flags_row)

        progress_row = QHBoxLayout()
        self.lbl_error_status = QLabel("", parent)
        progress_row.addWidget(self.lbl_error_status)
        self.progress_error = QProgressBar(parent)
        self.progress_error.setVisible(False)
        self.progress_error.setTextVisible(True)
        self.progress_error.setFormat("Working...")
        progress_row.addWidget(self.progress_error, 1)
        self.btn_error_cancel = QPushButton("Cancel", parent)
        self.btn_error_cancel.setEnabled(False)
        self.btn_error_cancel.clicked.connect(self._on_error_cancel_clicked)
        progress_row.addWidget(self.btn_error_cancel)
        layout.addLayout(progress_row)

        layout.addWidget(QLabel("Results", parent))
        self.table_error_results = QTableWidget(parent)
        self.table_error_results.setColumnCount(9)
        self.table_error_results.setHorizontalHeaderLabels(
            [
                "Parameter",
                "Estimate",
                "Median",
                "CI 2.5",
                "CI 97.5",
                "CI 16",
                "CI 84",
                "SE",
                "%err",
            ]
        )
        self.table_error_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_error_results.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table_error_results.setAlternatingRowColors(True)
        self.table_error_results.verticalHeader().setVisible(False)
        header = self.table_error_results.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, self.table_error_results.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_error_results, 1)

        self.lbl_error_corr = QLabel("Correlation matrix", parent)
        layout.addWidget(self.lbl_error_corr)
        self.table_error_corr = QTableWidget(parent)
        self.table_error_corr.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_error_corr.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table_error_corr.setAlternatingRowColors(True)
        layout.addWidget(self.table_error_corr)

        layout.addWidget(QLabel("Summary", parent))
        self.text_error_summary = QPlainTextEdit(parent)
        self.text_error_summary.setReadOnly(True)
        self.text_error_summary.setMinimumHeight(90)
        layout.addWidget(self.text_error_summary)

        self._apply_errors_ci_visibility()
        self._apply_errors_corr_visibility()
        self._on_error_method_changed()

    # ---- Errors tab ----
    def set_errors_context(self, context: dict[str, Any] | None, *, auto_compute: bool = False) -> None:
        self._errors_context = context
        self._errors_last_output = None
        self.btn_error_export.setEnabled(False)
        self._clear_errors_tables()
        if context and auto_compute:
            self.combo_error_method.setCurrentIndex(0)
            self._on_error_compute_clicked()

    def _on_error_method_changed(self) -> None:
        method = str(self.combo_error_method.currentData() or "")
        is_bootstrap = method.startswith("bootstrap")
        is_onestep = method == "bootstrap_onestep"
        is_full_refit = method == "bootstrap_full_refit"
        self.spin_error_b.setEnabled(is_bootstrap)
        self.combo_error_bootstrap.setEnabled(is_bootstrap)
        self.combo_error_wild.setEnabled(is_bootstrap)
        self.spin_error_lambda.setEnabled(is_onestep)
        self.lbl_error_max_iter.setVisible(is_full_refit)
        self.spin_error_max_iter.setVisible(is_full_refit)
        self.lbl_error_tol.setVisible(is_full_refit)
        self.spin_error_tol.setVisible(is_full_refit)
        self.lbl_error_fail_policy.setVisible(is_full_refit)
        self.combo_error_fail_policy.setVisible(is_full_refit)
        self.spin_error_max_iter.setEnabled(is_full_refit)
        self.spin_error_tol.setEnabled(is_full_refit)
        self.combo_error_fail_policy.setEnabled(is_full_refit)
        self.lbl_error_audit_warning.setVisible(is_full_refit)

        if is_full_refit and self.spin_error_b.value() in (300, 500):
            self.spin_error_b.setValue(50)
        if is_onestep and self.spin_error_b.value() in (50, 500):
            self.spin_error_b.setValue(300)
        if (not is_onestep) and (not is_full_refit) and self.spin_error_b.value() in (50, 300):
            self.spin_error_b.setValue(500)

    def _apply_errors_ci_visibility(self) -> None:
        show = bool(self.chk_error_ci_16_84.isChecked())
        self.table_error_results.setColumnHidden(5, not show)
        self.table_error_results.setColumnHidden(6, not show)

    def _apply_errors_corr_visibility(self) -> None:
        show = bool(self.chk_error_show_corr.isChecked())
        self.lbl_error_corr.setVisible(show)
        self.table_error_corr.setVisible(show)

    def _parse_error_seed(self) -> int | None:
        raw = str(self.edit_error_seed.text() or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError("Seed must be an integer.") from exc

    def _collect_errors_options(self) -> dict[str, Any]:
        method = str(self.combo_error_method.currentData() or "analytic")
        seed = self._parse_error_seed()
        wild = str(self.combo_error_wild.currentData() or "rademacher")
        bootstrap_kind = str(self.combo_error_bootstrap.currentData() or "wild")
        lam = float(self.spin_error_lambda.value())
        B = int(self.spin_error_b.value())
        include_16_84 = bool(self.chk_error_ci_16_84.isChecked())
        max_iter = int(self.spin_error_max_iter.value())
        tol = float(self.spin_error_tol.value())
        fail_policy = str(self.combo_error_fail_policy.currentData() or "skip")
        return {
            "method": method,
            "seed": seed,
            "wild": wild,
            "bootstrap_kind": bootstrap_kind,
            "lam": lam,
            "B": B,
            "include_16_84": include_16_84,
            "max_iter": max_iter,
            "tol": tol,
            "fail_policy": fail_policy,
        }

    def _on_error_compute_clicked(self) -> None:
        if self._errors_worker is not None:
            QMessageBox.information(self, "Errors", "An errors computation is already running.")
            return
        if not self._errors_context:
            QMessageBox.warning(self, "Errors", "Run a fit first.")
            return
        try:
            options = self._collect_errors_options()
        except Exception as exc:
            QMessageBox.warning(self, "Errors", str(exc))
            return
        payload = {"ctx": dict(self._errors_context), "options": options}
        self._start_errors_worker(payload)

    def _start_errors_worker(self, payload: dict[str, Any]) -> None:
        self._errors_last_output = None
        self.btn_error_export.setEnabled(False)
        self._set_errors_busy(True, payload.get("options") or {})
        self._errors_worker = ErrorsWorker(self._compute_errors_payload, payload=payload, parent=self)
        worker_thread = self._errors_worker.thread()
        worker_thread.finished.connect(self._on_error_thread_finished)
        self._errors_worker.progress.connect(self._on_error_progress)
        self._errors_worker.result.connect(self._on_error_worker_result)
        self._errors_worker.error.connect(self._on_error_worker_error)
        self._errors_worker.cancelled.connect(self._on_error_worker_cancelled)
        self._errors_worker.finished.connect(self._on_error_worker_finished)
        self._errors_worker.start()

    def _compute_errors_payload(
        self,
        payload: dict[str, Any],
        *,
        progress_cb=None,
        cancel_cb=None,
    ) -> dict[str, Any]:
        ctx = payload.get("ctx") or {}
        options = payload.get("options") or {}
        return self._compute_errors_from_context(
            ctx,
            options=options,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    def _set_errors_busy(self, running: bool, options: dict[str, Any]) -> None:
        controls = [
            self.combo_error_method,
            self.spin_error_b,
            self.edit_error_seed,
            self.combo_error_bootstrap,
            self.combo_error_wild,
            self.spin_error_lambda,
            self.spin_error_max_iter,
            self.spin_error_tol,
            self.combo_error_fail_policy,
            self.chk_error_ci_16_84,
            self.chk_error_show_corr,
        ]
        for widget in controls:
            widget.setEnabled(not running)
        self.btn_error_compute.setEnabled(not running)
        self.btn_error_cancel.setEnabled(running)
        self.progress_error.setVisible(running)
        self.lbl_error_status.setVisible(running)
        if running:
            self._configure_errors_progress(options)
            self.lbl_error_status.setText("Computing errors...")
        else:
            self.lbl_error_status.setText("")
            self.progress_error.setVisible(False)

    def _configure_errors_progress(self, options: dict[str, Any]) -> None:
        method = str(options.get("method") or "")
        if method == "bootstrap_full_refit":
            total = int(options.get("B") or 0)
            total = max(total, 1)
            self.progress_error.setRange(0, total)
            self.progress_error.setValue(0)
            self.progress_error.setFormat("Bootstrap %v/%m")
        else:
            self.progress_error.setRange(0, 0)
            self.progress_error.setFormat("Working...")

    def _on_error_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            current = payload.get("current")
            total = payload.get("total")
            n_success = payload.get("n_success")
            n_fail = payload.get("n_fail")
            if total is not None:
                try:
                    total_int = int(total)
                    current_int = int(current or 0)
                except Exception:
                    return
                if total_int > 0:
                    self.progress_error.setRange(0, total_int)
                    self.progress_error.setValue(current_int)
                    if n_success is not None and n_fail is not None:
                        self.progress_error.setFormat(
                            f"Bootstrap {current_int}/{total_int} (ok {int(n_success)}, fail {int(n_fail)})"
                        )
            return
        if isinstance(payload, str):
            self.lbl_error_status.setText(str(payload))

    def _on_error_cancel_clicked(self) -> None:
        if self._errors_worker is None:
            return
        self._errors_worker.request_cancel()
        self.lbl_error_status.setText("Cancelling...")

    def _on_error_worker_result(self, output: dict[str, Any]) -> None:
        self._apply_errors_output(output)

    def _on_error_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Errors", str(message))

    def _on_error_worker_cancelled(self) -> None:
        self.text_error_summary.setPlainText("Cancelled by user.")
        self.lbl_error_status.setText("Cancelled.")

    def _on_error_worker_finished(self) -> None:
        self._set_errors_busy(False, {})

    def _on_error_thread_finished(self) -> None:
        self._errors_worker = None

    def _compute_errors_from_context(
        self,
        ctx: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
        progress_cb=None,
        cancel_cb=None,
    ) -> dict[str, Any]:
        from hmfit_core.utils.errors import (
            bootstrap_full_refit,
            bootstrap_linearized_wild,
            wild_multipliers,
        )

        options = options or {}
        method = str(options.get("method") or "analytic")
        seed = options.get("seed")
        wild = str(options.get("wild") or "rademacher")
        bootstrap_kind = str(options.get("bootstrap_kind") or "wild")
        lam = float(options.get("lam") or 1e-3)
        B = int(options.get("B") or 0)
        include_16_84 = bool(options.get("include_16_84"))
        max_iter = int(options.get("max_iter") or 30)
        tol = float(options.get("tol") or 1e-8)
        fail_policy = str(options.get("fail_policy") or "skip")

        param_names = list(ctx.get("param_names") or [])
        k_hat_raw = ctx.get("k_hat")
        k_hat = np.asarray(k_hat_raw if k_hat_raw is not None else [], dtype=float).ravel()
        if k_hat.size == 0:
            raise ValueError("Missing fitted parameters.")

        jac_raw = ctx.get("jac")
        jac = np.asarray(jac_raw if jac_raw is not None else [], dtype=float)
        residuals_raw = ctx.get("residuals")
        residuals = np.asarray(
            residuals_raw if residuals_raw is not None else [], dtype=float
        ).ravel()
        nobs = residuals.size
        p = k_hat.size
        if jac.size == 0 or jac.shape[0] != nobs or jac.shape[1] != p:
            jac = np.asarray(jac_raw if jac_raw is not None else [], dtype=float)

        log_params = set(ctx.get("log_params") or [])
        jac_lin = jac
        if jac.size and log_params and param_names:
            scale = np.ones(p, dtype=float)
            for idx, name in enumerate(param_names):
                if name in log_params:
                    scale[idx] = np.log(10.0) * max(float(k_hat[idx]), 1e-300)
            jac_lin = jac / scale[None, :]

        cov = None
        se = None
        corr = None
        if method == "analytic":
            cov = ctx.get("covariance")
            if cov is None and jac_lin.size:
                s2 = ctx.get("sigma2")
                if s2 is None:
                    dof = max(nobs - p, 1)
                    s2 = np.sum(residuals**2) / float(dof)
                cov = float(s2) * np.linalg.pinv(jac_lin.T @ jac_lin)
            if cov is not None:
                cov = np.asarray(cov, dtype=float)
                se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
                denom = np.outer(se, se)
                corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)
        else:
            if jac_lin.size == 0:
                raise ValueError("Jacobian not available for bootstrap.")

        samples = None
        boot_summary = None
        n_success = None
        n_fail = None

        if method == "bootstrap_linear":
            boot = bootstrap_linearized_wild(
                k_hat,
                jac_lin.T,
                residuals,
                B,
                seed=seed,
                wild=wild,
                rcond=1e-10,
                ridge=0.0,
            )
            samples = boot["samples"]
        elif method == "bootstrap_onestep":
            rng = np.random.default_rng(seed)
            samples = np.zeros((B, p), dtype=float)
            jt_j = jac_lin.T @ jac_lin + float(lam) * np.eye(p)
            for b in range(B):
                if bootstrap_kind == "residual":
                    idx = rng.integers(0, nobs, size=nobs)
                    r_star = residuals[idx]
                else:
                    v = wild_multipliers(rng, nobs, kind=wild)
                    r_star = residuals * v
                try:
                    dk = np.linalg.solve(jt_j, jac_lin.T @ r_star)
                except np.linalg.LinAlgError:
                    dk = np.linalg.pinv(jt_j) @ (jac_lin.T @ r_star)
                samples[b, :] = k_hat + dk
        elif method == "bootstrap_full_refit":
            if "refit_fn" not in ctx:
                raise ValueError("Missing refit function for full refit bootstrap.")
            refit_fn = ctx.get("refit_fn")
            if not callable(refit_fn):
                raise ValueError("Invalid refit function for full refit bootstrap.")

            D_hat_raw = ctx.get("D_hat")
            D_obs_raw = ctx.get("D_obs")
            if isinstance(D_hat_raw, (list, tuple)) and isinstance(D_obs_raw, (list, tuple)):
                if len(D_hat_raw) != len(D_obs_raw):
                    raise ValueError("D_hat list length does not match D_obs.")
                d_hat_list = [np.asarray(arr, dtype=float) for arr in D_hat_raw]
                d_obs_list = [np.asarray(arr, dtype=float) for arr in D_obs_raw]
                if not d_hat_list or not d_obs_list:
                    raise ValueError("Missing fitted data for bootstrap.")

                r_hat_list = []
                for hat, obs in zip(d_hat_list, d_obs_list, strict=True):
                    if hat.shape != obs.shape:
                        raise ValueError("D_hat shape does not match D_obs.")
                    r_hat_list.append(obs - hat)

                def make_data_star_fn(rng, wild=wild):
                    data_star = []
                    for r_hat, d_hat in zip(r_hat_list, d_hat_list, strict=True):
                        if bootstrap_kind == "residual":
                            flat = r_hat.ravel()
                            idx = rng.integers(0, flat.size, size=flat.size)
                            r_star = flat[idx].reshape(r_hat.shape)
                        else:
                            v = wild_multipliers(rng, r_hat.size, kind=wild).reshape(r_hat.shape)
                            r_star = r_hat * v
                        data_star.append(d_hat + r_star)
                    return data_star
            else:
                D_hat = np.asarray(D_hat_raw if D_hat_raw is not None else [], dtype=float)
                D_obs = np.asarray(D_obs_raw if D_obs_raw is not None else [], dtype=float)
                if D_hat.size == 0 or D_obs.size == 0:
                    raise ValueError("Missing fitted data for bootstrap.")
                if D_hat.shape != D_obs.shape:
                    raise ValueError("D_hat shape does not match D_obs.")

                r_hat = D_obs - D_hat

                def make_data_star_fn(rng, wild=wild):
                    if bootstrap_kind == "residual":
                        flat = r_hat.ravel()
                        idx = rng.integers(0, flat.size, size=flat.size)
                        r_star = flat[idx].reshape(r_hat.shape)
                    else:
                        v = wild_multipliers(rng, r_hat.size, kind=wild).reshape(r_hat.shape)
                        r_star = r_hat * v
                    return D_hat + r_star

            boot_summary = bootstrap_full_refit(
                k_hat,
                make_data_star_fn,
                refit_fn,
                B,
                seed=seed,
                wild=wild,
                max_iter=max_iter,
                tol=tol,
                fail_policy=fail_policy,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            samples = boot_summary["samples"]
            n_success = boot_summary.get("n_success")
            n_fail = boot_summary.get("n_fail")
            corr = boot_summary.get("corr")

        if method == "analytic":
            if se is None:
                se = np.zeros_like(k_hat)
            z95 = 1.96
            z68 = 1.0
            median = k_hat.copy()
            ci_2p5 = k_hat - z95 * se
            ci_97p5 = k_hat + z95 * se
            ci_16 = k_hat - z68 * se if include_16_84 else None
            ci_84 = k_hat + z68 * se if include_16_84 else None
            perc_err = 100.0 * se / np.maximum(np.abs(k_hat), 1e-300)
        else:
            if samples is None or samples.size == 0:
                raise ValueError("Bootstrap samples are empty.")
            median = np.median(samples, axis=0)
            ci_2p5 = np.percentile(samples, 2.5, axis=0)
            ci_97p5 = np.percentile(samples, 97.5, axis=0)
            ci_16 = np.percentile(samples, 16.0, axis=0) if include_16_84 else None
            ci_84 = np.percentile(samples, 84.0, axis=0) if include_16_84 else None
            ddof = 1 if samples.shape[0] > 1 else 0
            se = np.std(samples, axis=0, ddof=ddof)
            perc_err = np.zeros_like(se)
            if np.any(se > 0):
                perc_err = 100.0 * se / np.maximum(np.abs(median), 1e-300)
            if corr is None and samples.shape[0] >= 2:
                corr = np.corrcoef(samples, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)
                for i in range(corr.shape[0]):
                    corr[i, i] = 1.0

        summary = self._build_errors_summary(
            method=method,
            B=B,
            seed=seed,
            wild=wild,
            lam=lam,
            bootstrap_kind=bootstrap_kind,
            max_iter=max_iter,
            tol=tol,
            fail_policy=fail_policy,
            n_success=n_success,
            n_fail=n_fail,
        )

        return {
            "param_names": param_names,
            "k_hat": k_hat,
            "median": median,
            "ci_2p5": ci_2p5,
            "ci_97p5": ci_97p5,
            "ci_16": ci_16,
            "ci_84": ci_84,
            "se": se,
            "perc_err": perc_err,
            "corr": corr,
            "summary": summary,
            "method": method,
            "B": B,
            "seed": seed,
            "wild": wild,
            "lam": lam,
            "max_iter": max_iter,
            "tol": tol,
            "n_success": n_success,
            "n_fail": n_fail,
        }

    def _build_errors_summary(
        self,
        *,
        method: str,
        B: int,
        seed: int | None,
        wild: str,
        lam: float,
        bootstrap_kind: str,
        max_iter: int | None = None,
        tol: float | None = None,
        fail_policy: str | None = None,
        n_success: int | None = None,
        n_fail: int | None = None,
    ) -> str:
        method_labels = {
            "analytic": "Analytical (covariance)",
            "bootstrap_linear": "Bootstrap (linearized)",
            "bootstrap_onestep": "Bootstrap (one-step LM)",
            "bootstrap_full_refit": "Bootstrap (full refit)",
        }
        lines = [f"Method: {method_labels.get(method, method)}"]
        if method != "analytic":
            seed_txt = "random" if seed is None else str(seed)
            if method == "bootstrap_full_refit":
                lines.append("method=full_refit")
                lines.append(f"B={B}, seed={seed_txt}, bootstrap={bootstrap_kind}, wild={wild}")
                if max_iter is not None and tol is not None:
                    lines.append(f"max_iter={int(max_iter)}, tol={float(tol):.3g}")
                if fail_policy:
                    lines.append(f"fail_policy={fail_policy}")
                if n_success is not None and n_fail is not None:
                    lines.append(f"n_success={int(n_success)}, n_fail={int(n_fail)}")
            else:
                lines.append(f"B={B}, seed={seed_txt}, bootstrap={bootstrap_kind}, wild={wild}, lambda={lam:g}")
        return "\n".join(lines)

    def _apply_errors_output(self, output: dict[str, Any]) -> None:
        param_names = list(output.get("param_names") or [])
        k_hat_raw = output.get("k_hat")
        k_hat = np.asarray(k_hat_raw if k_hat_raw is not None else [], dtype=float)
        median_raw = output.get("median")
        median = np.asarray(median_raw if median_raw is not None else [], dtype=float)
        ci_2p5_raw = output.get("ci_2p5")
        ci_2p5 = np.asarray(ci_2p5_raw if ci_2p5_raw is not None else [], dtype=float)
        ci_97p5_raw = output.get("ci_97p5")
        ci_97p5 = np.asarray(ci_97p5_raw if ci_97p5_raw is not None else [], dtype=float)
        ci_16 = output.get("ci_16")
        ci_84 = output.get("ci_84")
        se_raw = output.get("se")
        se = np.asarray(se_raw if se_raw is not None else [], dtype=float)
        perc_err_raw = output.get("perc_err")
        perc_err = np.asarray(perc_err_raw if perc_err_raw is not None else [], dtype=float)
        corr = output.get("corr")

        self.table_error_results.setRowCount(len(param_names))
        for row, name in enumerate(param_names):
            self._set_error_cell(row, 0, str(name), align_left=True)
            self._set_error_cell(row, 1, self._fmt_num(k_hat, row))
            self._set_error_cell(row, 2, self._fmt_num(median, row))
            self._set_error_cell(row, 3, self._fmt_num(ci_2p5, row))
            self._set_error_cell(row, 4, self._fmt_num(ci_97p5, row))
            self._set_error_cell(row, 5, self._fmt_num(ci_16, row))
            self._set_error_cell(row, 6, self._fmt_num(ci_84, row))
            self._set_error_cell(row, 7, self._fmt_num(se, row))
            self._set_error_cell(row, 8, self._fmt_num(perc_err, row))

        if corr is not None and self.chk_error_show_corr.isChecked():
            self._populate_corr_table(np.asarray(corr, dtype=float), param_names)
        else:
            self._clear_corr_table()

        summary = str(output.get("summary") or "")
        self.text_error_summary.setPlainText(summary)

        self._errors_last_output = output
        self.btn_error_export.setEnabled(True)

    def _set_error_cell(self, row: int, col: int, text: str, *, align_left: bool = False) -> None:
        item = QTableWidgetItem(text)
        if align_left:
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
        else:
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        self.table_error_results.setItem(row, col, item)

    def _fmt_num(self, arr, idx: int) -> str:
        if arr is None:
            return ""
        try:
            val = float(np.asarray(arr).ravel()[idx])
        except Exception:
            return ""
        if not np.isfinite(val):
            return ""
        return f"{val:.6g}"

    def _populate_corr_table(self, corr: np.ndarray, param_names: list[str]) -> None:
        if corr.size == 0:
            self._clear_corr_table()
            return
        p = corr.shape[0]
        self.table_error_corr.setRowCount(p)
        self.table_error_corr.setColumnCount(p)
        self.table_error_corr.setHorizontalHeaderLabels(param_names)
        self.table_error_corr.setVerticalHeaderLabels(param_names)
        self.table_error_corr.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(p):
            for j in range(p):
                val = corr[i, j]
                txt = "" if not np.isfinite(val) else f"{val:.3f}"
                item = QTableWidgetItem(txt)
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.table_error_corr.setItem(i, j, item)

    def _clear_corr_table(self) -> None:
        self.table_error_corr.clear()
        self.table_error_corr.setRowCount(0)
        self.table_error_corr.setColumnCount(0)

    def _clear_errors_tables(self) -> None:
        self.table_error_results.setRowCount(0)
        self._clear_corr_table()
        self.text_error_summary.clear()

    def _on_error_export_clicked(self) -> None:
        if not self._errors_last_output:
            QMessageBox.information(self, "Errors", "Compute errors first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export errors", "errors.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not str(path).lower().endswith(".xlsx"):
            path = f"{path}.xlsx"
        try:
            import pandas as pd

            output = self._errors_last_output
            param_names = list(output.get("param_names") or [])
            df = pd.DataFrame(
                {
                    "Parameter": param_names,
                    "Estimate": output.get("k_hat"),
                    "Median": output.get("median"),
                    "CI 2.5": output.get("ci_2p5"),
                    "CI 97.5": output.get("ci_97p5"),
                    "CI 16": output.get("ci_16"),
                    "CI 84": output.get("ci_84"),
                    "SE": output.get("se"),
                    "%err": output.get("perc_err"),
                }
            )
            method_raw = str(output.get("method") or "")
            seed_raw = output.get("seed") if method_raw != "analytic" else None
            seed_val = "random" if seed_raw is None and method_raw != "analytic" else seed_raw
            b_val = output.get("B") if method_raw != "analytic" else None
            meta_rows = [
                ("method", method_raw),
                ("B", b_val),
                ("seed", seed_val),
                ("wild_type", output.get("wild")),
                ("lambda", output.get("lam")),
                ("max_iter", output.get("max_iter")),
                ("tol", output.get("tol")),
                ("n_success", output.get("n_success")),
                ("n_fail", output.get("n_fail")),
            ]
            df_meta = pd.DataFrame(meta_rows, columns=["Key", "Value"])

            corr = output.get("corr")
            if corr is None:
                df_corr = pd.DataFrame({"Correlation": ["Not available (n<2)"]})
            else:
                df_corr = pd.DataFrame(corr, index=param_names, columns=param_names)

            with pd.ExcelWriter(path) as writer:
                df.to_excel(writer, sheet_name="Results", index=False)
                df_corr.to_excel(writer, sheet_name="Correlation")
                df_meta.to_excel(writer, sheet_name="Meta", index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
