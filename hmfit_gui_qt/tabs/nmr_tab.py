from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hmfit_core.api import run_nmr_fit
from hmfit_gui_qt.widgets.file_selectors import FileSelector
from hmfit_gui_qt.widgets.log_console import LogConsole
from hmfit_gui_qt.widgets.mpl_canvas import MplCanvas, NavigationToolbar
from hmfit_gui_qt.workers.fit_worker import FitWorker


def _list_excel_sheets(file_path: str) -> list[str]:
    import pandas as pd

    xls = pd.ExcelFile(file_path)
    return [str(s) for s in xls.sheet_names]


def _list_excel_columns(file_path: str, sheet_name: str) -> list[str]:
    import pandas as pd

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, nrows=0)
    return [str(c) for c in df.columns]


def _ensure_float_or_none(text: str) -> float | None:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class NMRTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._worker: FitWorker | None = None
        self._last_result: dict[str, Any] | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        self._vertical_split = QSplitter(Qt.Orientation.Vertical, self)
        outer.addWidget(self._vertical_split)

        top = QWidget(self._vertical_split)
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self._horizontal_split = QSplitter(Qt.Orientation.Horizontal, top)
        top_layout.addWidget(self._horizontal_split)

        controls_scroll = QScrollArea(self._horizontal_split)
        controls_scroll.setWidgetResizable(True)
        controls_container = QWidget(controls_scroll)
        self._controls_container = controls_container
        controls_scroll.setWidget(controls_container)
        self._controls_layout = QVBoxLayout(controls_container)

        self._build_controls()

        plot_container = QWidget(self._horizontal_split)
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        plot_top = QHBoxLayout()
        plot_top.addWidget(QLabel("Plot:", plot_container))
        self.plot_selector = QComboBox(plot_container)
        self.plot_selector.currentIndexChanged.connect(self._on_plot_selection_changed)
        plot_top.addWidget(self.plot_selector, 1)
        plot_layout.addLayout(plot_top)

        self.canvas = MplCanvas(plot_container)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(NavigationToolbar(self.canvas, plot_container))

        self._horizontal_split.setStretchFactor(0, 0)
        self._horizontal_split.setStretchFactor(1, 1)

        log_container = QWidget(self._vertical_split)
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_top = QHBoxLayout()
        log_top.addWidget(QLabel("Log:", log_container))
        log_top.addStretch(1)
        self.btn_clear_log = QPushButton("Clear", log_container)
        self.btn_clear_log.clicked.connect(self._on_clear_log_clicked)
        log_top.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_top)

        self.log = LogConsole(log_container)
        log_layout.addWidget(self.log, 1)

        self._vertical_split.addWidget(top)
        self._vertical_split.addWidget(log_container)
        self._vertical_split.setStretchFactor(0, 3)
        self._vertical_split.setStretchFactor(1, 1)

    def _build_controls(self) -> None:
        data_group = QGroupBox("Data", self._controls_container)
        self._data_group = data_group
        data_form = QFormLayout(data_group)

        self.file_selector = FileSelector("Excel file", parent=data_group)
        data_form.addRow(self.file_selector)
        self.file_selector.path_changed.connect(self._on_file_path_changed)

        self.btn_load_sheets = QPushButton("Reload sheets", data_group)
        self.btn_load_sheets.clicked.connect(self._load_sheets)
        data_form.addRow(self.btn_load_sheets)

        self.mode = QComboBox(data_group)
        self.mode.addItem("Single sheet", "single")
        self.mode.addItem("Multi sheet", "multi")
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        data_form.addRow("Mode", self.mode)

        self.nmr_sheet = QComboBox(data_group)
        self.nmr_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        data_form.addRow("NMR sheet", self.nmr_sheet)
        self._nmr_sheet_label = data_form.labelForField(self.nmr_sheet)

        self.conc_sheet = QComboBox(data_group)
        self.conc_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        data_form.addRow("Conc sheet", self.conc_sheet)
        self._conc_sheet_label = data_form.labelForField(self.conc_sheet)

        self.btn_load_columns = QPushButton("Reload columns", data_group)
        self.btn_load_columns.clicked.connect(self._load_columns)
        data_form.addRow(self.btn_load_columns)

        self.conc_columns = QListWidget(data_group)
        self.conc_columns.setMinimumHeight(100)
        self.conc_columns.itemChanged.connect(self._on_conc_columns_changed)
        conc_box = QWidget(data_group)
        conc_box_layout = QVBoxLayout(conc_box)
        conc_box_layout.setContentsMargins(0, 0, 0, 0)
        conc_box_layout.addWidget(self.conc_columns, 1)
        conc_btns = QHBoxLayout()
        self.btn_conc_all = QPushButton("Select all", conc_box)
        self.btn_conc_all.clicked.connect(lambda: self._set_list_checked(self.conc_columns, True))
        conc_btns.addWidget(self.btn_conc_all)
        self.btn_conc_none = QPushButton("Select none", conc_box)
        self.btn_conc_none.clicked.connect(lambda: self._set_list_checked(self.conc_columns, False))
        conc_btns.addWidget(self.btn_conc_none)
        conc_box_layout.addLayout(conc_btns)
        data_form.addRow("Conc columns", conc_box)

        self.signal_columns = QListWidget(data_group)
        self.signal_columns.setMinimumHeight(100)
        sig_box = QWidget(data_group)
        sig_box_layout = QVBoxLayout(sig_box)
        sig_box_layout.setContentsMargins(0, 0, 0, 0)
        sig_box_layout.addWidget(self.signal_columns, 1)
        sig_btns = QHBoxLayout()
        self.btn_sig_all = QPushButton("Select all", sig_box)
        self.btn_sig_all.clicked.connect(lambda: self._set_list_checked(self.signal_columns, True))
        sig_btns.addWidget(self.btn_sig_all)
        self.btn_sig_none = QPushButton("Select none", sig_box)
        self.btn_sig_none.clicked.connect(lambda: self._set_list_checked(self.signal_columns, False))
        sig_btns.addWidget(self.btn_sig_none)
        sig_box_layout.addLayout(sig_btns)
        data_form.addRow("Signal columns", sig_box)

        self.receptor_label = QComboBox(data_group)
        data_form.addRow("Receptor label", self.receptor_label)
        self.guest_label = QComboBox(data_group)
        data_form.addRow("Guest label", self.guest_label)

        self._controls_layout.addWidget(data_group)

        model_group = QGroupBox("Model", self._controls_container)
        self._model_group = model_group
        model_form = QFormLayout(model_group)
        self.model_preset = QComboBox(model_group)
        self.model_preset.addItems(["(custom)", "1:1", "1:2", "1:3", "2:1"])
        self.model_preset.currentIndexChanged.connect(self._apply_model_preset)
        model_form.addRow("Preset", self.model_preset)

        self.model_matrix = QTextEdit(model_group)
        self.model_matrix.setPlaceholderText('e.g. [[1,0,1],[0,1,1]]  # rows=components, cols=species')
        self.model_matrix.setMinimumHeight(90)
        model_form.addRow("Model matrix", self.model_matrix)

        self.non_abs_species = QLineEdit(model_group)
        self.non_abs_species.setPlaceholderText("comma-separated species indices, e.g. 0,2")
        model_form.addRow("Non-abs species", self.non_abs_species)

        self._controls_layout.addWidget(model_group)

        opt_group = QGroupBox("Optimization", self._controls_container)
        self._opt_group = opt_group
        opt_layout = QVBoxLayout(opt_group)
        opt_form = QFormLayout()

        self.algorithm = QComboBox(opt_group)
        self.algorithm.addItems(["Newton-Raphson", "Levenberg-Marquardt"])
        opt_form.addRow("Algorithm", self.algorithm)

        self.model_settings = QComboBox(opt_group)
        self.model_settings.addItems(["Free", "Step by step", "Non-cooperative"])
        opt_form.addRow("Model settings", self.model_settings)

        self.optimizer = QComboBox(opt_group)
        self.optimizer.addItems(
            [
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
            ]
        )
        opt_form.addRow("Optimizer", self.optimizer)
        opt_layout.addLayout(opt_form)

        self.params_table = QTableWidget(0, 5, opt_group)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Fixed"])
        self.params_table.setMinimumHeight(160)
        opt_layout.addWidget(self.params_table, 1)

        params_buttons = QHBoxLayout()
        self.btn_add_param = QPushButton("Add K", opt_group)
        self.btn_add_param.clicked.connect(lambda: self._add_param_row())
        params_buttons.addWidget(self.btn_add_param)
        self.btn_remove_param = QPushButton("Remove K", opt_group)
        self.btn_remove_param.clicked.connect(self._remove_selected_param_rows)
        params_buttons.addWidget(self.btn_remove_param)
        params_buttons.addStretch(1)
        opt_layout.addLayout(params_buttons)

        self._controls_layout.addWidget(opt_group)

        actions_group = QGroupBox("Actions", self._controls_container)
        self._actions_group = actions_group
        actions_layout = QVBoxLayout(actions_group)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run fit", actions_group)
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel", actions_group)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_cancel)
        actions_layout.addLayout(btn_row)

        io_row = QHBoxLayout()
        self.btn_import = QPushButton("Import config…", actions_group)
        self.btn_import.clicked.connect(self._on_import_clicked)
        io_row.addWidget(self.btn_import)
        self.btn_export = QPushButton("Export config…", actions_group)
        self.btn_export.clicked.connect(self._on_export_clicked)
        io_row.addWidget(self.btn_export)
        actions_layout.addLayout(io_row)

        self.btn_reset = QPushButton("Reset tab", actions_group)
        self.btn_reset.clicked.connect(self.reset_tab)
        actions_layout.addWidget(self.btn_reset)

        self._controls_layout.addWidget(actions_group)
        self._controls_layout.addStretch(1)

        self._apply_mode_ui()

    def _on_file_path_changed(self, _path: str) -> None:
        file_path = self.file_selector.path()
        if not file_path:
            self._clear_data_selections()
            return
        path = Path(file_path)
        if not path.exists():
            return
        if path.suffix.lower() != ".xlsx":
            QMessageBox.warning(self, "Unsupported file", "Only .xlsx files are supported.")
            return
        self._load_sheets()

    def _clear_data_selections(self) -> None:
        self.nmr_sheet.blockSignals(True)
        self.conc_sheet.blockSignals(True)
        self.nmr_sheet.clear()
        self.conc_sheet.clear()
        self.nmr_sheet.blockSignals(False)
        self.conc_sheet.blockSignals(False)

        self.conc_columns.clear()
        self.signal_columns.clear()
        self.receptor_label.clear()
        self.guest_label.clear()

    def _on_mode_changed(self, _index: int) -> None:
        self._apply_mode_ui()
        self._load_columns()

    def _apply_mode_ui(self) -> None:
        mode = str(self.mode.currentData() or "single")
        is_single = mode == "single"
        if self._conc_sheet_label is not None:
            self._conc_sheet_label.setVisible(not is_single)
        self.conc_sheet.setVisible(not is_single)
        if self._nmr_sheet_label is not None:
            self._nmr_sheet_label.setText("Sheet" if is_single else "NMR sheet")
        if is_single:
            self.conc_sheet.blockSignals(True)
            self.conc_sheet.setCurrentText(self.nmr_sheet.currentText())
            self.conc_sheet.blockSignals(False)

    def _on_sheet_changed(self, _index: int) -> None:
        mode = str(self.mode.currentData() or "single")
        if mode == "single":
            self.conc_sheet.blockSignals(True)
            self.conc_sheet.setCurrentText(self.nmr_sheet.currentText())
            self.conc_sheet.blockSignals(False)
        self._load_columns()

    def _set_list_checked(self, widget: QListWidget, checked: bool) -> None:
        widget.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(widget.count()):
            item = widget.item(i)
            if item is not None:
                item.setCheckState(state)
        widget.blockSignals(False)
        if widget is self.conc_columns:
            self._sync_role_combos_from_conc_selection()

    def _checked_texts(self, widget: QListWidget) -> list[str]:
        out: list[str] = []
        for i in range(widget.count()):
            item = widget.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.text()))
        return out

    def _on_conc_columns_changed(self, _item: QListWidgetItem) -> None:
        self._sync_role_combos_from_conc_selection()

    def _sync_role_combos_from_conc_selection(self) -> None:
        selected = self._checked_texts(self.conc_columns)
        prev_receptor = self.receptor_label.currentText()
        prev_guest = self.guest_label.currentText()

        self.receptor_label.blockSignals(True)
        self.guest_label.blockSignals(True)
        self.receptor_label.clear()
        self.guest_label.clear()
        self.receptor_label.addItems(selected)
        self.guest_label.addItems(selected)
        self.receptor_label.blockSignals(False)
        self.guest_label.blockSignals(False)

        def _restore(combo: QComboBox, value: str) -> None:
            ix = combo.findText(value)
            if ix >= 0:
                combo.setCurrentIndex(ix)

        _restore(self.receptor_label, prev_receptor)
        _restore(self.guest_label, prev_guest)

    def _apply_model_preset(self) -> None:
        preset = self.model_preset.currentText()
        if preset == "(custom)":
            return

        if preset == "1:1":
            matrix = [[1, 0, 1], [0, 1, 1]]
        elif preset == "1:2":
            matrix = [[1, 0, 1, 1], [0, 1, 1, 2]]
        elif preset == "1:3":
            matrix = [[1, 0, 1, 1, 1], [0, 1, 1, 2, 3]]
        elif preset == "2:1":
            matrix = [[1, 0, 2], [0, 1, 1]]
        else:
            return

        self.model_matrix.setPlainText(json.dumps(matrix))
        n_complex = max(0, len(matrix[0]) - len(matrix))
        self._set_param_row_count(n_complex)

    def _set_param_row_count(self, n: int) -> None:
        n = max(0, int(n))
        current = self.params_table.rowCount()
        if current == n:
            return
        if current < n:
            for _ in range(n - current):
                self._add_param_row()
        else:
            for row in reversed(range(n, current)):
                self.params_table.removeRow(row)

    def _add_param_row(self) -> None:
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)
        name_item = QTableWidgetItem(f"K{row + 1}")
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.params_table.setItem(row, 0, name_item)
        for col in (1, 2, 3):
            self.params_table.setItem(row, col, QTableWidgetItem(""))
        fixed_item = QTableWidgetItem("")
        fixed_item.setFlags(fixed_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        fixed_item.setCheckState(Qt.CheckState.Unchecked)
        self.params_table.setItem(row, 4, fixed_item)

    def _remove_selected_param_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.params_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.params_table.removeRow(row)
        for row in range(self.params_table.rowCount()):
            it = self.params_table.item(row, 0)
            if it is not None:
                it.setText(f"K{row + 1}")

    def _load_sheets(self) -> None:
        file_path = self.file_selector.path()
        if not file_path:
            QMessageBox.warning(self, "Missing file", "Select an Excel file first.")
            return
        try:
            sheets = _list_excel_sheets(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            return

        sheets_with_blank = [""] + sheets
        self.nmr_sheet.blockSignals(True)
        self.conc_sheet.blockSignals(True)
        self.nmr_sheet.clear()
        self.nmr_sheet.addItems(sheets_with_blank)
        self.nmr_sheet.setCurrentIndex(0)
        self.conc_sheet.clear()
        self.conc_sheet.addItems(sheets_with_blank)
        self.conc_sheet.setCurrentIndex(0)
        self.nmr_sheet.blockSignals(False)
        self.conc_sheet.blockSignals(False)

        self._apply_mode_ui()
        self.conc_columns.clear()
        self.signal_columns.clear()
        self.receptor_label.clear()
        self.guest_label.clear()

    def _load_columns(self) -> None:
        file_path = self.file_selector.path()
        nmr_sheet = self.nmr_sheet.currentText().strip()
        conc_sheet = self.conc_sheet.currentText().strip()
        if not file_path or not nmr_sheet or not conc_sheet:
            self.conc_columns.clear()
            self.signal_columns.clear()
            self.receptor_label.clear()
            self.guest_label.clear()
            return
        try:
            conc_cols = _list_excel_columns(file_path, conc_sheet)
            sig_cols = _list_excel_columns(file_path, nmr_sheet)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            return

        self.conc_columns.blockSignals(True)
        self.conc_columns.clear()
        for c in conc_cols:
            item = QListWidgetItem(str(c), self.conc_columns)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.conc_columns.blockSignals(False)

        self.signal_columns.blockSignals(True)
        self.signal_columns.clear()
        for c in sig_cols:
            item = QListWidgetItem(str(c), self.signal_columns)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.signal_columns.blockSignals(False)

        self._sync_role_combos_from_conc_selection()

    def _collect_config(self) -> dict[str, Any]:
        file_path = self.file_selector.path()
        if not file_path:
            raise ValueError("Missing Excel file.")

        mode = str(self.mode.currentData() or "single")
        nmr_sheet = self.nmr_sheet.currentText().strip()
        conc_sheet = self.conc_sheet.currentText().strip()
        if not nmr_sheet or not conc_sheet:
            raise ValueError("Select NMR and conc sheets.")

        column_names: list[str] = []
        for i in range(self.conc_columns.count()):
            item = self.conc_columns.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                column_names.append(str(item.text()))
        if not column_names:
            raise ValueError("Select at least one concentration column.")

        signal_names: list[str] = []
        for i in range(self.signal_columns.count()):
            item = self.signal_columns.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                signal_names.append(str(item.text()))
        if not signal_names:
            raise ValueError("Select at least one signal column.")

        receptor_label = self.receptor_label.currentText().strip()
        guest_label = self.guest_label.currentText().strip()
        if not receptor_label or not guest_label:
            raise ValueError("Select receptor and guest labels.")
        if receptor_label == guest_label:
            raise ValueError("Receptor and guest cannot be the same column.")
        if receptor_label not in column_names or guest_label not in column_names:
            raise ValueError("Receptor/guest must be selected among checked concentration columns.")

        model_text = self.model_matrix.toPlainText().strip()
        if not model_text:
            raise ValueError("Model matrix is empty.")
        try:
            modelo = json.loads(model_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid model matrix JSON: {exc}") from exc

        non_abs_species: list[int] = []
        raw_nas = self.non_abs_species.text().strip()
        if raw_nas:
            try:
                non_abs_species = [int(x.strip()) for x in raw_nas.split(",") if x.strip()]
            except ValueError as exc:
                raise ValueError("Non-abs species must be comma-separated integers.") from exc

        initial_k: list[float] = []
        bounds: list[list[float | None]] = []
        fixed_mask: list[bool] = []
        for row in range(self.params_table.rowCount()):
            v_item = self.params_table.item(row, 1)
            val_s = (v_item.text() if v_item is not None else "").strip()
            if not val_s:
                raise ValueError(f"Missing initial value for K{row + 1}.")
            try:
                k_val = float(val_s)
            except ValueError as exc:
                raise ValueError(f"Invalid value for K{row + 1}.") from exc

            fixed_item = self.params_table.item(row, 4)
            fixed = fixed_item is not None and fixed_item.checkState() == Qt.CheckState.Checked

            min_item = self.params_table.item(row, 2)
            max_item = self.params_table.item(row, 3)
            min_v = _ensure_float_or_none(min_item.text() if min_item is not None else "")
            max_v = _ensure_float_or_none(max_item.text() if max_item is not None else "")

            if fixed:
                min_v = k_val
                max_v = k_val

            initial_k.append(k_val)
            bounds.append([min_v, max_v])
            fixed_mask.append(bool(fixed))

        return {
            "file_path": file_path,
            "nmr_mode": mode,
            "nmr_sheet": nmr_sheet,
            "conc_sheet": conc_sheet,
            "column_names": column_names,
            "signal_names": signal_names,
            "receptor_label": receptor_label,
            "guest_label": guest_label,
            "modelo": modelo,
            "non_abs_species": non_abs_species,
            "algorithm": self.algorithm.currentText() or "Newton-Raphson",
            "model_settings": self.model_settings.currentText() or "Free",
            "optimizer": self.optimizer.currentText() or "powell",
            "initial_k": initial_k,
            "bounds": bounds,
            "fixed_mask": fixed_mask,
            "k_fixed": fixed_mask,
        }

    def _set_running(self, running: bool) -> None:
        self._data_group.setEnabled(not running)
        self._model_group.setEnabled(not running)
        self._opt_group.setEnabled(not running)
        self.btn_run.setEnabled(not running)
        self.btn_import.setEnabled(not running)
        self.btn_export.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _on_run_clicked(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "A fit is already running.")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return

        self.log.append_text("Starting NMR fit…")
        self._set_running(True)

        worker = FitWorker(run_nmr_fit, config=config, parent=self)
        self._worker = worker
        worker.progress.connect(lambda msg: self.log.append_text(str(msg).rstrip()))
        worker.result.connect(self._on_fit_result)
        worker.error.connect(self._on_fit_error)
        worker.finished.connect(self._on_fit_finished)
        worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.log.append_text("Cancellation requested…")

    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            self.log.append_text("Unexpected result type.")
            return
        self._last_result = result

        graphs = (result.get("graphs") or result.get("legacy_graphs") or {}) if isinstance(result, dict) else {}
        self.plot_selector.blockSignals(True)
        self.plot_selector.clear()
        self.plot_selector.addItems([str(k) for k in graphs.keys()])
        self.plot_selector.blockSignals(False)
        if self.plot_selector.count() > 0:
            self.plot_selector.setCurrentIndex(0)
            self._render_selected_plot()
        else:
            self.canvas.clear()

        results_text = result.get("results_text") or ""
        if results_text:
            self.log.append_text("")
            self.log.append_text(str(results_text).rstrip())
            self.log.append_text("")

    def _on_fit_error(self, message: str) -> None:
        self.log.append_text(f"ERROR: {message}")
        QMessageBox.critical(self, "Fit error", str(message))

    def _on_fit_finished(self) -> None:
        self._worker = None
        self._set_running(False)

    def _on_plot_selection_changed(self) -> None:
        self._render_selected_plot()

    def _render_selected_plot(self) -> None:
        if not self._last_result:
            return
        graphs = self._last_result.get("graphs") or self._last_result.get("legacy_graphs") or {}
        key = self.plot_selector.currentText()
        if not key or key not in graphs:
            return
        self.canvas.show_image_base64(str(graphs[key]), title=str(key))

    def _on_clear_log_clicked(self) -> None:
        self.log.clear()

    def reset_tab(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "Cancel the running fit before resetting.")
            return
        self.log.clear()
        self.canvas.clear()
        self._last_result = None
        self.plot_selector.clear()

    def _on_export_clicked(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export config", "nmr_config.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
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
        except Exception as exc:
            QMessageBox.critical(self, "Config error", str(exc))

    def load_config(self, config: dict[str, Any]) -> None:
        file_path = str(config.get("file_path") or "")
        if not file_path:
            raise ValueError("Config missing 'file_path'.")
        if not Path(file_path).exists():
            raise ValueError(f"Excel file not found: {file_path}")

        raw_mode = str(config.get("nmr_mode") or "").strip().lower()
        nmr_sheet_value = str(config.get("nmr_sheet") or config.get("spectra_sheet") or "")
        conc_sheet_value = str(config.get("conc_sheet") or "")
        if raw_mode not in {"single", "multi"}:
            if nmr_sheet_value and conc_sheet_value and nmr_sheet_value != conc_sheet_value:
                raw_mode = "multi"
            else:
                raw_mode = "single"

        mode_ix = self.mode.findData(raw_mode)
        if mode_ix >= 0:
            self.mode.setCurrentIndex(mode_ix)

        self.file_selector.set_path(file_path)
        self._load_sheets()

        def _set_combo(combo: QComboBox, value: str) -> None:
            ix = combo.findText(value)
            if ix >= 0:
                combo.setCurrentIndex(ix)

        missing: list[str] = []

        if nmr_sheet_value:
            before = self.nmr_sheet.currentText()
            _set_combo(self.nmr_sheet, nmr_sheet_value)
            if self.nmr_sheet.currentText() == before and before != nmr_sheet_value:
                missing.append(f"NMR sheet '{nmr_sheet_value}' not found.")

        if raw_mode == "multi" and conc_sheet_value:
            before = self.conc_sheet.currentText()
            _set_combo(self.conc_sheet, conc_sheet_value)
            if self.conc_sheet.currentText() == before and before != conc_sheet_value:
                missing.append(f"Conc sheet '{conc_sheet_value}' not found.")

        if raw_mode == "single":
            self._apply_mode_ui()

        self._load_columns()

        selected_conc = set(str(c) for c in (config.get("column_names") or []))
        if selected_conc:
            available_conc = {
                self.conc_columns.item(i).text()
                for i in range(self.conc_columns.count())
                if self.conc_columns.item(i) is not None
            }
            missing_conc = sorted(selected_conc - available_conc)
            if missing_conc:
                missing.append(f"Conc columns not found: {missing_conc}")
            self.conc_columns.blockSignals(True)
            for i in range(self.conc_columns.count()):
                item = self.conc_columns.item(i)
                if item is None:
                    continue
                item.setCheckState(Qt.CheckState.Checked if item.text() in selected_conc else Qt.CheckState.Unchecked)
            self.conc_columns.blockSignals(False)
        self._sync_role_combos_from_conc_selection()

        selected_sig = set(str(c) for c in (config.get("signal_names") or []))
        if selected_sig:
            available_sig = {
                self.signal_columns.item(i).text()
                for i in range(self.signal_columns.count())
                if self.signal_columns.item(i) is not None
            }
            missing_sig = sorted(selected_sig - available_sig)
            if missing_sig:
                missing.append(f"Signal columns not found: {missing_sig}")
            self.signal_columns.blockSignals(True)
            for i in range(self.signal_columns.count()):
                item = self.signal_columns.item(i)
                if item is None:
                    continue
                item.setCheckState(Qt.CheckState.Checked if item.text() in selected_sig else Qt.CheckState.Unchecked)
            self.signal_columns.blockSignals(False)

        receptor_value = str(config.get("receptor_label") or "")
        guest_value = str(config.get("guest_label") or "")
        if receptor_value:
            _set_combo(self.receptor_label, receptor_value)
            if self.receptor_label.currentText() != receptor_value:
                missing.append(f"Receptor label '{receptor_value}' not found among selected conc columns.")
        if guest_value:
            _set_combo(self.guest_label, guest_value)
            if self.guest_label.currentText() != guest_value:
                missing.append(f"Guest label '{guest_value}' not found among selected conc columns.")

        self.model_matrix.setPlainText(json.dumps(config.get("modelo") or []))
        self.non_abs_species.setText(",".join(str(x) for x in (config.get("non_abs_species") or [])))

        _set_combo(self.algorithm, str(config.get("algorithm") or "Newton-Raphson"))
        _set_combo(self.model_settings, str(config.get("model_settings") or "Free"))
        _set_combo(self.optimizer, str(config.get("optimizer") or "powell"))

        initial_k = list(config.get("initial_k") or [])
        bounds = list(config.get("bounds") or [])
        fixed_mask = list(config.get("fixed_mask") or config.get("k_fixed") or [])
        self.params_table.setRowCount(0)
        for i, k in enumerate(initial_k):
            self._add_param_row()
            self.params_table.item(i, 1).setText(str(k))
            if i < len(bounds) and isinstance(bounds[i], (list, tuple)) and len(bounds[i]) >= 2:
                b0, b1 = bounds[i][0], bounds[i][1]
                if b0 is not None:
                    self.params_table.item(i, 2).setText(str(b0))
                if b1 is not None:
                    self.params_table.item(i, 3).setText(str(b1))
            if i < len(fixed_mask) and fixed_mask[i]:
                self.params_table.item(i, 4).setCheckState(Qt.CheckState.Checked)

        if missing:
            msg = "\n".join(missing)
            self.log.append_text(f"Config loaded with warnings:\n{msg}")
            QMessageBox.information(self, "Config warnings", msg)
