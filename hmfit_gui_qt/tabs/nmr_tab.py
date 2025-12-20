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
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hmfit_core.api import run_nmr_fit
from hmfit_core.exports import write_results_xlsx
from hmfit_gui_qt.widgets.log_console import LogConsole
from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsState, ModelOptPlotsWidget
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


class NMRTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FitWorker | None = None
        self._thread: QThread | None = None
        self._last_result: dict[str, Any] | None = None
        self._plot_pages: list[dict[str, str]] = []
        self._plot_index: int = -1
        self._file_path: str = ""

        self._build_ui()
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
        self.combo_conc_sheet = QComboBox(self._data_group)
        self.combo_shift_sheet = QComboBox(self._data_group)
        self.combo_conc_sheet.currentIndexChanged.connect(self._on_conc_sheet_changed)
        self.combo_shift_sheet.currentIndexChanged.connect(self._on_shift_sheet_changed)

        conc_box = QVBoxLayout()
        conc_box.addWidget(QLabel("Concentration Sheet Name", self._data_group))
        conc_box.addWidget(self.combo_conc_sheet)
        shift_box = QVBoxLayout()
        shift_box.addWidget(QLabel("Chemical Shift Sheet Name", self._data_group))
        shift_box.addWidget(self.combo_shift_sheet)
        sheets_row.addLayout(conc_box, 1)
        sheets_row.addLayout(shift_box, 1)
        data_layout.addLayout(sheets_row)

        # Signals selector
        data_layout.addWidget(QLabel("Chemical Shifts (Signals)", self._data_group))
        self.list_signals = QListWidget(self._data_group)
        self.list_signals.setMinimumHeight(120)
        data_layout.addWidget(self.list_signals, 1)
        sig_btns = QHBoxLayout()
        self.btn_signals_all = QPushButton("Select all", self._data_group)
        self.btn_signals_all.clicked.connect(lambda: self._set_list_checked(self.list_signals, True))
        sig_btns.addWidget(self.btn_signals_all)
        self.btn_signals_none = QPushButton("Select none", self._data_group)
        self.btn_signals_none.clicked.connect(lambda: self._set_list_checked(self.list_signals, False))
        sig_btns.addWidget(self.btn_signals_none)
        sig_btns.addStretch(1)
        data_layout.addLayout(sig_btns)

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

        left_layout.addWidget(self._data_group)

        # Sub-tabs: Model / Optimization / Plots
        self.model_opt_plots = ModelOptPlotsWidget(left_container)
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

    # ---- Plot navigation (Prev/Next) ----
    def _plot_title_for_key(self, key: str) -> str:
        titles = {
            "fit": "Chemical shifts fit",
            "concentrations": "Concentration profile",
            "residuals": "Residuals",
        }
        return titles.get(str(key), str(key))

    def _graphs_from_last_result(self) -> dict[str, Any]:
        if not self._last_result:
            return {}
        graphs = self._last_result.get("legacy_graphs") or self._last_result.get("graphs") or {}
        return graphs if isinstance(graphs, dict) else {}

    def _rebuild_plot_pages_from_result(self) -> None:
        graphs = self._graphs_from_last_result()
        if not graphs:
            self._plot_pages = []
            self._plot_index = -1
            self._update_plot_nav()
            return

        order = ["fit", "concentrations", "residuals"]
        pages: list[dict[str, str]] = []

        for key in order:
            if graphs.get(key):
                pages.append({"key": str(key), "title": self._plot_title_for_key(str(key))})

        for key in graphs.keys():
            if key in order:
                continue
            if graphs.get(key):
                pages.append({"key": str(key), "title": self._plot_title_for_key(str(key))})

        self._plot_pages = pages
        self._plot_index = -1
        for i, p in enumerate(pages):
            if p.get("key") == "fit":
                self._plot_index = i
                break
        if self._plot_index < 0 and pages:
            self._plot_index = 0
        self._update_plot_nav()

    def _update_plot_nav(self) -> None:
        has_pages = bool(self._plot_pages) and self._plot_index >= 0
        can_nav = bool(has_pages and len(self._plot_pages) > 1)
        if hasattr(self, "btn_plot_prev"):
            self.btn_plot_prev.setEnabled(can_nav)
        if hasattr(self, "btn_plot_next"):
            self.btn_plot_next.setEnabled(can_nav)
        if hasattr(self, "lbl_plot_title"):
            if has_pages:
                self.lbl_plot_title.setText(str(self._plot_pages[self._plot_index].get("title") or ""))
            else:
                self.lbl_plot_title.setText("Main spectra / titration plot")

    def _render_current_plot(self) -> None:
        graphs = self._graphs_from_last_result()
        if not self._plot_pages or self._plot_index < 0:
            self.canvas_main.clear()
            self._update_plot_nav()
            return
        if self._plot_index >= len(self._plot_pages):
            self._plot_index = 0
        page = self._plot_pages[self._plot_index]
        key = str(page.get("key") or "")
        b64 = graphs.get(key)
        if not key or not b64:
            self.canvas_main.clear()
            self._update_plot_nav()
            return
        self._update_plot_nav()
        self.canvas_main.show_image_base64(str(b64), title=str(page.get("title") or key))

    def _navigate_plot(self, delta: int) -> None:
        if not self._plot_pages or self._plot_index < 0:
            return
        n = len(self._plot_pages)
        if n <= 1:
            return
        self._plot_index = (self._plot_index + int(delta)) % n
        self._render_current_plot()

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
        self.combo_conc_sheet.blockSignals(True)
        self.combo_shift_sheet.blockSignals(True)
        self.combo_conc_sheet.clear()
        self.combo_shift_sheet.clear()
        self.combo_conc_sheet.addItems(sheets_with_blank)
        self.combo_shift_sheet.addItems(sheets_with_blank)
        self.combo_conc_sheet.setCurrentIndex(0)
        self.combo_shift_sheet.setCurrentIndex(0)
        self.combo_conc_sheet.blockSignals(False)
        self.combo_shift_sheet.blockSignals(False)

        self._clear_conc_columns()
        self._populate_signals([])

    def _on_conc_sheet_changed(self) -> None:
        sheet = self.combo_conc_sheet.currentText().strip()
        if not self._file_path or not sheet:
            self._clear_conc_columns()
            return
        try:
            cols = _list_excel_columns(self._file_path, sheet)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            self._clear_conc_columns()
            return
        self._populate_conc_columns(cols)

    def _on_shift_sheet_changed(self) -> None:
        sheet = self.combo_shift_sheet.currentText().strip()
        if not self._file_path or not sheet:
            self._populate_signals([])
            return
        try:
            cols = _list_excel_columns(self._file_path, sheet)
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))
            self._populate_signals([])
            return
        self._populate_signals(cols)

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

        for col in columns:
            cb = QCheckBox(str(col), self._columns_widget)
            cb.setChecked(True)  # match Tauri default
            cb.toggled.connect(self._on_conc_columns_toggled)
            self._columns_layout.addWidget(cb)

        self._columns_layout.addStretch(1)
        self._on_conc_columns_toggled()

    def _on_conc_columns_toggled(self) -> None:
        selected = self._selected_conc_columns()
        self.model_opt_plots.set_available_conc_columns(selected)

    def _selected_conc_columns(self) -> list[str]:
        selected: list[str] = []
        for i in range(self._columns_layout.count()):
            w = self._columns_layout.itemAt(i).widget()
            if isinstance(w, QCheckBox) and w.isChecked():
                selected.append(str(w.text()))
        return selected

    def _populate_signals(self, columns: list[str]) -> None:
        self.list_signals.blockSignals(True)
        self.list_signals.clear()
        for c in columns or []:
            item = QListWidgetItem(str(c), self.list_signals)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.list_signals.blockSignals(False)

    def _set_list_checked(self, widget: QListWidget, checked: bool) -> None:
        widget.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(widget.count()):
            it = widget.item(i)
            if it is not None:
                it.setCheckState(state)
        widget.blockSignals(False)

    def _selected_signals(self) -> list[str]:
        out: list[str] = []
        for i in range(self.list_signals.count()):
            it = self.list_signals.item(i)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                out.append(str(it.text()))
        return out

    # ---- Run / Cancel / Results ----
    def _set_running(self, running: bool) -> None:
        self.btn_process.setEnabled(not running)
        self.btn_choose_file.setEnabled(not running)
        self.combo_conc_sheet.setEnabled(not running)
        self.combo_shift_sheet.setEnabled(not running)
        self.list_signals.setEnabled(not running)
        self.btn_signals_all.setEnabled(not running)
        self.btn_signals_none.setEnabled(not running)
        self.model_opt_plots.setEnabled(not running)
        self.btn_import.setEnabled(not running)
        self.btn_export.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_save.setEnabled(bool(self._last_result) and bool(self._last_result.get("success", True)) and not running)

        self.btn_process.setText("Processing..." if running else "Process Data")
        self._update_plot_nav()

    def _collect_config(self) -> dict[str, Any]:
        if not self._file_path:
            raise ValueError("No Excel file selected.")

        conc_sheet = self.combo_conc_sheet.currentText().strip()
        nmr_sheet = self.combo_shift_sheet.currentText().strip()
        if not conc_sheet or not nmr_sheet:
            raise ValueError("Select Concentration and Chemical Shift sheets.")

        column_names = self._selected_conc_columns()
        if not column_names:
            raise ValueError("Select at least one concentration column in 'Column names'.")

        signal_names = self._selected_signals()
        if not signal_names:
            raise ValueError("Select at least one signal in 'Chemical Shifts (Signals)'.")

        state = self.model_opt_plots.collect_state()
        receptor_label = state.receptor_label
        guest_label = state.guest_label
        if not receptor_label or not guest_label:
            raise ValueError("Select Receptor and Guest roles (or keep Auto with valid column names).")
        if receptor_label == guest_label:
            raise ValueError("Receptor and Guest cannot be the same column.")

        config = {
            "file_path": self._file_path,
            "nmr_sheet": nmr_sheet,
            "conc_sheet": conc_sheet,
            "column_names": column_names,
            "signal_names": signal_names,
            "receptor_label": receptor_label,
            "guest_label": guest_label,
            "modelo": state.modelo,
            "non_abs_species": state.non_abs_species,
            "algorithm": state.algorithm,
            "model_settings": state.model_settings,
            "optimizer": state.optimizer,
            "initial_k": state.initial_k,
            "bounds": state.bounds,
            "fixed_mask": state.fixed_mask,
            "k_fixed": state.fixed_mask,
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
        self._plot_pages = []
        self._plot_index = -1
        self._update_plot_nav()
        self.btn_save.setEnabled(False)
        self.canvas_main.clear()

        self._worker = FitWorker(run_nmr_fit, config=config, parent=self)
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
        self.log.append_text(str(msg))

    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._last_result = result
        self.btn_save.setEnabled(bool(result.get("success", True)))

        self._rebuild_plot_pages_from_result()
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
        path, _ = QFileDialog.getSaveFileName(self, "Export config", "nmr_config.json", "JSON (*.json)")
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

        conc_sheet = str(config.get("conc_sheet") or "")
        nmr_sheet = str(config.get("nmr_sheet") or config.get("spectra_sheet") or config.get("signals_sheet") or "")
        if conc_sheet:
            ix = self.combo_conc_sheet.findText(conc_sheet)
            if ix >= 0:
                self.combo_conc_sheet.setCurrentIndex(ix)
        if nmr_sheet:
            ix = self.combo_shift_sheet.findText(nmr_sheet)
            if ix >= 0:
                self.combo_shift_sheet.setCurrentIndex(ix)

        # Ensure dependent UI is loaded
        self._on_conc_sheet_changed()
        self._on_shift_sheet_changed()

        missing: list[str] = []

        # Restore column_names
        wanted_cols = {str(c) for c in (config.get("column_names") or [])}
        if wanted_cols:
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

        # Restore signals
        wanted_sigs = {str(s) for s in (config.get("signal_names") or [])}
        if wanted_sigs:
            available_sigs = {self.list_signals.item(i).text() for i in range(self.list_signals.count()) if self.list_signals.item(i) is not None}
            missing_sigs = sorted(wanted_sigs - available_sigs)
            if missing_sigs:
                missing.append(f"Missing signals: {missing_sigs}")
            self.list_signals.blockSignals(True)
            for i in range(self.list_signals.count()):
                it = self.list_signals.item(i)
                if it is None:
                    continue
                it.setCheckState(Qt.CheckState.Checked if it.text() in wanted_sigs else Qt.CheckState.Unchecked)
            self.list_signals.blockSignals(False)

        # Restore Model/Optimization
        modelo = config.get("modelo") or []
        n_rows = len(modelo) if isinstance(modelo, list) else 0
        n_cols = len(modelo[0]) if n_rows and isinstance(modelo[0], list) else 0
        n_species = max(0, n_rows - n_cols)
        fixed_mask = list(config.get("fixed_mask") or config.get("k_fixed") or [])
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
            fixed_mask=fixed_mask,
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
        self.combo_conc_sheet.clear()
        self.combo_shift_sheet.clear()
        self._clear_conc_columns()
        self._populate_signals([])
        self.model_opt_plots.reset()

        self.canvas_main.clear()
        self._plot_pages = []
        self._plot_index = -1
        self._update_plot_nav()
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
