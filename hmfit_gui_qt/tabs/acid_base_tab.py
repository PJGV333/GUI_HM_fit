from __future__ import annotations

from pathlib import Path
from typing import Any

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
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hmfit_core.api import run_acid_base_fit
from hmfit_core.exports import write_results_xlsx
from hmfit_gui_qt.widgets.log_console import LogConsole
from hmfit_gui_qt.widgets.mpl_canvas import MplCanvas, NavigationToolbar
from hmfit_gui_qt.workers.fit_worker import FitWorker


def _optional_float(text: str) -> float | None:
    value = str(text or "").strip()
    if not value:
        return None
    return float(value)


class AcidBaseTab(QWidget):
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
        self._build_ui()
        self._set_running(False)
        self.canvas_main.show_message("Import a CSV file to begin")
        self.log.append_text("Ready. Import potentiometry, spectroscopy, or NMR CSV data.")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        outer.addWidget(splitter, 1)

        left_scroll = QScrollArea(splitter)
        left_scroll.setWidgetResizable(True)
        left = QWidget(left_scroll)
        left_scroll.setWidget(left)
        left_layout = QVBoxLayout(left)

        self.tabs = QTabWidget(left)
        left_layout.addWidget(self.tabs, 1)

        self._build_data_tab()
        self._build_system_tab()
        self._build_model_tab()
        self._build_parameters_tab()
        self._build_fit_tab()
        self._build_graphs_tab()
        self._build_export_tab()

        right = QSplitter(Qt.Orientation.Vertical, splitter)
        plot_panel = QWidget(right)
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

        log_panel = QWidget(right)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Diagnostics / Log", log_panel))
        self.log = LogConsole(log_panel)
        self.log.setMinimumHeight(160)
        log_layout.addWidget(self.log, 1)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

    def _build_data_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.combo_data_type = QComboBox(tab)
        self.combo_data_type.addItem("Potentiometry pH/EMF", "potentiometry")
        self.combo_data_type.addItem("Spectroscopy signal vs pH", "spectroscopy")
        self.combo_data_type.addItem("1H NMR shifts vs pH", "nmr")
        form.addRow("Data type", self.combo_data_type)
        layout.addLayout(form)

        file_row = QHBoxLayout()
        self.btn_choose_file = QPushButton("Choose CSV...", tab)
        self.btn_choose_file.clicked.connect(self._on_choose_file_clicked)
        file_row.addWidget(self.btn_choose_file)
        self.lbl_file_status = QLabel("No file selected", tab)
        self.lbl_file_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_row.addWidget(self.lbl_file_status, 1)
        layout.addLayout(file_row)

        self.preview_text = QPlainTextEdit(tab)
        self.preview_text.setReadOnly(True)
        self.preview_text.setMinimumHeight(180)
        layout.addWidget(self.preview_text, 1)
        self.tabs.addTab(tab, "Data")

    def _build_system_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QFormLayout(tab)
        self.edit_component_name = QLineEdit("L", tab)
        self.edit_pka = QLineEdit("5.0", tab)
        self.edit_pka.setPlaceholderText("Example: 4.2 or 4.2, 8.7")
        self.spin_analyte_conc = QDoubleSpinBox(tab)
        self.spin_analyte_conc.setDecimals(8)
        self.spin_analyte_conc.setRange(0.0, 1.0e6)
        self.spin_analyte_conc.setValue(1.0e-3)
        self.spin_analyte_conc.setSingleStep(1.0e-4)
        self.spin_base_charge = QSpinBox(tab)
        self.spin_base_charge.setRange(-10, 10)
        self.spin_base_charge.setValue(-1)
        layout.addRow("Component name", self.edit_component_name)
        layout.addRow("Initial pKa", self.edit_pka)
        layout.addRow("Analyte concentration", self.spin_analyte_conc)
        layout.addRow("Charge of L", self.spin_base_charge)
        self.tabs.addTab(tab, "Chemical system")

    def _build_model_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QFormLayout(tab)
        self.spin_initial_volume = QDoubleSpinBox(tab)
        self.spin_initial_volume.setDecimals(6)
        self.spin_initial_volume.setRange(1.0e-12, 1.0e9)
        self.spin_initial_volume.setValue(10.0)
        self.spin_titrant_conc = QDoubleSpinBox(tab)
        self.spin_titrant_conc.setDecimals(8)
        self.spin_titrant_conc.setRange(0.0, 1.0e6)
        self.spin_titrant_conc.setValue(1.0e-3)
        self.spin_titrant_conc.setSingleStep(1.0e-4)
        self.combo_titrant_type = QComboBox(tab)
        self.combo_titrant_type.addItem("Base", "base")
        self.combo_titrant_type.addItem("Acid", "acid")
        self.edit_e0 = QLineEdit("", tab)
        self.edit_e0.setPlaceholderText("fixed E0, mV")
        self.edit_slope = QLineEdit("-59.16", tab)
        self.edit_slope.setPlaceholderText("fixed slope, mV/pH")
        self.chk_fit_electrode = QCheckBox("Fit E0 and slope for EMF data", tab)
        layout.addRow("Initial volume", self.spin_initial_volume)
        layout.addRow("Titrant concentration", self.spin_titrant_conc)
        layout.addRow("Titrant type", self.combo_titrant_type)
        layout.addRow("Electrode E0", self.edit_e0)
        layout.addRow("Electrode slope", self.edit_slope)
        layout.addRow("", self.chk_fit_electrode)
        self.tabs.addTab(tab, "Experimental model")

    def _build_parameters_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QFormLayout(tab)
        self.spin_sigma_ph = QDoubleSpinBox(tab)
        self.spin_sigma_ph.setDecimals(6)
        self.spin_sigma_ph.setRange(1.0e-12, 1.0e6)
        self.spin_sigma_ph.setValue(1.0)
        self.spin_sigma_emf = QDoubleSpinBox(tab)
        self.spin_sigma_emf.setDecimals(6)
        self.spin_sigma_emf.setRange(1.0e-12, 1.0e6)
        self.spin_sigma_emf.setValue(1.0)
        self.chk_baseline = QCheckBox("Include linear baseline in spectroscopy", tab)
        layout.addRow("sigma pH", self.spin_sigma_ph)
        layout.addRow("sigma EMF", self.spin_sigma_emf)
        layout.addRow("", self.chk_baseline)
        self.tabs.addTab(tab, "Parameters")

    def _build_fit_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        actions = QHBoxLayout()
        self.btn_process = QPushButton("Fit", tab)
        self.btn_process.clicked.connect(self._on_process_clicked)
        actions.addWidget(self.btn_process)
        self.btn_cancel = QPushButton("Cancel", tab)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        actions.addWidget(self.btn_cancel)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.results_text = QPlainTextEdit(tab)
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(260)
        layout.addWidget(self.results_text, 1)
        self.tabs.addTab(tab, "Fit")

    def _build_graphs_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        box = QGroupBox("Plot pages", tab)
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(QLabel("Use Prev/Next above the plot to browse fit, residual, and species diagrams.", box))
        layout.addWidget(box)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Plots")

    def _build_export_tab(self) -> None:
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        self.btn_save = QPushButton("Save results...", tab)
        self.btn_save.clicked.connect(self._on_save_results_clicked)
        layout.addWidget(self.btn_save)
        layout.addWidget(QLabel("Excel export includes pKa, log_beta, species tables, residuals, and covariance when available.", tab))
        layout.addStretch(1)
        self.tabs.addTab(tab, "Export")

    def _on_choose_file_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV (*.csv *.txt)")
        if not path:
            return
        self._set_file_path(path)

    def _set_file_path(self, file_path: str) -> None:
        path = Path(str(file_path or ""))
        if not path.exists():
            QMessageBox.warning(self, "Missing file", f"CSV file not found:\n{path}")
            return
        self._file_path = str(path)
        self.lbl_file_status.setText(self._file_path)
        try:
            df = pd.read_csv(path, nrows=8)
            self.preview_text.setPlainText(df.to_string(index=False))
        except Exception as exc:
            self.preview_text.setPlainText(f"Could not preview CSV: {exc}")

    def _collect_config(self) -> dict[str, Any]:
        if not self._file_path:
            raise ValueError("No CSV file selected.")
        cfg = {
            "file_path": self._file_path,
            "data_type": str(self.combo_data_type.currentData() or "potentiometry"),
            "component_name": self.edit_component_name.text().strip() or "L",
            "pka_initial": self.edit_pka.text().strip() or "5.0",
            "analyte_concentration": float(self.spin_analyte_conc.value()),
            "base_charge": int(self.spin_base_charge.value()),
            "initial_volume": float(self.spin_initial_volume.value()),
            "titrant_concentration": float(self.spin_titrant_conc.value()),
            "titrant_type": str(self.combo_titrant_type.currentData() or "base"),
            "electrode_e0": _optional_float(self.edit_e0.text()),
            "electrode_slope": _optional_float(self.edit_slope.text()),
            "fit_electrode": bool(self.chk_fit_electrode.isChecked()),
            "sigma_pH": float(self.spin_sigma_ph.value()),
            "sigma_E": float(self.spin_sigma_emf.value()),
            "baseline": bool(self.chk_baseline.isChecked()),
        }
        return cfg

    def _set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        self.btn_process.setEnabled(not running)
        self.btn_choose_file.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_save.setEnabled((not running) and bool(self._last_result))
        self.btn_process.setText("Fitting..." if running else "Fit")
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
        self.results_text.setPlainText(str(result.get("results_text") or ""))
        self._plot_pages = []
        graphs = result.get("legacy_graphs") or result.get("graphs") or {}
        titles = {item.get("id"): item.get("title") for item in (result.get("availablePlots") or [])}
        for key, png in graphs.items():
            if png:
                self._plot_pages.append((str(key), str(titles.get(key) or key)))
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
