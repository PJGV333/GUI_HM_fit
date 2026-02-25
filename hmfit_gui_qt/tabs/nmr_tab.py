from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

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
from hmfit_gui_qt.plots.nmr_registry import build_nmr_registry
from hmfit_gui_qt.plots.nmr_sources import build_nmr_plot_sources
from hmfit_gui_qt.plots.plot_controller import PlotController
from hmfit_gui_qt.widgets.channel_spec import ChannelSpecWidget
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
        self._preview_worker: FitWorker | None = None
        self._preview_thread: QThread | None = None
        self._pending_graph_preview = False
        self._graph_solver_inputs: dict[str, Any] | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_config: dict[str, Any] | None = None
        self._last_fit_context: dict[str, Any] | None = None
        self._plot_controller: PlotController | None = None
        self._is_running = False
        self._file_path: str = ""

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
            registry=build_nmr_registry(),
            build_plot_data=build_nmr_plot_sources,
            legacy_title_for_key=self._plot_title_for_key,
            legacy_order=["fit", "concentrations", "residuals"],
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

        # Column names (concentration columns)
        data_layout.addWidget(QLabel("Column names / Components", self._data_group))
        self._columns_scroll = QScrollArea(self._data_group)
        self._columns_scroll.setWidgetResizable(True)
        self._columns_widget = QWidget(self._columns_scroll)
        self._columns_layout = QVBoxLayout(self._columns_widget)
        self._columns_layout.setContentsMargins(0, 0, 0, 0)
        self._columns_scroll.setWidget(self._columns_widget)
        self._columns_scroll.setMinimumHeight(100)
        data_layout.addWidget(self._columns_scroll)

        # Signals selector
        data_layout.addWidget(QLabel("Chemical Shifts (Signals) & Parent Assignment", self._data_group))
        self.channel_spec_widget = ChannelSpecWidget(self._data_group)
        self.channel_spec_widget.setMinimumHeight(180)
        data_layout.addWidget(self.channel_spec_widget, 1)
        sig_btns = QHBoxLayout()
        self.btn_signals_all = QPushButton("Select all", self._data_group)
        self.btn_signals_all.clicked.connect(lambda: self.channel_spec_widget.set_all_checked(True))
        sig_btns.addWidget(self.btn_signals_all)
        self.btn_signals_none = QPushButton("Select none", self._data_group)
        self.btn_signals_none.clicked.connect(lambda: self.channel_spec_widget.set_all_checked(False))
        sig_btns.addWidget(self.btn_signals_none)
        sig_btns.addStretch(1)
        data_layout.addLayout(sig_btns)

        left_layout.addWidget(self._data_group)

        # Model-definition area inside Model tab: matrix or equations.
        self.model_opt_plots = ModelOptPlotsWidget(left_container, enable_equation_editor=True)
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

    # ---- Plot navigation (Prev/Next) ----
    def _plot_title_for_key(self, key: str) -> str:
        titles = {
            "fit": "Chemical shifts fit",
            "concentrations": "Concentration profile",
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
            cb.setChecked(False)  # match main: start unchecked
            cb.toggled.connect(self._on_conc_columns_toggled)
            self._columns_layout.addWidget(cb)

        self._columns_layout.addStretch(1)
        self._on_conc_columns_toggled()

    def _on_conc_columns_toggled(self) -> None:
        selected = self._selected_conc_columns()
        self.model_opt_plots.set_available_conc_columns(selected)
        # Tarea 3: Update parent options in signals widget
        self.channel_spec_widget.update_parent_options(selected)

    def _selected_conc_columns(self) -> list[str]:
        selected: list[str] = []
        for i in range(self._columns_layout.count()):
            w = self._columns_layout.itemAt(i).widget()
            if isinstance(w, QCheckBox) and w.isChecked():
                selected.append(str(w.text()))
        return selected

    def _all_conc_columns(self) -> list[str]:
        columns: list[str] = []
        for i in range(self._columns_layout.count()):
            w = self._columns_layout.itemAt(i).widget()
            if isinstance(w, QCheckBox):
                columns.append(str(w.text()))
        return columns

    @staticmethod
    def _component_key(label: object) -> str:
        text = str(label or "").strip()
        if not text:
            return ""
        match = re.search(r"\w+", text, flags=re.UNICODE)
        if match:
            return match.group(0).strip().lower()
        return text.lower()

    def _ui_component_keys(self) -> list[str]:
        headers = self._selected_conc_columns() or self._all_conc_columns()
        keys: list[str] = []
        seen: set[str] = set()
        for header in headers:
            key = self._component_key(header)
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def _solver_component_names(self, solver_inputs: dict[str, Any]) -> list[str]:
        components_block = solver_inputs.get("components")
        if isinstance(components_block, dict):
            names = components_block.get("names")
            if isinstance(names, str):
                return [names]
            if isinstance(names, (list, tuple, np.ndarray)):
                return [str(x) for x in list(names)]
        if isinstance(components_block, (list, tuple, np.ndarray)):
            return [str(x) for x in list(components_block)]
        solver_block_raw = solver_inputs.get("solver_inputs")
        solver_block = solver_block_raw if isinstance(solver_block_raw, dict) else {}
        names = solver_block.get("components")
        if isinstance(names, str):
            return [names]
        if isinstance(names, (list, tuple, np.ndarray)):
            return [str(x) for x in list(names)]
        return []

    def _reorder_solver_inputs_to_ui(self, solver_inputs: dict[str, Any]) -> dict[str, Any]:
        ui_comps = self._ui_component_keys()
        solver_comps = self._solver_component_names(solver_inputs)
        if not ui_comps or not solver_comps:
            return solver_inputs

        solver_keys = [self._component_key(name) for name in solver_comps]
        used: set[int] = set()
        indices: list[int] = []
        for ui_key in ui_comps:
            for idx, solver_key in enumerate(solver_keys):
                if idx in used:
                    continue
                if solver_key == ui_key:
                    used.add(idx)
                    indices.append(idx)
                    break
        for idx in range(len(solver_comps)):
            if idx not in used:
                indices.append(idx)

        if indices == list(range(len(solver_comps))):
            return solver_inputs

        reordered = dict(solver_inputs)
        components_block = solver_inputs.get("components")
        if isinstance(components_block, dict):
            components_copy = dict(components_block)
            for key in ("names", "species", "indices", "total_concentrations"):
                raw = components_block.get(key)
                if raw is None:
                    continue
                raw_list = list(raw)
                if len(raw_list) == len(indices):
                    components_copy[key] = [raw_list[i] for i in indices]
            reordered["components"] = components_copy
        elif isinstance(components_block, (list, tuple, np.ndarray)) and len(components_block) == len(indices):
            reordered["components"] = [components_block[i] for i in indices]

        solver_block = dict(solver_inputs.get("solver_inputs") or {})
        model_raw = solver_block.get("modelo")
        if model_raw is not None:
            model = np.asarray(model_raw, dtype=float)
            if model.ndim == 2 and model.shape[0] == len(indices):
                solver_block["modelo"] = model[indices, :].tolist()

        ctot_raw = solver_block.get("ctot")
        if ctot_raw is not None:
            ctot = np.asarray(ctot_raw, dtype=float)
            if ctot.ndim == 2 and ctot.shape[1] == len(indices):
                solver_block["ctot"] = ctot[:, indices].tolist()

        solver_comp_raw = solver_block.get("components")
        if isinstance(solver_comp_raw, (list, tuple, np.ndarray)) and len(solver_comp_raw) == len(indices):
            solver_block["components"] = [solver_comp_raw[i] for i in indices]

        reordered["solver_inputs"] = solver_block
        return reordered

    def _populate_signals(self, columns: list[str]) -> None:
        self.channel_spec_widget.set_channels(columns or [])

    def _selected_signals(self) -> list[dict[str, str]]:
        return self.channel_spec_widget.get_selected_channels()

    # ---- Graph-driven model sync ----
    @Slot(object)
    def _on_equation_model_parsed(self, solver_inputs_obj: object) -> None:
        if not isinstance(solver_inputs_obj, dict):
            return
        solver_inputs = dict(solver_inputs_obj)
        try:
            solver_inputs = self._reorder_solver_inputs_to_ui(solver_inputs)
        except Exception as exc:
            self.log.append_text(f"[Graph] Component reorder skipped: {exc}")
        self._graph_solver_inputs = solver_inputs
        try:
            self._apply_solver_inputs_to_classic_matrix(solver_inputs)
            self._pending_graph_preview = True
            self._trigger_forward_preview_from_graph()
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

        # Apply @NA tags to both base components and complexes.
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

        # Si el sistema corresponde a una serie por pasos, mostrar en UI las K
        # originales del editor y usar Step by step para conservar equivalencia.
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

    def _trigger_forward_preview_from_graph(self) -> None:
        if not self._pending_graph_preview:
            return
        if self._worker is not None:
            return

        if self._preview_worker is not None:
            self._preview_worker.request_cancel()
            return

        try:
            config = self._collect_config()
            self._preflight_nmr_shapes(config)
            self._inject_stoichiometry_map(config, debug=False)
        except Exception:
            # No bloquea la edición si aún faltan entradas (archivo/columnas/señales).
            return

        initial_k = list(config.get("initial_k") or [])
        if not initial_k:
            return

        config["optimizer"] = "powell"
        config["multi_start_runs"] = 1
        fixed_mask = [True] * len(initial_k)
        config["fixed_mask"] = fixed_mask
        config["k_fixed"] = fixed_mask
        config["show_stability_diagnostics"] = False

        self._pending_graph_preview = False
        self._preview_worker = FitWorker(run_nmr_fit, config=config, parent=self)
        self._preview_thread = self._preview_worker.thread()
        self._preview_thread.finished.connect(
            self._on_preview_thread_finished, Qt.ConnectionType.QueuedConnection
        )
        self._preview_worker.result.connect(
            self._on_preview_result, Qt.ConnectionType.QueuedConnection
        )
        self._preview_worker.error.connect(
            self._on_preview_error, Qt.ConnectionType.QueuedConnection
        )
        self._preview_worker.finished.connect(
            self._on_preview_finished, Qt.ConnectionType.QueuedConnection
        )
        self._preview_worker.start()

    @Slot(object)
    def _on_preview_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._build_plot_state_from_result(result)
        self._render_current_plot()
        self.log.append_text("[Graph] Forward preview updated.")

    @Slot(str)
    def _on_preview_error(self, message: str) -> None:
        self.log.append_text(f"[Graph] Preview error: {message}")

    def _on_preview_finished(self) -> None:
        pass

    def _on_preview_thread_finished(self) -> None:
        self._preview_worker = None
        self._preview_thread = None
        if self._pending_graph_preview:
            self._trigger_forward_preview_from_graph()

    # ---- Run / Cancel / Results ----
    def _set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        self.btn_process.setEnabled(not running)
        self.btn_choose_file.setEnabled(not running)
        self.combo_conc_sheet.setEnabled(not running)
        self.combo_shift_sheet.setEnabled(not running)
        self.channel_spec_widget.setEnabled(not running)
        self.btn_signals_all.setEnabled(not running)
        self.btn_signals_none.setEnabled(not running)
        if self.equation_editor is not None:
            self.equation_editor.setEnabled(not running)
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

        selected_signals = self._selected_signals()
        if not selected_signals:
            raise ValueError("Select at least one signal in 'Chemical Shifts (Signals)'.")
            
        signal_names = [s["col_name"] for s in selected_signals]

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
            "nmr_sheet": nmr_sheet,
            "conc_sheet": conc_sheet,
            "column_names": column_names,
            "signal_names": signal_names,
            "signal_data": selected_signals, # Store parents too
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
            "show_stability_diagnostics": self.chk_show_diag.isChecked(),
            "multi_start_runs": runs,
        }
        config["equation_text"] = (
            self.equation_editor.get_text()
            if hasattr(self, "equation_editor") and self.equation_editor is not None
            else ""
        )
        if not config["equation_text"]:
            config["equation_text"] = ""
        if seeds is not None:
            config["multi_start_seeds"] = seeds
        return config

    def _preflight_nmr_shapes(self, config: dict[str, Any]) -> None:
        import pandas as pd

        file_path = str(config.get("file_path") or "")
        conc_sheet = str(config.get("conc_sheet") or "")
        column_names = [str(c) for c in (config.get("column_names") or [])]
        modelo = config.get("modelo") or []

        if not modelo:
            raise ValueError("Model matrix is empty. Define model dimensions and grid.")

        n_rows = len(modelo) if isinstance(modelo, list) else 0
        n_cols = len(modelo[0]) if n_rows and isinstance(modelo[0], list) else 0
        if n_rows == 0 or n_cols == 0:
            raise ValueError("Model matrix is empty. Define model dimensions and grid.")
        for row in modelo:
            if not isinstance(row, list) or len(row) != n_cols:
                raise ValueError("Model matrix rows have inconsistent length.")

        m_shape = (n_cols, n_rows)
        n_comp = n_cols
        nspec = n_rows

        df = pd.read_excel(file_path, sheet_name=conc_sheet, header=0)
        missing = [c for c in column_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing concentration columns: {missing}")

        ctot = df[column_names].to_numpy(dtype=float)
        if ctot.ndim != 2:
            raise ValueError("Concentration data shape is invalid.")

        ctot_row_shape = (ctot.shape[1],)
        self.log.append_text(
            f"Preflight NMR shapes: ctot_row={ctot_row_shape}, M={m_shape}, nspec={nspec}, n_comp={n_comp}"
        )

        if ctot.shape[1] != n_comp:
            raise ValueError(
                "Concentration columns "
                f"({ctot.shape[1]}) do not match model components ({n_comp}). "
                "Update Column names or the model dimensions."
            )

    def _inject_stoichiometry_map(self, config: dict[str, Any], *, debug: bool = True) -> None:
        model_mat = np.array(config.get("modelo") or [])
        comp_names = config.get("column_names") or []
        sig_data = config.get("signal_data") or []

        def _norm(text: str) -> str:
            return re.sub(r"\s+", " ", str(text or "").strip()).lower()

        comp_names_norm = [_norm(x) for x in comp_names]
        comp_to_idx = {name: i for i, name in enumerate(comp_names_norm)}

        if model_mat.size <= 0 or not sig_data:
            return

        stoich_cols = []
        for sig in sig_data:
            parent_raw = sig.get("parent", "Auto (1:1)")
            parent_norm = _norm(parent_raw)
            parent_idx = None
            if parent_norm.startswith("auto"):
                parent_idx = None
            elif parent_norm == "mezcla":
                parent_idx = None
            else:
                parent_idx = comp_to_idx.get(parent_norm)

            if parent_idx is not None:
                sig_stoich = model_mat[:, parent_idx]
            else:
                sig_stoich = np.ones(model_mat.shape[0])

            stoich_cols.append(sig_stoich)
            if debug:
                self.log.append_text(
                    f"[DEBUG] signal={sig.get('col_name')} parent='{parent_raw}' -> parent_idx={parent_idx}"
                )

        stoich_map = np.array(stoich_cols).T.tolist()
        config["stoichiometry_map"] = stoich_map
        if debug:
            self.log.append_text(
                f"Stoichiometry map generated ({len(stoich_map)}x{len(stoich_map[0])})."
            )

    def _on_process_clicked(self) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "A fit is already running. Cancel it first.")
            return
        try:
            config = self._collect_config()
            self._preflight_nmr_shapes(config)
            self._inject_stoichiometry_map(config, debug=True)

        except Exception as exc:
            self.log.append_text(f"ERROR: {exc}")
            QMessageBox.warning(self, "Config error", str(exc))
            return

        self.log.append_text("Iniciando optimización…")
        self._last_config = config
        self._last_result = None
        self._reset_plot_state()
        self.btn_save.setEnabled(False)
        self.canvas_main.clear()
        if self._preview_worker is not None:
            self._preview_worker.request_cancel()

        self._worker = FitWorker(run_nmr_fit, config=config, parent=self)
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
        self.log.append_text("Cancel requested…")

    @Slot(str)
    def _on_worker_progress(self, msg: str) -> None:
        self.log.append_text(str(msg))

    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
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

        plot_sources = build_nmr_plot_sources(result)
        shifts = plot_sources.get("nmr_shifts_fit") or {}
        signal_opts = shifts.get("signalOptions") or []
        x_vals = shifts.get("x") or []
        self.log.append_text(
            f"NMR result received. signals={len(signal_opts)}, points={len(x_vals)}"
        )
        if not signal_opts:
            self.log.append_text("NMR plot warning: no signals found for plot sources.")

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

        self._update_errors_context_from_result(result)

    @Slot(str)
    def _on_fit_error(self, message: str) -> None:
        self.log.append_text(f"ERROR: {message}")
        QMessageBox.critical(self, "Fit error", str(message))

    def _on_fit_finished(self) -> None:
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
        wanted_sigs = config.get("signal_data")
        if wanted_sigs and isinstance(wanted_sigs, list) and isinstance(wanted_sigs[0], dict):
            self.channel_spec_widget.set_selected_data(wanted_sigs)
        else:
            wanted_names = config.get("signal_names") or []
            self.channel_spec_widget.set_selected_channels(wanted_names)

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
        self.model_opt_plots.set_multi_start(
            config.get("multi_start_runs", 1),
            config.get("multi_start_seeds"),
        )
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
        if self._preview_worker is not None:
            self._preview_worker.request_cancel()

        self._file_path = ""
        self.lbl_file_status.setText("No file selected")
        self.combo_conc_sheet.clear()
        self.combo_shift_sheet.clear()
        self._clear_conc_columns()
        self._populate_signals([])
        if self.equation_editor is not None:
            self.equation_editor.set_text("")
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
        self._pending_graph_preview = False
        self.btn_save.setEnabled(False)

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

        model_matrix = export_data.get("modelo") or []
        modelo_solver = np.asarray(model_matrix, dtype=float).T if model_matrix else []

        context = {
            "technique": "nmr",
            "k_hat": k_hat,
            "C_T": export_data.get("C_T") or [],
            "dq": export_data.get("Chemical_Shifts") or [],
            "dq_fit": export_data.get("Calculated_Chemical_Shifts") or [],
            "y_fit_hat": export_data.get("Calculated_Chemical_Shifts") or [],
            "column_names": export_data.get("column_names") or [],
            "signal_names": export_data.get("signal_names") or [],
            "non_abs_species": export_data.get("non_absorbent_species") or [],
            "fixed_mask": export_data.get("fixed_mask") or [],
            "stoichiometry_map": export_data.get("stoichiometry_map") or [],
            "modelo_solver": modelo_solver,
            "algorithm": (self._last_config or {}).get("algorithm", "Newton-Raphson"),
            "model_settings": (self._last_config or {}).get("model_settings", "Free"),
            "optimizer": (self._last_config or {}).get("optimizer", "powell"),
            "bounds": (self._last_config or {}).get("bounds", []),
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

            from hmfit_core.processors.nmr_processor import (
                build_D_cols,
                project_coeffs_block_onp_frac,
            )
            from hmfit_core.processors.spectroscopy_processor import _build_bounds_list
            from hmfit_core.solvers import NewtonRaphson, LevenbergMarquardt

            dq_star = np.asarray(data_star, dtype=float)
            dq_obs_raw = ctx.get("dq")
            dq_obs = np.asarray(dq_obs_raw if dq_obs_raw is not None else [], dtype=float)
            c_t_raw = ctx.get("C_T")
            C_T = np.asarray(c_t_raw if c_t_raw is not None else [], dtype=float)
            modelo_raw = ctx.get("modelo_solver")
            modelo = np.asarray(modelo_raw if modelo_raw is not None else [], dtype=float)
            nas_raw = ctx.get("non_abs_species")
            nas = list(nas_raw) if nas_raw is not None else []
            algorithm = str(ctx.get("algorithm") or "Newton-Raphson")
            model_settings = str(ctx.get("model_settings") or "Free")
            optimizer = str(ctx.get("optimizer") or "powell")
            bounds_raw = ctx.get("bounds")
            bounds_raw = list(bounds_raw) if bounds_raw is not None else []
            column_names_raw = ctx.get("column_names")
            column_names = list(column_names_raw) if column_names_raw is not None else []
            signal_names_raw = ctx.get("signal_names")
            signal_names = list(signal_names_raw) if signal_names_raw is not None else []
            stoich_raw = ctx.get("stoichiometry_map")
            stoichiometry = None
            if stoich_raw is not None:
                try:
                    stoichiometry = np.asarray(stoich_raw, dtype=float)
                    if stoichiometry.size == 0:
                        stoichiometry = None
                except Exception:
                    stoichiometry = None

            if algorithm == "Newton-Raphson":
                res = NewtonRaphson(C_T, modelo, nas, model_settings)
            elif algorithm == "Levenberg-Marquardt":
                res = LevenbergMarquardt(C_T, modelo, nas, model_settings)
            else:
                return np.asarray(theta0, dtype=float), False, {"error": f"Unknown algorithm: {algorithm}"}

            D_cols = ctx.get("D_cols")
            if D_cols is None or len(np.asarray(D_cols).shape) == 0:
                if not column_names or not signal_names:
                    return np.asarray(theta0, dtype=float), False, {"error": "Missing D_cols and signal metadata."}
                D_cols, _ = build_D_cols(C_T, column_names, signal_names, default_idx=0)
            D_cols = np.asarray(D_cols, dtype=float)

            mask = ctx.get("mask")
            if mask is None:
                base = dq_obs if dq_obs.size else dq_star
                mask = np.isfinite(base) & np.isfinite(D_cols) & (np.abs(D_cols) > 0)
            mask = np.asarray(mask, dtype=bool)

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
                    dq_cal = project_coeffs_block_onp_frac(
                        dq_star, C, D_cols, mask, stoichiometry=stoichiometry, rcond=1e-10, ridge=1e-8
                    )
                    diff = dq_star - dq_cal
                    valid_residuals = mask & np.isfinite(dq_cal)
                    r = diff[valid_residuals].ravel()
                    if (r.size <= np.asarray(theta_free).ravel().size) or (not np.isfinite(r).all()):
                        return 1e9
                    return float(np.sqrt(np.mean(r * r)))
                except Exception:
                    return 1e9

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
