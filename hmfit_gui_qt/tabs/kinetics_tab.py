from __future__ import annotations

import copy
import json
from typing import Any, Sequence

import numpy as np

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
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

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import fit_global
from hmfit.kinetics.gui.import_wizard import ImportWizard
from hmfit.kinetics.mechanism_editor.parser import MechanismParseError, parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit_core.exports import write_results_xlsx
from hmfit_gui_qt.plots.kinetics_registry import build_kinetics_registry
from hmfit_gui_qt.plots.kinetics_sources import build_kinetics_plot_sources
from hmfit_gui_qt.plots.plot_controller import PlotController
from hmfit_gui_qt.widgets.kinetics_model_opt_plots import (
    KineticsModelOptPlotsWidget,
    KineticsParamState,
)
from hmfit_gui_qt.widgets.log_console import LogConsole
from hmfit_gui_qt.widgets.mpl_canvas import MplCanvas, NavigationToolbar
from hmfit_gui_qt.workers.fit_worker import FitWorker


class KineticsTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[KineticsFitDataset] = []
        self._model: KineticsModel | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_config: dict[str, Any] | None = None
        self._plot_controller: PlotController | None = None
        self._worker: FitWorker | None = None
        self._thread: QThread | None = None
        self._param_defaults: KineticsParamState | None = None
        self._is_running = False
        self._in_metadata_update = False

        self._build_ui()
        self._plot_controller = PlotController(
            canvas=self.canvas_main,
            log=self.log,
            model_opt_plots=self.model_opt_plots,
            btn_prev=self.btn_plot_prev,
            btn_next=self.btn_plot_next,
            lbl_title=self.lbl_plot_title,
            registry=build_kinetics_registry(),
            build_plot_data=build_kinetics_plot_sources,
            legacy_title_for_key=self._plot_title_for_key,
            legacy_order=[
                "kinetics_concentrations",
                "kinetics_d_fit",
                "kinetics_residuals",
                "kinetics_a_profiles",
            ],
            default_title="Kinetics plots",
            is_running=lambda: self._is_running,
        )
        self._wire_plot_controls()
        self._reset_plot_state()
        self.log.append_text("Listo. Carga datasets para comenzar.")
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

        data_layout.addWidget(QLabel("Datasets", self._data_group))
        self._dataset_list = QListWidget(self._data_group)
        self._dataset_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._dataset_list.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        data_layout.addWidget(self._dataset_list, 1)

        row1 = QHBoxLayout()
        self.btn_import = QPushButton("Import...", self._data_group)
        self.btn_import.clicked.connect(self._on_import)
        row1.addWidget(self.btn_import)
        self.btn_duplicate = QPushButton("Duplicate", self._data_group)
        self.btn_duplicate.clicked.connect(self._on_duplicate)
        row1.addWidget(self.btn_duplicate)
        self.btn_delete = QPushButton("Delete", self._data_group)
        self.btn_delete.clicked.connect(self._on_delete)
        row1.addWidget(self.btn_delete)
        data_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_fit_selected = QPushButton("Fit selected", self._data_group)
        self.btn_fit_selected.clicked.connect(self._on_fit_selected)
        row2.addWidget(self.btn_fit_selected)
        row2.addStretch(1)
        data_layout.addLayout(row2)

        self.lbl_dataset_status = QLabel("", self._data_group)
        self.lbl_dataset_status.setStyleSheet("color: #B00020;")
        self.lbl_dataset_status.setWordWrap(True)
        data_layout.addWidget(self.lbl_dataset_status)

        left_layout.addWidget(self._data_group)

        # ----- Sub-tabs: Model/Optimization/Plots/Errors -----
        self.model_opt_plots = KineticsModelOptPlotsWidget(left_container)
        self.model_opt_plots.validate_requested.connect(self._on_validate_mechanism)
        self.model_opt_plots.import_requested.connect(self._on_import_config)
        self.model_opt_plots.export_requested.connect(self._on_export_config)
        self.model_opt_plots.reset_requested.connect(self.reset_tab)
        self.model_opt_plots.process_requested.connect(self._on_process_clicked)
        self.model_opt_plots.cancel_requested.connect(self._on_cancel_clicked)
        self.model_opt_plots.save_requested.connect(self._on_save_results_clicked)
        self.model_opt_plots.y0_table.itemChanged.connect(self._on_metadata_table_changed)
        self.model_opt_plots.y0_table.cellChanged.connect(self._on_metadata_table_changed)
        self.model_opt_plots.fixed_table.itemChanged.connect(self._on_metadata_table_changed)
        self.model_opt_plots.fixed_table.cellChanged.connect(self._on_metadata_table_changed)
        self.model_opt_plots.metadata_changed.connect(self._on_metadata_table_changed)
        left_layout.addWidget(self.model_opt_plots, 1)

        # ----- Right panel: main plot + diagnostics -----
        right_split = QSplitter(Qt.Orientation.Vertical, self._main_split)
        main_plot_panel = QWidget(right_split)
        main_plot_layout = QVBoxLayout(main_plot_panel)
        main_plot_layout.setContentsMargins(0, 0, 0, 0)
        nav = QHBoxLayout()
        self.btn_plot_prev = QPushButton("Prev", main_plot_panel)
        self.btn_plot_prev.clicked.connect(lambda: self._navigate_plot(-1))
        nav.addWidget(self.btn_plot_prev)
        self.lbl_plot_title = QLabel("Kinetics plots", main_plot_panel)
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

        self._main_split.addWidget(left_scroll)
        self._main_split.addWidget(right_split)

        self._main_split.setStretchFactor(0, 0)
        self._main_split.setStretchFactor(1, 1)

        self._update_plot_nav()

    # ---- Plot navigation ----
    def _plot_title_for_key(self, key: str) -> str:
        titles = {
            "kinetics_concentrations": "C(t) profiles",
            "kinetics_d_fit": "D vs D_hat",
            "kinetics_residuals": "Residuals",
            "kinetics_a_profiles": "A profiles",
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

    # ---- Dataset management ----
    def _on_import(self) -> None:
        wizard = ImportWizard(
            self,
            dynamic_species=self._dynamic_species(),
            fixed_species=self._fixed_species(),
        )
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return
        datasets = wizard.datasets
        if not datasets:
            return
        self._datasets.extend(datasets)
        self._refresh_dataset_list()
        self._dataset_list.setCurrentRow(len(self._datasets) - 1)
        self._update_fit_state()

    def _on_duplicate(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            return
        cloned = copy.deepcopy(dataset)
        cloned.name = f"{dataset.name} (copy)"
        self._datasets.append(cloned)
        self._refresh_dataset_list()
        self._dataset_list.setCurrentRow(len(self._datasets) - 1)
        self._update_fit_state()

    def _on_delete(self) -> None:
        indices = self._selected_indices()
        if not indices:
            return
        for idx in reversed(indices):
            del self._datasets[idx]
        self._refresh_dataset_list()
        self._update_fit_state()

    def _on_dataset_selection_changed(self) -> None:
        selected = self._selected_datasets()
        if selected:
            self._load_dataset_metadata(selected[0])
            self.model_opt_plots.y0_table.setEnabled(True)
            self.model_opt_plots.fixed_table.setEnabled(True)
        else:
            self.model_opt_plots.y0_table.setEnabled(False)
            self.model_opt_plots.fixed_table.setEnabled(False)
        self._update_fit_state()

    def _refresh_dataset_list(self) -> None:
        selected_indices = self._selected_indices()
        self._dataset_list.blockSignals(True)
        self._dataset_list.clear()
        for dataset in self._datasets:
            item = QListWidgetItem(dataset.name or "Dataset")
            item.setData(Qt.ItemDataRole.UserRole, dataset)
            if not self._dataset_is_complete(dataset):
                item.setForeground(QColor("#B00020"))
                item.setToolTip(
                    "Initial concentrations (y0) are required. "
                    "Edit y0/fixed in the Model tab."
                )
            self._dataset_list.addItem(item)
        self._dataset_list.blockSignals(False)
        for idx in selected_indices:
            if 0 <= idx < self._dataset_list.count():
                self._dataset_list.item(idx).setSelected(True)

    def _selected_indices(self) -> list[int]:
        return [idx.row() for idx in self._dataset_list.selectionModel().selectedRows()]

    def _selected_dataset(self) -> KineticsFitDataset | None:
        row = self._dataset_list.currentRow()
        if row < 0 or row >= len(self._datasets):
            return None
        return self._datasets[row]

    def _selected_datasets(self) -> list[KineticsFitDataset]:
        indices = self._selected_indices()
        return [self._datasets[idx] for idx in indices if 0 <= idx < len(self._datasets)]

    # ---- Model ----
    def _on_validate_mechanism(self) -> None:
        text = self.model_opt_plots.mechanism_text().strip()
        if not text:
            QMessageBox.warning(self, "Mechanism", "Mechanism text is empty.")
            return
        try:
            ast = parse_mechanism(text)
        except MechanismParseError as exc:
            self.model_opt_plots.set_summary_text(f"Parse error: {exc}")
            self._model = None
            self._update_fit_state()
            return

        self._model = KineticsModel(ast)
        params = _extract_param_names(ast)
        self.model_opt_plots.set_params(params)
        self._param_defaults = self.model_opt_plots.get_params_state()

        summary = [
            f"Species: {', '.join(ast.species)}",
            f"Fixed: {', '.join(sorted(ast.fixed)) if ast.fixed else 'none'}",
            f"Reactions: {len(ast.reactions)}",
            f"Parameters: {', '.join(params)}",
        ]
        self.model_opt_plots.set_summary_text("\n".join(summary))
        self._in_metadata_update = True
        try:
            self.model_opt_plots.set_species_tables(
                self._dynamic_species(), self._fixed_species()
            )
        finally:
            self._in_metadata_update = False
        current = self._selected_dataset()
        if current is not None:
            self._load_dataset_metadata(current)
        self._refresh_dataset_list()
        self._update_fit_state()

    # ---- Fit ----
    def _on_process_clicked(self) -> None:
        self._run_fit(self._datasets, title="Fit")

    def _on_fit_selected(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            QMessageBox.warning(self, "Fit selected", "Select one or more datasets.")
            return
        self._run_fit(selected, title="Fit selected")

    def _run_fit(self, datasets: Sequence[KineticsFitDataset], *, title: str) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Busy", "A fit is already running. Cancel it first.")
            return
        if self._model is None:
            QMessageBox.warning(self, title, "Validate a mechanism first.")
            return
        if not datasets:
            QMessageBox.warning(self, title, "No datasets loaded.")
            return
        self._sync_selected_metadata_from_tables()
        if not self._datasets_complete(datasets):
            QMessageBox.warning(
                self, title, "Some datasets are incomplete (missing y0/fixed)."
            )
            return

        try:
            config = self._collect_fit_config(datasets)
        except ValueError as exc:
            QMessageBox.warning(self, title, str(exc))
            return

        self.log.append_text("Iniciando optimizacion...")
        self._last_config = config
        self._last_result = None
        self._reset_plot_state()
        self.model_opt_plots.btn_save.setEnabled(False)
        self.canvas_main.clear()

        self._worker = FitWorker(_run_kinetics_fit, config=config, parent=self)
        self._thread = self._worker.thread()
        self._thread.finished.connect(self._on_fit_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.progress.connect(self._on_worker_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.result.connect(self._on_fit_result, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_fit_error, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_fit_finished, Qt.ConnectionType.QueuedConnection)
        self._set_running(True)
        self._worker.start()

    def _collect_fit_config(self, datasets: Sequence[KineticsFitDataset]) -> dict[str, Any]:
        if self._model is None:
            raise ValueError("Validate a mechanism first.")

        params0, bounds, param_names, log_params = self._read_params_from_table()
        method = self.model_opt_plots.combo_algorithm.currentText().strip().lower()
        backend = self.model_opt_plots.combo_backend.currentText().strip().lower()
        nnls = self.model_opt_plots.chk_nnls.isChecked()
        if backend == "jax" and nnls:
            raise ValueError("NNLS is not supported with JAX backend.")
        runs, seeds = self.model_opt_plots.get_multi_start()
        if method == "multistart" and runs <= 1:
            runs = 10
        config = {
            "model": self._model,
            "datasets": list(datasets),
            "params0": params0,
            "bounds": bounds,
            "param_names": param_names,
            "log_params": list(log_params),
            "method": method,
            "backend": backend,
            "nnls": nnls,
            "multi_start_runs": runs,
            "multi_start_seeds": seeds,
            "show_stability_diagnostics": self.chk_show_diag.isChecked(),
        }
        return config

    def _read_params_from_table(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[str], set[str]]:
        state = self.model_opt_plots.get_params_state()
        param_names = list(state.param_names)
        if not param_names:
            raise ValueError("No parameters defined.")

        values = []
        lower = []
        upper = []
        log_params: set[str] = set()
        for name, value, bounds, fixed, log_flag in zip(
            state.param_names,
            state.values,
            state.bounds,
            state.fixed_mask,
            state.log_mask,
            strict=True,
        ):
            min_val, max_val = bounds
            if fixed:
                min_val = value
                max_val = value
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError(f"Min > Max for parameter '{name}'.")

            opt_value = float(value)
            opt_min = float(min_val) if min_val is not None else -np.inf
            opt_max = float(max_val) if max_val is not None else np.inf
            if log_flag:
                if value <= 0:
                    raise ValueError(f"Parameter '{name}' must be > 0 for log10.")
                if min_val is not None and min_val <= 0:
                    raise ValueError(f"Min for '{name}' must be > 0 for log10.")
                if max_val is not None and max_val <= 0:
                    raise ValueError(f"Max for '{name}' must be > 0 for log10.")
                opt_value = float(np.log10(value))
                opt_min = float(np.log10(min_val)) if min_val is not None else -np.inf
                opt_max = float(np.log10(max_val)) if max_val is not None else np.inf
                log_params.add(name)

            values.append(opt_value)
            lower.append(opt_min)
            upper.append(opt_max)

        params0 = np.array(values, dtype=float)
        bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))
        return params0, bounds, param_names, log_params

    def _on_cancel_clicked(self) -> None:
        if self._worker is None:
            return
        self._worker.request_cancel()
        self.log.append_text("Cancel requested...")

    @Slot(str)
    def _on_worker_progress(self, msg: str) -> None:
        self.log.append_text(str(msg))

    def _on_fit_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._last_result = result
        self.model_opt_plots.btn_save.setEnabled(bool(result.get("success", True)))

        fit_results = result.get("fit_results") or []
        datasets = result.get("datasets") or []
        for dataset, fit_info in zip(datasets, fit_results, strict=False):
            dataset.fit_C = fit_info.get("C")
            dataset.fit_A = fit_info.get("A")
            dataset.fit_D_hat = fit_info.get("D_hat")
            dataset.fit_residuals = fit_info.get("residuals")

        self._build_plot_state_from_result(result)
        self._render_current_plot()

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
        self._update_fit_state()

    @Slot(str)
    def _on_fit_error(self, message: str) -> None:
        self.log.append_text(f"ERROR: {message}")
        QMessageBox.critical(self, "Fit error", str(message))

    def _on_fit_finished(self) -> None:
        self._set_running(False)

    def _on_fit_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    # ---- Errors ----
    def _update_errors_context_from_result(self, result: dict[str, Any]) -> None:
        param_names = list(result.get("param_names") or [])
        params = result.get("params") or {}
        k_hat = np.array([float(params.get(name, 0.0)) for name in param_names], dtype=float)
        residuals_raw = result.get("residuals")
        residuals = np.asarray(
            residuals_raw if residuals_raw is not None else [], dtype=float
        ).ravel()
        jac = result.get("jac")
        jac_arr = None
        if jac is not None:
            jac_arr = np.asarray(jac, dtype=float)
        covariance = result.get("covariance")
        sigma2 = result.get("sigma2")
        if sigma2 is None and residuals.size and k_hat.size:
            dof = max(residuals.size - k_hat.size, 1)
            sigma2 = float(np.sum(residuals**2) / float(dof))

        datasets = result.get("datasets") or []
        d_obs = [ds.D for ds in datasets if isinstance(ds, KineticsFitDataset)]
        d_hat = [ds.fit_D_hat for ds in datasets if isinstance(ds, KineticsFitDataset)]

        errors_ctx = {
            "param_names": param_names,
            "k_hat": k_hat,
            "residuals": residuals,
            "jac": jac_arr,
            "covariance": covariance,
            "sigma2": sigma2,
            "D_obs": d_obs,
            "D_hat": d_hat,
            "log_params": list(result.get("log_params") or []),
            "refit_fn": result.get("refit_fn"),
        }
        self.model_opt_plots.set_errors_context(errors_ctx, auto_compute=False)

    # ---- Import/Export config ----
    def _collect_config(self) -> dict[str, Any]:
        state = self.model_opt_plots.get_params_state()
        params = {}
        for name, value, bounds, fixed, log_flag in zip(
            state.param_names,
            state.values,
            state.bounds,
            state.fixed_mask,
            state.log_mask,
            strict=True,
        ):
            params[name] = {
                "value": value,
                "min": bounds[0],
                "max": bounds[1],
                "fixed": fixed,
                "log": log_flag,
            }
        runs, seeds = self.model_opt_plots.get_multi_start()
        config = {
            "mechanism_text": self.model_opt_plots.mechanism_text(),
            "params": params,
            "algorithm": self.model_opt_plots.combo_algorithm.currentText(),
            "backend": self.model_opt_plots.combo_backend.currentText(),
            "nnls": self.model_opt_plots.chk_nnls.isChecked(),
            "multi_start_runs": runs,
            "multi_start_seeds": seeds,
            "y0": self.model_opt_plots.get_y0_values(),
            "fixed_conc": self.model_opt_plots.get_fixed_conc_values(),
            "show_stability_diagnostics": self.chk_show_diag.isChecked(),
        }
        return config

    def _on_export_config(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export config", "kinetics_config.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=True)
            self.log.append_text(f"Config exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _on_import_config(self) -> None:
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
            self._apply_config(config)
            self.log.append_text(f"Config imported from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Config error", str(exc))

    def _apply_config(self, config: dict[str, Any]) -> None:
        mech_text = str(config.get("mechanism_text") or "")
        if mech_text:
            self.model_opt_plots.set_mechanism_text(mech_text)
            self._on_validate_mechanism()

        params_raw = config.get("params") or {}
        if isinstance(params_raw, dict) and params_raw:
            names = list(params_raw.keys())
            values = []
            bounds = []
            fixed_mask = []
            log_mask = []
            for name in names:
                entry = params_raw.get(name) or {}
                values.append(float(entry.get("value", 1.0)))
                bounds.append((entry.get("min"), entry.get("max")))
                fixed_mask.append(bool(entry.get("fixed", False)))
                log_mask.append(bool(entry.get("log", False)))
            state = KineticsParamState(
                param_names=names,
                values=values,
                bounds=bounds,
                fixed_mask=fixed_mask,
                log_mask=log_mask,
            )
            self.model_opt_plots.set_params_state(state)

        algo = str(config.get("algorithm") or "")
        if algo:
            ix = self.model_opt_plots.combo_algorithm.findText(algo)
            if ix >= 0:
                self.model_opt_plots.combo_algorithm.setCurrentIndex(ix)

        backend = str(config.get("backend") or "")
        if backend:
            ix = self.model_opt_plots.combo_backend.findText(backend)
            if ix >= 0:
                self.model_opt_plots.combo_backend.setCurrentIndex(ix)

        self.model_opt_plots.chk_nnls.setChecked(bool(config.get("nnls", False)))
        if "show_stability_diagnostics" in config:
            self.chk_show_diag.setChecked(bool(config.get("show_stability_diagnostics")))

        runs = config.get("multi_start_runs")
        seeds = config.get("multi_start_seeds")
        self.model_opt_plots.set_multi_start(runs, seeds)

        y0 = config.get("y0") if isinstance(config.get("y0"), dict) else None
        fixed = config.get("fixed_conc") if isinstance(config.get("fixed_conc"), dict) else None
        if y0 is not None:
            self.model_opt_plots.set_y0_values(y0)
        if fixed is not None:
            self.model_opt_plots.set_fixed_conc_values(fixed)
        self._update_fit_state()

    # ---- Reset / Save ----
    def reset_tab(self) -> None:
        for dataset in self._datasets:
            self._clear_fit_results(dataset)
        self._last_result = None
        self._reset_plot_state()
        self.canvas_main.clear()
        self.lbl_stability_light.setText("Stability: -")
        self.model_opt_plots.set_errors_context(None)
        if self._param_defaults is not None:
            self.model_opt_plots.set_params_state(self._param_defaults)
        self._update_fit_state()

    def _on_save_results_clicked(self) -> None:
        if not self._last_result:
            QMessageBox.information(self, "Save results", "Run a fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", "kinetics_results.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        constants = []
        params = self._last_result.get("params") or {}
        for name, value in params.items():
            constants.append({"Parameter": name, "Value": value})
        stats = self._last_result.get("statistics") or {}
        results_text = self._last_result.get("results_text") or ""
        try:
            write_results_xlsx(
                path,
                constants=constants,
                statistics=stats,
                results_text=results_text,
                export_data={},
            )
            self.log.append_text(f"Results saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    # ---- Helpers ----
    def _set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        self.btn_import.setEnabled(not running)
        self.btn_duplicate.setEnabled(not running)
        self.btn_delete.setEnabled(not running)
        self.btn_fit_selected.setEnabled(not running)
        self.model_opt_plots.tabs.setEnabled(not running)
        self.model_opt_plots.btn_import.setEnabled(not running)
        self.model_opt_plots.btn_export.setEnabled(not running)
        self.model_opt_plots.btn_reset.setEnabled(not running)
        self.model_opt_plots.btn_process.setEnabled(not running)
        self.model_opt_plots.btn_cancel.setEnabled(running)
        self.model_opt_plots.btn_save.setEnabled(
            bool(self._last_result) and bool(self._last_result.get("success", True)) and not running
        )
        self.btn_plot_prev.setEnabled(not running)
        self.btn_plot_next.setEnabled(not running)

    def _update_fit_state(self) -> None:
        fit_enabled = (
            self._model is not None
            and bool(self._datasets)
            and self._datasets_complete()
        )
        self.model_opt_plots.btn_process.setEnabled(fit_enabled and not self._is_running)
        selected = self._selected_datasets()
        fit_selected_enabled = (
            self._model is not None
            and bool(selected)
            and self._datasets_complete(selected)
        )
        self.btn_fit_selected.setEnabled(fit_selected_enabled and not self._is_running)

        if not self._datasets:
            self.lbl_dataset_status.setText("Import datasets to start.")
        elif not self._datasets_complete():
            self.lbl_dataset_status.setText(
                "Missing y0 or fixed concentrations for some datasets. "
                "Edit y0/fixed in the Model tab."
            )
        else:
            self.lbl_dataset_status.setText("")

    def _clear_fit_results(self, dataset: KineticsFitDataset) -> None:
        dataset.fit_C = None
        dataset.fit_A = None
        dataset.fit_D_hat = None
        dataset.fit_residuals = None

    def _dataset_is_complete(self, dataset: KineticsFitDataset) -> bool:
        dynamic = set(self._dynamic_species())
        fixed = set(self._fixed_species())
        if dataset.D is None:
            return False
        if dynamic:
            if dataset.y0 is None:
                return False
            if any(species not in dataset.y0 for species in dynamic):
                return False
        if fixed:
            if dataset.fixed_conc is None:
                return False
            if any(species not in dataset.fixed_conc for species in fixed):
                return False
        return True

    def _datasets_complete(
        self, datasets: Sequence[KineticsFitDataset] | None = None
    ) -> bool:
        datasets = list(datasets) if datasets is not None else self._datasets
        return all(self._dataset_is_complete(dataset) for dataset in datasets)

    def _dynamic_species(self) -> list[str]:
        if self._model is None:
            return []
        return list(self._model.dynamic_species)

    def _fixed_species(self) -> list[str]:
        if self._model is None:
            return []
        return list(self._model.fixed_species)

    def _load_dataset_metadata(self, dataset: KineticsFitDataset) -> None:
        self._in_metadata_update = True
        try:
            self.model_opt_plots.set_y0_values(dataset.y0 or {})
            self.model_opt_plots.set_fixed_conc_values(dataset.fixed_conc or {})
        finally:
            self._in_metadata_update = False

    def _sync_selected_metadata_from_tables(self) -> None:
        if self._in_metadata_update:
            return
        datasets = self._datasets
        if not datasets:
            return
        y0_values = self.model_opt_plots.get_y0_values()
        fixed_values = self.model_opt_plots.get_fixed_conc_values()
        for dataset in datasets:
            dataset.y0 = dict(y0_values)
            dataset.fixed_conc = dict(fixed_values)

    def _on_metadata_table_changed(self, _item: object | None = None) -> None:
        if self._in_metadata_update:
            return
        self._sync_selected_metadata_from_tables()
        self._refresh_dataset_list()
        self._update_fit_state()


def _extract_param_names(ast) -> list[str]:
    params = set()
    for reaction in ast.reactions:
        params.add(reaction.k_forward)
        if reaction.k_reverse:
            params.add(reaction.k_reverse)

    for k_name, temp_model in ast.temp_models.items():
        if k_name in params:
            params.remove(k_name)
        params.update(temp_model.params.values())

    return sorted(params)


def _params_vector_to_dict(
    x: np.ndarray, param_names: Sequence[str], log_params: set[str]
) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, value in zip(param_names, x, strict=True):
        if name in log_params:
            params[name] = float(10.0 ** float(value))
        else:
            params[name] = float(value)
    return params


def _compute_covariance(
    result,
    param_names: Sequence[str],
    log_params: set[str],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    errors = None
    covariance = None
    correlations = None
    condition_number = None
    if result.jac is None:
        return errors, covariance, correlations, condition_number
    J = np.asarray(result.jac, dtype=float)
    if J.size == 0:
        return errors, covariance, correlations, condition_number
    n_data = result.fun.size if result.fun is not None else J.shape[0]
    m = J.shape[1]
    dof = max(n_data - m, 1)
    sigma2 = 2.0 * result.cost / dof
    jt_j = J.T @ J
    try:
        condition_number = float(np.linalg.cond(jt_j))
    except np.linalg.LinAlgError:
        condition_number = None
    covariance = sigma2 * np.linalg.pinv(jt_j)
    if log_params:
        scale = np.ones(m, dtype=float)
        for idx, name in enumerate(param_names):
            if name in log_params:
                params = _params_vector_to_dict(result.x, param_names, log_params)
                scale[idx] = np.log(10.0) * float(params[name])
        covariance = (scale[:, None] * covariance) * scale[None, :]
    errors = np.sqrt(np.diag(covariance))
    denom = np.outer(errors, errors)
    correlations = np.divide(
        covariance,
        denom,
        out=np.zeros_like(covariance),
        where=denom != 0,
    )
    return errors, covariance, correlations, condition_number


def _randomize_params(
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    lower, upper = bounds
    x = np.array(x0, dtype=float)
    for idx in range(x.size):
        lo = lower[idx] if idx < lower.size else -np.inf
        hi = upper[idx] if idx < upper.size else np.inf
        if np.isfinite(lo) and np.isfinite(hi):
            x[idx] = rng.uniform(lo, hi)
        else:
            scale = 0.2 if x0[idx] != 0 else 1.0
            x[idx] = x0[idx] + rng.normal(0.0, scale)
    return x


def _run_kinetics_fit(
    config: dict[str, Any],
    *,
    progress_cb=None,
    cancel=None,
) -> dict[str, Any]:
    from hmfit_core.api import FitCancelled
    from hmfit_core.utils.errors import build_identifiability_report

    model = config["model"]
    datasets = list(config["datasets"] or [])
    params0 = np.asarray(config["params0"], dtype=float)
    bounds = config["bounds"]
    param_names = list(config["param_names"] or [])
    log_params = set(config.get("log_params") or [])
    method = str(config.get("method") or "least_squares")
    backend = str(config.get("backend") or "scipy")
    nnls = bool(config.get("nnls", False))
    show_stability_diagnostics = bool(config.get("show_stability_diagnostics", False))
    runs = int(config.get("multi_start_runs") or 1)
    seeds = config.get("multi_start_seeds")
    seeds_list = list(seeds) if isinstance(seeds, (list, tuple)) else None

    if nnls and backend == "jax":
        raise ValueError("NNLS is not supported with JAX backend.")

    if method == "trust-region":
        ls_method = "trf"
    else:
        ls_method = "trf"

    objective = GlobalKineticsObjective(
        model,
        datasets,
        param_names=param_names,
        nnls=nnls,
        log_params=log_params,
    )

    def _run_once(x0: np.ndarray) -> dict[str, Any]:
        if backend == "jax":
            return _run_once_jax(x0)
        result = fit_global(
            objective,
            x0,
            bounds=bounds,
            method=ls_method,
            max_nfev=300,
        )
        payload = {
            "params": result.params,
            "ssq": result.ssq,
            "nfev": result.nfev,
            "njev": result.njev,
            "jac": result.result.jac if result.result is not None else None,
            "residuals": result.result.fun if result.result is not None else None,
            "errors": result.errors,
            "covariance": result.covariance,
            "correlations": result.correlations,
            "condition_number": result.condition_number,
            "raw_result": result.result,
        }
        return payload

    def _run_once_jax(x0: np.ndarray) -> dict[str, Any]:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise ValueError("JAX backend is not available.") from exc

        from scipy.optimize import least_squares
        from hmfit.kinetics.observation.linear_matrix import prepare_weights

        jax_datasets = []
        for dataset in datasets:
            if dataset.D is None:
                raise ValueError("Dataset is missing observed data D.")
            weights = None
            if dataset.weights is not None or dataset.sigma is not None:
                weights = prepare_weights(
                    dataset.weights if dataset.weights is not None else 1.0 / dataset.sigma,
                    dataset.D.shape[0],
                    dataset.D.shape[1],
                )
            jax_datasets.append(
                {
                    "dataset": dataset,
                    "t": jnp.asarray(dataset.t, dtype=float),
                    "D": jnp.asarray(dataset.D, dtype=float),
                    "weights": jnp.asarray(weights, dtype=float) if weights is not None else None,
                }
            )

        def _params_from_vector(x_vec):
            params = {}
            for name, value in zip(param_names, x_vec, strict=True):
                if name in log_params:
                    params[name] = jnp.power(10.0, value)
                else:
                    params[name] = value
            return params

        def _solve_A_ls(C, D, weights=None):
            if weights is None:
                A = jnp.linalg.pinv(C) @ D
                return A
            if weights.ndim == 1 or weights.shape[1] == 1:
                w = weights.reshape(-1, 1)
                Cw = C * w
                Dw = D * w
                return jnp.linalg.pinv(Cw) @ Dw
            cols = []
            for j in range(D.shape[1]):
                w = weights[:, j : j + 1]
                Cw = C * w
                Dw = D[:, j : j + 1] * w
                cols.append(jnp.linalg.pinv(Cw) @ Dw)
            return jnp.concatenate(cols, axis=1)

        def residuals_jax(x_vec):
            params = _params_from_vector(x_vec)
            resid_list = []
            for entry in jax_datasets:
                ds = entry["dataset"]
                C = model.solve_concentrations_jax(ds.t, ds.y0, params, ds)
                A = _solve_A_ls(C, entry["D"], weights=entry["weights"])
                D_hat = C @ A
                resid = entry["D"] - D_hat
                if entry["weights"] is not None:
                    resid = resid * entry["weights"]
                resid_list.append(jnp.ravel(resid))
            if not resid_list:
                return jnp.array([])
            return jnp.concatenate(resid_list)

        jac_fn = jax.jacrev(residuals_jax)

        def residuals_np(x_vec):
            return np.asarray(residuals_jax(jnp.asarray(x_vec, dtype=float)))

        def jac_np(x_vec):
            return np.asarray(jac_fn(jnp.asarray(x_vec, dtype=float)))

        result = least_squares(
            residuals_np,
            x0=x0,
            bounds=bounds,
            method=ls_method,
            jac=jac_np,
            max_nfev=300,
        )

        params = _params_vector_to_dict(result.x, param_names, log_params)
        errors, covariance, correlations, condition_number = _compute_covariance(
            result, param_names, log_params
        )
        payload = {
            "params": params,
            "ssq": float(2.0 * result.cost),
            "nfev": result.nfev,
            "njev": result.njev,
            "jac": result.jac,
            "residuals": result.fun,
            "errors": errors,
            "covariance": covariance,
            "correlations": correlations,
            "condition_number": condition_number,
            "raw_result": result,
        }
        return payload

    best_payload = None
    best_ssq = np.inf
    total_runs = max(1, runs)
    run_seeds: list[int | None] = []
    if seeds_list:
        run_seeds = list(seeds_list)
    else:
        run_seeds = [None] * total_runs

    for run_idx in range(total_runs):
        if cancel is not None and cancel():
            raise FitCancelled("Fit cancelled.")
        seed = run_seeds[run_idx] if run_idx < len(run_seeds) else None
        rng = np.random.default_rng(seed)
        x0 = params0 if run_idx == 0 else _randomize_params(params0, bounds, rng)
        if progress_cb is not None:
            progress_cb(f"Run {run_idx + 1}/{total_runs}...")
        payload = _run_once(x0)
        if payload["ssq"] < best_ssq:
            best_ssq = payload["ssq"]
            best_payload = payload

    if best_payload is None:
        raise RuntimeError("Fit failed.")

    params = best_payload["params"]
    fit_results = []
    for dataset in datasets:
        C, A, D_hat = objective.predict_dataset(params, dataset)
        residuals = dataset.D - D_hat if dataset.D is not None else None
        fit_results.append(
            {
                "C": C,
                "A": A,
                "D_hat": D_hat,
                "residuals": residuals,
            }
        )

    residuals_raw = best_payload.get("residuals")
    residuals = np.asarray(
        residuals_raw if residuals_raw is not None else [], dtype=float
    ).ravel()
    jac = best_payload.get("jac")
    jac_arr = np.asarray(jac, dtype=float) if jac is not None else None
    sigma2 = None
    if residuals.size and jac_arr is not None and jac_arr.size:
        dof = max(residuals.size - jac_arr.shape[1], 1)
        sigma2 = float(np.sum(residuals**2) / float(dof))

    stability_indicator = None
    stability_diag = None
    if jac_arr is not None and jac_arr.size:
        cov = best_payload.get("covariance")
        if cov is not None:
            jtj = jac_arr.T @ jac_arr
            if log_params:
                scale = np.ones(len(param_names), dtype=float)
                for idx, name in enumerate(param_names):
                    if name in log_params:
                        val = float(params.get(name, 0.0))
                        scale[idx] = np.log(10.0) * max(val, 1e-300)
                jac_lin = jac_arr / scale[None, :]
                jtj = jac_lin.T @ jac_lin
            report = build_identifiability_report(jtj, cov, param_names)
            stability_indicator = report.get("stability_indicator")
            stability_diag = report

    stats = {}
    rms_val = None
    if residuals.size:
        rms_val = float(np.sqrt(np.mean(residuals**2)))
    if rms_val is not None:
        stats["RMS"] = rms_val

    results_lines = [
        "Fit complete.",
        f"SSQ: {best_payload['ssq']:.6g}",
        f"nfev: {best_payload['nfev']}",
    ]
    for name, value in params.items():
        results_lines.append(f"{name} = {value:.6g}")
    if stability_diag:
        status = str(stability_diag.get("status") or "")
        summary = str(stability_diag.get("diag_summary") or "")
        if status == "critical":
            results_lines.append(
                f">>> CRITICAL WARNING: Ill-conditioned system ({summary}). "
                "Parameters might not be identifiable."
            )
        elif status == "warn":
            results_lines.append(
                f">>> WARNING: Poor conditioning ({summary}). "
                "High correlations might be present."
            )
        if show_stability_diagnostics:
            full = str(stability_diag.get("diag_full") or "")
            if full:
                results_lines.append("")
                results_lines.append(full)

    def refit_fn(data_star, theta0, max_iter=30, tol=1e-8):
        data_list = data_star if isinstance(data_star, (list, tuple)) else [data_star]
        if len(data_list) != len(datasets):
            return theta0, False, {"error": "Bootstrap data length mismatch."}

        datasets_star = []
        for dataset, d_star in zip(datasets, data_list, strict=True):
            ds = copy.deepcopy(dataset)
            ds.D = np.asarray(d_star, dtype=float)
            ds.fit_C = None
            ds.fit_A = None
            ds.fit_D_hat = None
            ds.fit_residuals = None
            datasets_star.append(ds)

        obj_star = GlobalKineticsObjective(
            model,
            datasets_star,
            param_names=param_names,
            nnls=nnls,
            log_params=log_params,
        )

        theta0 = np.asarray(theta0, dtype=float).ravel()
        x0_opt = []
        for name, val in zip(param_names, theta0, strict=True):
            if name in log_params:
                x0_opt.append(np.log10(max(val, 1e-300)))
            else:
                x0_opt.append(val)
        x0_opt = np.asarray(x0_opt, dtype=float)

        best = None
        best_ssq = np.inf
        run_total = max(1, runs)
        seed_list = list(seeds_list) if seeds_list else [None] * run_total
        for run_idx in range(run_total):
            seed = seed_list[run_idx] if run_idx < len(seed_list) else None
            rng = np.random.default_rng(seed)
            x0 = x0_opt if run_idx == 0 else _randomize_params(x0_opt, bounds, rng)
            try:
                res = fit_global(
                    obj_star,
                    x0,
                    bounds=bounds,
                    method=ls_method,
                    max_nfev=max_iter,
                    ftol=tol,
                    xtol=tol,
                    gtol=tol,
                )
            except Exception as exc:
                if run_total == 1:
                    return theta0, False, {"error": str(exc)}
                continue
            if res.ssq < best_ssq:
                best_ssq = res.ssq
                best = res

        if best is None:
            return theta0, False, {"error": "Refit failed."}

        theta_star = np.array([best.params[name] for name in param_names], dtype=float)
        info = {
            "optimizer_used": ls_method,
            "optimizer_requested": ls_method,
            "fallback_used": bool(run_total > 1),
        }
        return theta_star, True, info

    return {
        "success": True,
        "availablePlots": [
            {"id": "kinetics_concentrations", "title": "C(t)"},
            {"id": "kinetics_d_fit", "title": "D vs D_hat"},
            {"id": "kinetics_residuals", "title": "Residuals"},
            {"id": "kinetics_a_profiles", "title": "A profiles"},
        ],
        "params": params,
        "ssq": best_payload["ssq"],
        "nfev": best_payload["nfev"],
        "njev": best_payload["njev"],
        "param_names": param_names,
        "log_params": list(log_params),
        "dynamic_species": list(model.dynamic_species),
        "jac": jac_arr,
        "residuals": residuals,
        "covariance": best_payload.get("covariance"),
        "correlations": best_payload.get("correlations"),
        "condition_number": best_payload.get("condition_number"),
        "sigma2": sigma2,
        "fit_results": fit_results,
        "datasets": datasets,
        "stability_indicator": stability_indicator,
        "statistics": stats,
        "results_text": "\n".join(results_lines),
        "refit_fn": refit_fn,
    }
