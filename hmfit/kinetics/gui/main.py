"""Main GUI for the kinetics module."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hmfit_gui_qt.widgets.mpl_canvas import MplCanvas, NavigationToolbar

from ..data.fit_dataset import KineticsFitDataset
from ..data.loaders import load_matrix_file, load_xlsx
from ..fit.objective import GlobalKineticsObjective
from ..fit.optimizer import fit_global
from ..mechanism_editor.parser import MechanismParseError, parse_mechanism
from ..model.kinetics_model import KineticsModel
from .dataset_editor import DatasetEditorDialog
from .import_wizard import ImportWizard


class KineticsMainWidget(QWidget):
    """Main widget for kinetics datasets, mechanism, and fitting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[KineticsFitDataset] = []
        self._model: KineticsModel | None = None
        self._mechanism_text: str = ""
        self._param_settings: dict[str, tuple[float, float, float, bool, bool]] = {}
        self._param_defaults: dict[str, tuple[float, float, float, bool, bool]] = {}
        self._last_fit_result = None
        self._settings = QSettings("HMFit", "Kinetics")

        self._build_ui()
        self._update_fit_state()

    # ---- UI ----
    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Left panel
        left_panel = QVBoxLayout()
        self._dataset_list = QListWidget(self)
        self._dataset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._dataset_list.currentRowChanged.connect(self._on_dataset_selected)
        self._dataset_list.itemSelectionChanged.connect(self._update_fit_state)
        self._dataset_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._dataset_list.customContextMenuRequested.connect(
            self._on_dataset_context_menu
        )
        left_panel.addWidget(QLabel("Datasets", self))
        left_panel.addWidget(self._dataset_list, 1)

        btn_row = QHBoxLayout()
        self._btn_import = QPushButton("Import…", self)
        self._btn_import.clicked.connect(self._on_import)
        btn_row.addWidget(self._btn_import)
        self._btn_duplicate = QPushButton("Duplicate", self)
        self._btn_duplicate.clicked.connect(self._on_duplicate)
        btn_row.addWidget(self._btn_duplicate)
        left_panel.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self._btn_edit = QPushButton("Edit metadata", self)
        self._btn_edit.clicked.connect(self._on_edit_metadata)
        btn_row2.addWidget(self._btn_edit)
        self._btn_delete = QPushButton("Delete", self)
        self._btn_delete.clicked.connect(self._on_delete)
        btn_row2.addWidget(self._btn_delete)
        left_panel.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        self._btn_batch_edit = QPushButton("Batch edit metadata...", self)
        self._btn_batch_edit.clicked.connect(self._on_batch_edit_metadata)
        btn_row3.addWidget(self._btn_batch_edit)
        btn_row3.addStretch(1)
        left_panel.addLayout(btn_row3)

        self._btn_preprocess = QPushButton("Preprocess", self)
        self._btn_preprocess.clicked.connect(self._on_preprocess)
        left_panel.addWidget(self._btn_preprocess)

        session_row = QHBoxLayout()
        self._btn_load_session = QPushButton("Load session", self)
        self._btn_load_session.clicked.connect(self._on_load_session)
        session_row.addWidget(self._btn_load_session)
        self._btn_save_session = QPushButton("Save session", self)
        self._btn_save_session.clicked.connect(self._on_save_session)
        session_row.addWidget(self._btn_save_session)
        left_panel.addLayout(session_row)

        self._status_label = QLabel("", self)
        self._status_label.setStyleSheet("color: #B00020;")
        self._status_label.setWordWrap(True)
        left_panel.addWidget(self._status_label)

        left_container = QWidget(self)
        left_container.setLayout(left_panel)

        # Tabs
        self._tabs = QTabWidget(self)
        self._tab_data = QWidget(self._tabs)
        self._tab_mechanism = QWidget(self._tabs)
        self._tab_fit = QWidget(self._tabs)
        self._tabs.addTab(self._tab_data, "Data")
        self._tabs.addTab(self._tab_mechanism, "Mechanism")
        self._tabs.addTab(self._tab_fit, "Fit")

        outer.addWidget(left_container, 1)
        outer.addWidget(self._tabs, 3)

        self._build_data_tab()
        self._build_mechanism_tab()
        self._build_fit_tab()

    def _build_data_tab(self) -> None:
        layout = QVBoxLayout(self._tab_data)
        tools_row = QHBoxLayout()
        self._btn_svd_efa = QPushButton("SVD/EFA…", self._tab_data)
        self._btn_svd_efa.clicked.connect(self._on_svd_efa)
        tools_row.addWidget(self._btn_svd_efa)
        tools_row.addStretch(1)
        layout.addLayout(tools_row)
        self._data_preview = QTableWidget(self._tab_data)
        layout.addWidget(self._data_preview, 1)

        self._plot_canvas = MplCanvas(self._tab_data)
        self._plot_toolbar = NavigationToolbar(self._plot_canvas, self._tab_data)
        layout.addWidget(self._plot_toolbar)
        layout.addWidget(self._plot_canvas, 2)
        self._plot_canvas.clear()

    def _build_mechanism_tab(self) -> None:
        layout = QVBoxLayout(self._tab_mechanism)
        self._mechanism_edit = QTextEdit(self._tab_mechanism)
        layout.addWidget(self._mechanism_edit, 1)

        validate_row = QHBoxLayout()
        self._btn_validate = QPushButton("Validate", self._tab_mechanism)
        self._btn_validate.clicked.connect(self._on_validate_mechanism)
        validate_row.addWidget(self._btn_validate)
        validate_row.addStretch(1)
        layout.addLayout(validate_row)

        self._mechanism_output = QTextEdit(self._tab_mechanism)
        self._mechanism_output.setReadOnly(True)
        layout.addWidget(self._mechanism_output, 1)

    def _build_fit_tab(self) -> None:
        layout = QVBoxLayout(self._tab_fit)

        self._param_table = QTableWidget(self._tab_fit)
        self._param_table.setColumnCount(6)
        self._param_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Min", "Max", "Fixed", "Log10?"]
        )
        layout.addWidget(self._param_table, 2)

        options_row = QHBoxLayout()
        self._nnls_check = QCheckBox("NNLS (A >= 0)", self._tab_fit)
        options_row.addWidget(self._nnls_check)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        fit_row = QHBoxLayout()
        self._btn_fit = QPushButton("Fit", self._tab_fit)
        self._btn_fit.clicked.connect(self._on_fit)
        fit_row.addWidget(self._btn_fit)
        self._btn_fit_selected = QPushButton("Fit selected", self._tab_fit)
        self._btn_fit_selected.clicked.connect(self._on_fit_selected)
        fit_row.addWidget(self._btn_fit_selected)
        self._btn_reset_fit = QPushButton("Reset fit", self._tab_fit)
        self._btn_reset_fit.clicked.connect(self._reset_fit)
        fit_row.addWidget(self._btn_reset_fit)
        fit_row.addStretch(1)
        layout.addLayout(fit_row)

        self._fit_output = QTextEdit(self._tab_fit)
        self._fit_output.setReadOnly(True)
        layout.addWidget(self._fit_output, 1)

        self._corr_label = QLabel("Parameter correlations", self._tab_fit)
        self._corr_label.setVisible(False)
        layout.addWidget(self._corr_label)
        self._corr_table = QTableWidget(self._tab_fit)
        self._corr_table.setVisible(False)
        layout.addWidget(self._corr_table)

        self._fit_plots = QTabWidget(self._tab_fit)
        self._fit_c_canvas = self._create_plot_tab("C(t)")
        self._fit_d_canvas = self._create_plot_tab("D vs D_hat")
        self._fit_resid_canvas = self._create_plot_tab("Residuals")
        self._fit_a_canvas = self._create_plot_tab("A profiles")
        layout.addWidget(self._fit_plots, 3)
        self._clear_fit_plots()

    def _create_plot_tab(self, label: str) -> MplCanvas:
        widget = QWidget(self._fit_plots)
        layout = QVBoxLayout(widget)
        canvas = MplCanvas(widget)
        toolbar = NavigationToolbar(canvas, widget)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        self._fit_plots.addTab(widget, label)
        return canvas

    def _clear_fit_plots(self) -> None:
        message = "Run a fit to see results."
        self._fit_c_canvas.show_message(message)
        self._fit_d_canvas.show_message(message)
        self._fit_resid_canvas.show_message(message)
        self._fit_a_canvas.show_message(message)

    def _reset_canvas(self, canvas: MplCanvas, *, grid: bool = True) -> None:
        canvas.figure.clear()
        canvas.ax = canvas.figure.add_subplot(111)
        if grid:
            canvas.ax.grid(True, alpha=0.3)

    def _clear_fit_results(self, dataset: KineticsFitDataset) -> None:
        dataset.fit_C = None
        dataset.fit_A = None
        dataset.fit_D_hat = None
        dataset.fit_residuals = None

    def _reset_fit(self) -> None:
        for dataset in self._datasets:
            self._clear_fit_results(dataset)
        self._fit_output.clear()
        self._last_fit_result = None
        self._update_fit_plots(None)
        self._reset_param_table()
        self._clear_correlation_table()
        self._update_fit_state()

    def _reset_param_table(self) -> None:
        if not self._param_defaults:
            return
        self._param_settings = dict(self._param_defaults)
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name or name not in self._param_defaults:
                continue
            value, min_val, max_val, fixed, log = self._param_defaults[name]
            self._param_table.setItem(row, 1, QTableWidgetItem(f"{value:.6g}"))
            self._param_table.setItem(row, 2, QTableWidgetItem(f"{min_val:.6g}"))
            self._param_table.setItem(row, 3, QTableWidgetItem(f"{max_val:.6g}"))
            fixed_item = QTableWidgetItem()
            fixed_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            fixed_item.setCheckState(
                Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked
            )
            self._param_table.setItem(row, 4, fixed_item)
            log_item = QTableWidgetItem()
            log_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            log_item.setCheckState(
                Qt.CheckState.Checked if log else Qt.CheckState.Unchecked
            )
            self._param_table.setItem(row, 5, log_item)

    def _clear_correlation_table(self) -> None:
        self._corr_table.clear()
        self._corr_table.setRowCount(0)
        self._corr_table.setColumnCount(0)
        self._corr_table.setVisible(False)
        self._corr_label.setVisible(False)

    def _update_fit_plots(self, dataset: KineticsFitDataset | None) -> None:
        if dataset is None or dataset.fit_C is None or self._model is None:
            self._clear_fit_plots()
            return
        self._plot_concentrations(dataset)
        self._plot_data_fit(dataset)
        self._plot_residuals(dataset)
        self._plot_profiles(dataset)

    def _plot_concentrations(self, dataset: KineticsFitDataset) -> None:
        if dataset.fit_C is None or self._model is None:
            self._fit_c_canvas.show_message("No fitted concentrations.")
            return
        t = dataset.t
        C = dataset.fit_C
        self._reset_canvas(self._fit_c_canvas)
        ax = self._fit_c_canvas.ax
        for idx, species in enumerate(self._model.dynamic_species):
            if idx >= C.shape[1]:
                break
            ax.plot(t, C[:, idx], label=species)
        ax.set_xlabel(f"time [{dataset.time_unit}]")
        ax.set_ylabel("concentration")
        ax.set_title("Concentration profiles")
        if C.shape[1] <= 8:
            ax.legend(fontsize=8)
        self._fit_c_canvas.figure.tight_layout()
        self._fit_c_canvas.draw_idle()

    def _plot_data_fit(self, dataset: KineticsFitDataset) -> None:
        if dataset.D is None or dataset.fit_D_hat is None:
            self._fit_d_canvas.show_message("No fitted data matrix.")
            return
        t = dataset.t
        D = dataset.D
        D_hat = dataset.fit_D_hat
        labels = dataset.channel_labels
        self._reset_canvas(self._fit_d_canvas)
        ax = self._fit_d_canvas.ax
        indices = _select_channels(D.shape[1], 8)
        for idx in indices:
            label = labels[idx] if idx < len(labels) else f"ch{idx}"
            line = ax.plot(t, D[:, idx], label=f"{label} data")
            ax.plot(
                t,
                D_hat[:, idx],
                linestyle="--",
                color=line[0].get_color(),
                label=f"{label} fit",
            )
        ax.set_xlabel(f"time [{dataset.time_unit}]")
        ax.set_ylabel(f"signal [{dataset.signal_unit}]")
        ax.set_title("Observed vs fitted traces")
        if len(indices) <= 4:
            ax.legend(fontsize=8)
        self._fit_d_canvas.figure.tight_layout()
        self._fit_d_canvas.draw_idle()

    def _plot_residuals(self, dataset: KineticsFitDataset) -> None:
        if dataset.D is None or dataset.fit_D_hat is None:
            self._fit_resid_canvas.show_message("No residuals available.")
            return
        resid = dataset.fit_residuals
        if resid is None:
            resid = dataset.D - dataset.fit_D_hat
            dataset.fit_residuals = resid

        use_heatmap = (
            dataset.technique in {"spec_full", "nmr_full"}
            and dataset.x is not None
            and resid.ndim == 2
        )
        if use_heatmap:
            self._reset_canvas(self._fit_resid_canvas, grid=False)
            ax = self._fit_resid_canvas.ax
            t = dataset.t
            x = dataset.x
            extent = [float(x.min()), float(x.max()), float(t.min()), float(t.max())]
            im = ax.imshow(
                resid,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="coolwarm",
            )
            ax.set_xlabel(f"x [{dataset.x_unit}]")
            ax.set_ylabel(f"time [{dataset.time_unit}]")
            ax.set_title("Residual heatmap")
            self._fit_resid_canvas.figure.colorbar(im, ax=ax, shrink=0.8)
        else:
            t = dataset.t
            labels = dataset.channel_labels
            self._reset_canvas(self._fit_resid_canvas)
            ax = self._fit_resid_canvas.ax
            indices = _select_channels(resid.shape[1], 8)
            for idx in indices:
                label = labels[idx] if idx < len(labels) else f"ch{idx}"
                ax.plot(t, resid[:, idx], label=label)
            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
            ax.set_xlabel(f"time [{dataset.time_unit}]")
            ax.set_ylabel("residual")
            ax.set_title("Residual traces")
            if len(indices) <= 8:
                ax.legend(fontsize=8)
        self._fit_resid_canvas.figure.tight_layout()
        self._fit_resid_canvas.draw_idle()

    def _plot_profiles(self, dataset: KineticsFitDataset) -> None:
        if dataset.fit_A is None:
            self._fit_a_canvas.show_message("No fitted profiles available.")
            return
        if dataset.x is None:
            self._fit_a_canvas.show_message("No spectral axis available for A profiles.")
            return
        if self._model is None:
            self._fit_a_canvas.show_message("No model loaded.")
            return
        A = dataset.fit_A
        x = dataset.x
        self._reset_canvas(self._fit_a_canvas)
        ax = self._fit_a_canvas.ax
        for idx, species in enumerate(self._model.dynamic_species):
            if idx >= A.shape[0]:
                break
            ax.plot(x, A[idx, :], label=species)
        ax.set_xlabel(f"x [{dataset.x_unit}]")
        ax.set_ylabel("A")
        ax.set_title("Pure profiles")
        if A.shape[0] <= 8:
            ax.legend(fontsize=8)
        self._fit_a_canvas.figure.tight_layout()
        self._fit_a_canvas.draw_idle()

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

    def _on_edit_metadata(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            return
        dialog = DatasetEditorDialog(
            dataset,
            dynamic_species=self._dynamic_species(),
            fixed_species=self._fixed_species(),
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._clear_fit_results(dataset)
            self._refresh_dataset_list()
            self._update_fit_state()

    def _on_delete(self) -> None:
        row = self._dataset_list.currentRow()
        if row < 0:
            return
        del self._datasets[row]
        self._refresh_dataset_list()
        self._update_fit_state()

    def _on_batch_edit_metadata(self) -> None:
        datasets = self._selected_datasets()
        if not datasets:
            QMessageBox.warning(self, "Batch edit", "Select one or more datasets.")
            return
        template = copy.deepcopy(datasets[0])
        dialog = DatasetEditorDialog(
            template,
            dynamic_species=self._dynamic_species(),
            fixed_species=self._fixed_species(),
            parent=self,
        )
        dialog.setWindowTitle("Batch Edit Metadata")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = dialog.dataset
        for dataset in datasets:
            dataset.temperature = updated.temperature
            dataset.time_unit = updated.time_unit
            dataset.x_unit = updated.x_unit
            dataset.signal_unit = updated.signal_unit
            dataset.y0 = copy.deepcopy(updated.y0)
            dataset.fixed_conc = copy.deepcopy(updated.fixed_conc)
            self._clear_fit_results(dataset)
        self._refresh_dataset_list()
        self._update_fit_state()

    def _save_metadata_template(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "Template", "Select a dataset first.")
            return
        template = {
            "y0": dict(dataset.y0) if dataset.y0 is not None else None,
            "fixed_conc": dict(dataset.fixed_conc) if dataset.fixed_conc is not None else None,
            "temperature": float(dataset.temperature),
            "time_unit": dataset.time_unit,
            "x_unit": dataset.x_unit,
            "signal_unit": dataset.signal_unit,
        }
        self._settings.setValue("metadata_template", json.dumps(template))

    def _apply_metadata_template(self) -> None:
        template = self._load_metadata_template()
        if template is None:
            QMessageBox.warning(self, "Template", "No metadata template saved.")
            return
        datasets = self._selected_datasets()
        if not datasets:
            QMessageBox.warning(self, "Template", "Select one or more datasets.")
            return
        for dataset in datasets:
            dataset.y0 = copy.deepcopy(template.get("y0"))
            dataset.fixed_conc = copy.deepcopy(template.get("fixed_conc") or {})
            dataset.temperature = float(template.get("temperature", dataset.temperature))
            dataset.time_unit = str(template.get("time_unit", dataset.time_unit))
            dataset.x_unit = str(template.get("x_unit", dataset.x_unit))
            dataset.signal_unit = str(template.get("signal_unit", dataset.signal_unit))
            self._clear_fit_results(dataset)
        self._refresh_dataset_list()
        self._update_fit_state()

    def _load_metadata_template(self) -> dict[str, object] | None:
        raw = self._settings.value("metadata_template")
        if not raw:
            return None
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            return None

    def _on_delete_selected(self) -> None:
        indices = self._selected_indices()
        if not indices:
            return
        for idx in reversed(indices):
            del self._datasets[idx]
        self._refresh_dataset_list()
        self._update_fit_state()

    def _on_duplicate_all(self) -> None:
        if not self._datasets:
            return
        clones = []
        for dataset in self._datasets:
            cloned = copy.deepcopy(dataset)
            cloned.name = f"{dataset.name} (copy)"
            clones.append(cloned)
        self._datasets.extend(clones)
        self._refresh_dataset_list()
        self._update_fit_state()

    def _on_dataset_context_menu(self, pos) -> None:
        menu = QMenu(self)
        action_edit = QAction("Edit metadata for selected", self)
        action_edit.triggered.connect(self._on_batch_edit_metadata)
        menu.addAction(action_edit)
        action_save_template = QAction("Save metadata template", self)
        action_save_template.triggered.connect(self._save_metadata_template)
        menu.addAction(action_save_template)
        action_apply_template = QAction("Apply metadata template", self)
        action_apply_template.triggered.connect(self._apply_metadata_template)
        menu.addAction(action_apply_template)
        action_duplicate = QAction("Duplicate all", self)
        action_duplicate.triggered.connect(self._on_duplicate_all)
        menu.addAction(action_duplicate)
        action_delete = QAction("Delete selected", self)
        action_delete.triggered.connect(self._on_delete_selected)
        menu.addAction(action_delete)
        action_fit = QAction("Fit selected", self)
        action_fit.triggered.connect(self._on_fit_selected)
        menu.addAction(action_fit)
        menu.exec(self._dataset_list.mapToGlobal(pos))

    def _on_svd_efa(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "SVD/EFA", "Select a dataset first.")
            return
        if dataset.D is None:
            QMessageBox.warning(self, "SVD/EFA", "Dataset has no data matrix.")
            return
        dialog = SVDEFADialog(dataset, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.applied_data is None:
            return
        dataset.D = dialog.applied_data
        self._clear_fit_results(dataset)
        self._update_data_view(dataset)
        self._update_fit_plots(dataset)

    def _on_preprocess(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            return
        if dataset.D is None:
            QMessageBox.warning(self, "Preprocess", "Dataset has no data matrix.")
            return
        dialog = ChannelSelectorDialog(dataset, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dialog.selected_indices
        if not indices:
            return
        dataset.D = dataset.D[:, indices]
        dataset.channel_labels = [dataset.channel_labels[i] for i in indices]
        if dataset.x is not None:
            dataset.x = dataset.x[indices]
        if dataset.sigma is not None and dataset.sigma.ndim == 2:
            dataset.sigma = dataset.sigma[:, indices]
        if dataset.weights is not None and dataset.weights.ndim == 2:
            dataset.weights = dataset.weights[:, indices]
        if dataset.channel_indices:
            dataset.channel_indices = [dataset.channel_indices[i] for i in indices]
        else:
            dataset.channel_indices = list(indices)
        self._clear_fit_results(dataset)
        self._on_dataset_selected(self._dataset_list.currentRow())

    # ---- Session ----
    def _on_save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            "",
            "HM Fit Session (*.hmfit.json *.hmfit);;JSON (*.json)",
        )
        if not path:
            return
        session = {
            "version": 1,
            "mechanism_text": self._mechanism_edit.toPlainText(),
            "datasets": [self._serialize_dataset(dataset) for dataset in self._datasets],
            "param_settings": {
                name: {
                    "value": value,
                    "min": min_val,
                    "max": max_val,
                    "fixed": fixed,
                    "log": log,
                }
                for name, (value, min_val, max_val, fixed, log) in self._param_settings.items()
            },
        }
        if self._last_fit_result is not None:
            if isinstance(self._last_fit_result, dict):
                session["fit_result"] = self._last_fit_result
            else:
                session["fit_result"] = {
                    "params": self._last_fit_result.params,
                    "ssq": self._last_fit_result.ssq,
                    "nfev": self._last_fit_result.nfev,
                    "njev": self._last_fit_result.njev,
                }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(session, handle, indent=2, ensure_ascii=True)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _on_load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "HM Fit Session (*.hmfit.json *.hmfit);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                session = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        if not isinstance(session, dict):
            QMessageBox.warning(self, "Load failed", "Session file format is invalid.")
            return

        self._datasets.clear()
        self._model = None
        self._mechanism_text = ""
        self._param_settings = {}
        self._param_defaults = {}
        self._last_fit_result = None

        text = str(session.get("mechanism_text", "") or "")
        self._mechanism_edit.setPlainText(text)
        if text.strip():
            try:
                ast = parse_mechanism(text)
                self._model = KineticsModel(ast)
                self._mechanism_text = text
                params = _extract_param_names(ast)
                self._param_settings = self._load_param_settings(
                    session.get("param_settings", {})
                )
                self._update_param_table(params)
                summary = [
                    f"Species: {', '.join(ast.species)}",
                    f"Fixed: {', '.join(sorted(ast.fixed)) if ast.fixed else 'none'}",
                    f"Reactions: {len(ast.reactions)}",
                    f"Parameters: {', '.join(params)}",
                ]
                self._mechanism_output.setText("\n".join(summary))
            except MechanismParseError as exc:
                self._mechanism_output.setText(f"Parse error: {exc}")
                self._model = None
        else:
            self._mechanism_output.clear()

        errors: list[str] = []
        for entry in session.get("datasets", []):
            try:
                dataset = self._deserialize_dataset(entry)
            except Exception as exc:
                errors.append(str(exc))
                continue
            self._datasets.append(dataset)

        if session.get("fit_result") is not None:
            self._last_fit_result = session.get("fit_result")

        self._refresh_dataset_list()
        if self._datasets:
            self._dataset_list.setCurrentRow(0)
        else:
            self._clear_fit_plots()
            self._plot_canvas.show_message("No dataset selected")

        self._update_fit_state()
        self._clear_correlation_table()
        self._fit_output.clear()
        if (
            isinstance(self._last_fit_result, dict)
            and self._model is not None
            and self._datasets
        ):
            params = self._last_fit_result.get("params")
            if isinstance(params, dict):
                objective = GlobalKineticsObjective(
                    self._model,
                    self._datasets,
                    param_names=list(params.keys()),
                    nnls=self._nnls_check.isChecked(),
                    log_params={name for name, settings in self._param_settings.items() if settings[4]},
                )
                try:
                    for dataset in self._datasets:
                        C, A, D_hat = objective.predict_dataset(params, dataset)
                        dataset.fit_C = C
                        dataset.fit_A = A
                        dataset.fit_D_hat = D_hat
                        dataset.fit_residuals = (
                            dataset.D - D_hat if dataset.D is not None else None
                        )
                    lines = [
                        "Fit loaded.",
                        f"SSQ: {float(self._last_fit_result.get('ssq', 0.0)):.6g}",
                        f"nfev: {self._last_fit_result.get('nfev', '')}",
                    ]
                    for name, value in params.items():
                        lines.append(f"{name} = {float(value):.6g}")
                    self._fit_output.setText("\n".join(lines))
                    self._update_fit_plots(self._selected_dataset())
                except Exception:
                    self._fit_output.clear()
        if errors:
            QMessageBox.warning(
                self,
                "Load warnings",
                "Some datasets could not be loaded:\n" + "\n".join(errors),
            )

    def _serialize_dataset(self, dataset: KineticsFitDataset) -> dict[str, object]:
        entry: dict[str, object] = {
            "name": dataset.name,
            "technique": dataset.technique,
            "source_path": dataset.source_path,
            "time_unit": dataset.time_unit,
            "x_unit": dataset.x_unit,
            "signal_unit": dataset.signal_unit,
            "temperature": float(dataset.temperature),
            "y0": _serialize_mapping(dataset.y0),
            "fixed_conc": _serialize_mapping(dataset.fixed_conc),
            "loader": {
                "kind": dataset.loader_kind,
                "delimiter": dataset.loader_delimiter,
                "header": dataset.loader_header,
                "transpose": dataset.loader_transpose,
                "time_col": dataset.loader_time_col,
                "sheet": dataset.loader_sheet,
            },
            "channel_indices": (
                list(dataset.channel_indices)
                if dataset.channel_indices
                else list(range(len(dataset.channel_labels)))
            ),
        }
        if dataset.source_path is None and dataset.D is not None:
            entry["data"] = {
                "t": dataset.t.tolist(),
                "D": dataset.D.tolist(),
                "x": dataset.x.tolist() if dataset.x is not None else None,
                "labels": list(dataset.channel_labels),
            }
        return entry

    def _deserialize_dataset(self, entry: object) -> KineticsFitDataset:
        if not isinstance(entry, dict):
            raise ValueError("Dataset entry is invalid.")
        source_path = entry.get("source_path")
        loader = entry.get("loader", {}) if isinstance(entry.get("loader"), dict) else {}
        channel_indices = entry.get("channel_indices") or []
        data_block = entry.get("data") if isinstance(entry.get("data"), dict) else None

        if source_path:
            path = Path(str(source_path))
            if not path.exists():
                raise FileNotFoundError(f"Missing data file: {path}")
            kind = loader.get("kind") or ("xlsx" if path.suffix.lower() in {".xlsx", ".xls"} else "matrix")
            header = bool(loader.get("header", True))
            transpose = bool(loader.get("transpose", False))
            time_col = loader.get("time_col", 0)
            delimiter = loader.get("delimiter", None)
            sheet = loader.get("sheet", "data")

            if kind == "xlsx":
                t, D, x, labels = load_xlsx(
                    path,
                    sheet=str(sheet),
                    transpose=transpose,
                    time_col=time_col,
                    header=header,
                )
            else:
                t, D, x, labels = load_matrix_file(
                    path,
                    delimiter=delimiter,
                    transpose=transpose,
                    time_col=time_col,
                    header=header,
                )
        elif data_block is not None:
            t = np.asarray(data_block.get("t", []), dtype=float)
            D = np.asarray(data_block.get("D", []), dtype=float)
            x_vals = data_block.get("x")
            x = np.asarray(x_vals, dtype=float) if x_vals is not None else None
            labels = [str(label) for label in data_block.get("labels", [])]
        else:
            raise ValueError("Dataset entry is missing data and source_path.")

        t = np.asarray(t, dtype=float).reshape(-1)
        D = np.asarray(D, dtype=float)
        if D.ndim == 1 and D.size:
            D = D.reshape(-1, 1)

        if channel_indices:
            indices = [int(idx) for idx in channel_indices]
        else:
            indices = list(range(len(labels)))

        if labels and any(idx >= len(labels) or idx < 0 for idx in indices):
            raise ValueError("Channel indices are out of range.")

        D_sel = D[:, indices] if D.size else D
        labels_sel = [labels[idx] for idx in indices] if labels else []
        x_sel = x[indices] if x is not None else None

        dataset = KineticsFitDataset(
            t=np.asarray(t, dtype=float),
            y0=_deserialize_mapping(entry.get("y0")),
            fixed_conc=_deserialize_mapping(entry.get("fixed_conc")) or {},
            temperature=float(entry.get("temperature", 298.15)),
            D=D_sel,
            x=x_sel,
            channel_labels=labels_sel,
            technique=str(entry.get("technique", "spec_channels")),
            time_unit=str(entry.get("time_unit", "s")),
            x_unit=str(entry.get("x_unit", "")),
            signal_unit=str(entry.get("signal_unit", "")),
            name=str(entry.get("name", "Dataset")),
            source_path=str(source_path) if source_path else None,
            loader_kind=str(loader.get("kind")) if loader.get("kind") else None,
            loader_delimiter=loader.get("delimiter"),
            loader_header=bool(loader.get("header", True)),
            loader_transpose=bool(loader.get("transpose", False)),
            loader_time_col=loader.get("time_col", 0),
            loader_sheet=loader.get("sheet"),
            channel_indices=indices,
        )
        return dataset

    def _load_param_settings(
        self, settings: object
    ) -> dict[str, tuple[float, float, float, bool, bool]]:
        if not isinstance(settings, dict):
            return {}
        parsed: dict[str, tuple[float, float, float, bool, bool]] = {}
        for name, values in settings.items():
            if not isinstance(values, dict):
                continue
            try:
                parsed[str(name)] = (
                    float(values.get("value", 1.0)),
                    float(values.get("min", -np.inf)),
                    float(values.get("max", np.inf)),
                    bool(values.get("fixed", False)),
                    bool(values.get("log", False)),
                )
            except (TypeError, ValueError):
                continue
        return parsed

    def _refresh_dataset_list(self) -> None:
        self._dataset_list.clear()
        incomplete_count = 0
        for dataset in self._datasets:
            label = f"{dataset.name} ({dataset.technique})"
            item = QListWidgetItem(label)
            if not self._dataset_is_complete(dataset):
                item.setForeground(QColor("#B00020"))
                item.setToolTip("Incomplete metadata - click Edit metadata.")
                incomplete_count += 1
            self._dataset_list.addItem(item)
        if incomplete_count:
            self._status_label.setText(
                "Incomplete metadata: edit y0/fixed concentrations before fitting."
            )
        else:
            self._status_label.clear()

    def _selected_dataset(self) -> KineticsFitDataset | None:
        row = self._dataset_list.currentRow()
        if row < 0 or row >= len(self._datasets):
            return None
        return self._datasets[row]

    def _selected_indices(self) -> list[int]:
        indices = {self._dataset_list.row(item) for item in self._dataset_list.selectedItems()}
        if not indices and self._dataset_list.currentRow() >= 0:
            indices = {self._dataset_list.currentRow()}
        return sorted(idx for idx in indices if 0 <= idx < len(self._datasets))

    def _selected_datasets(self) -> list[KineticsFitDataset]:
        return [self._datasets[idx] for idx in self._selected_indices()]

    def _on_dataset_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._datasets):
            self._data_preview.setRowCount(0)
            self._data_preview.setColumnCount(0)
            self._plot_canvas.show_message("No dataset selected")
            self._clear_fit_plots()
            self._update_fit_state()
            return
        dataset = self._datasets[row]
        self._update_data_view(dataset)
        self._update_fit_plots(dataset)
        self._update_fit_state()

    def _update_data_view(self, dataset: KineticsFitDataset) -> None:
        if dataset.D is None:
            self._data_preview.setRowCount(0)
            self._data_preview.setColumnCount(0)
            self._plot_canvas.show_message("Dataset has no data")
            return
        t = dataset.t
        D = dataset.D
        labels = dataset.channel_labels

        max_rows = min(50, D.shape[0])
        max_cols = min(10, D.shape[1])
        self._data_preview.setRowCount(max_rows)
        self._data_preview.setColumnCount(max_cols + 1)
        self._data_preview.setHorizontalHeaderLabels(["t"] + labels[:max_cols])
        for row in range(max_rows):
            self._data_preview.setItem(row, 0, QTableWidgetItem(f"{t[row]:.6g}"))
            for col in range(max_cols):
                self._data_preview.setItem(
                    row, col + 1, QTableWidgetItem(f"{D[row, col]:.6g}")
                )

        self._plot_canvas.ax.clear()
        self._plot_canvas.ax.grid(True, alpha=0.3)
        channel_count = D.shape[1]
        if channel_count > 0:
            indices = _select_channels(channel_count, 8)
            for idx in indices:
                label = labels[idx] if idx < len(labels) else f"ch{idx}"
                self._plot_canvas.ax.plot(t, D[:, idx], label=str(label))
            if len(indices) <= 8:
                self._plot_canvas.ax.legend(fontsize=8)
        self._plot_canvas.figure.tight_layout()
        self._plot_canvas.draw_idle()

    # ---- Mechanism ----
    def _on_validate_mechanism(self) -> None:
        text = self._mechanism_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Mechanism", "Mechanism text is empty.")
            return
        try:
            ast = parse_mechanism(text)
        except MechanismParseError as exc:
            self._mechanism_output.setText(f"Parse error: {exc}")
            self._model = None
            self._update_fit_state()
            return

        self._model = KineticsModel(ast)
        self._mechanism_text = text
        for dataset in self._datasets:
            self._clear_fit_results(dataset)
        params = _extract_param_names(ast)
        self._update_param_table(params)

        summary = [
            f"Species: {', '.join(ast.species)}",
            f"Fixed: {', '.join(sorted(ast.fixed)) if ast.fixed else 'none'}",
            f"Reactions: {len(ast.reactions)}",
            f"Parameters: {', '.join(params)}",
        ]
        self._mechanism_output.setText("\n".join(summary))
        self._refresh_dataset_list()
        self._update_fit_state()
        self._update_fit_plots(self._selected_dataset())

    def _update_param_table(self, params: Sequence[str]) -> None:
        self._param_table.setRowCount(len(params))
        for row, name in enumerate(params):
            value, min_val, max_val, fixed, log = self._param_settings.get(
                name, (1.0, 0.0, np.inf, False, False)
            )
            self._param_table.setItem(row, 0, QTableWidgetItem(name))
            self._param_table.setItem(row, 1, QTableWidgetItem(f"{value:.6g}"))
            self._param_table.setItem(row, 2, QTableWidgetItem(f"{min_val:.6g}"))
            self._param_table.setItem(row, 3, QTableWidgetItem(f"{max_val:.6g}"))
            fixed_item = QTableWidgetItem()
            fixed_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            fixed_item.setCheckState(
                Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked
            )
            self._param_table.setItem(row, 4, fixed_item)
            log_item = QTableWidgetItem()
            log_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            log_item.setCheckState(
                Qt.CheckState.Checked if log else Qt.CheckState.Unchecked
            )
            self._param_table.setItem(row, 5, log_item)
            self._param_settings[name] = (value, min_val, max_val, fixed, log)
        self._param_defaults = dict(self._param_settings)

    # ---- Fit ----
    def _on_fit(self) -> None:
        self._run_fit(self._datasets, title="Fit")

    def _on_fit_selected(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            QMessageBox.warning(self, "Fit selected", "Select one or more datasets.")
            return
        self._run_fit(selected, title="Fit selected")

    def _run_fit(self, datasets: Sequence[KineticsFitDataset], *, title: str) -> None:
        if self._model is None:
            QMessageBox.warning(self, title, "Validate a mechanism first.")
            return
        if not datasets:
            QMessageBox.warning(self, title, "No datasets loaded.")
            return
        if not self._datasets_complete(datasets):
            QMessageBox.warning(
                self, title, "Some datasets are incomplete (missing y0/fixed)."
            )
            return

        try:
            params0, bounds, param_names, log_params = self._read_params_from_table()
        except ValueError as exc:
            QMessageBox.warning(self, title, str(exc))
            return

        objective = GlobalKineticsObjective(
            self._model,
            datasets,
            param_names=param_names,
            nnls=self._nnls_check.isChecked(),
            log_params=log_params,
        )
        try:
            result = fit_global(
                objective,
                params0,
                bounds=bounds,
                max_nfev=300,
            )
            for dataset in datasets:
                C, A, D_hat = objective.predict_dataset(result.params, dataset)
                dataset.fit_C = C
                dataset.fit_A = A
                dataset.fit_D_hat = D_hat
                dataset.fit_residuals = (
                    dataset.D - D_hat if dataset.D is not None else None
                )
        except Exception as exc:
            QMessageBox.warning(self, "Fit failed", str(exc))
            return

        lines = [
            "Fit complete.",
            f"SSQ: {result.ssq:.6g}",
            f"nfev: {result.nfev}",
        ]
        errors = result.errors
        error_lookup = None
        if errors is not None and errors.shape[0] == len(param_names):
            error_lookup = {
                name: float(err) for name, err in zip(param_names, errors, strict=True)
            }
        for name, value in result.params.items():
            if error_lookup and name in error_lookup:
                lines.append(f"{name} = {value:.6g} ± {error_lookup[name]:.3g}")
            else:
                lines.append(f"{name} = {value:.6g}")

        strong_corr = False
        if result.correlations is not None and result.correlations.size:
            off_diag = result.correlations.copy()
            np.fill_diagonal(off_diag, 0.0)
            strong_corr = bool(np.any(np.abs(off_diag) >= 0.95))
        if strong_corr or (result.condition_number is not None and result.condition_number > 1e8):
            lines.append(
                "Warning: parameters are strongly correlated; consider fixing one."
            )

        self._fit_output.setText("\n".join(lines))
        self._last_fit_result = result
        self._update_correlation_table(param_names, result.correlations)
        self._update_fit_plots(self._selected_dataset())

    def _read_params_from_table(
        self,
    ) -> tuple[
        dict[str, float],
        tuple[np.ndarray, np.ndarray],
        list[str],
        set[str],
    ]:
        param_names: list[str] = []
        values: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        log_params: set[str] = set()

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            value_item = self._param_table.item(row, 1)
            min_item = self._param_table.item(row, 2)
            max_item = self._param_table.item(row, 3)
            fixed_item = self._param_table.item(row, 4)
            log_item = self._param_table.item(row, 5)

            if name_item is None or value_item is None:
                raise ValueError("Parameter table is incomplete.")
            name = name_item.text().strip()
            if not name:
                raise ValueError("Parameter name is empty.")
            value = float(value_item.text())
            min_val = float(min_item.text()) if min_item and min_item.text() else -np.inf
            max_val = float(max_item.text()) if max_item and max_item.text() else np.inf
            fixed = fixed_item and fixed_item.checkState() == Qt.CheckState.Checked
            log_param = log_item and log_item.checkState() == Qt.CheckState.Checked
            if fixed:
                min_val = value
                max_val = value
            if min_val > max_val:
                raise ValueError(f"Min > Max for parameter '{name}'.")

            opt_value = value
            opt_min = min_val
            opt_max = max_val
            if log_param:
                if value <= 0:
                    raise ValueError(f"Parameter '{name}' must be > 0 for log10.")
                if np.isfinite(min_val) and min_val <= 0:
                    raise ValueError(f"Min for '{name}' must be > 0 for log10.")
                if np.isfinite(max_val) and max_val <= 0:
                    raise ValueError(f"Max for '{name}' must be > 0 for log10.")
                opt_value = float(np.log10(value))
                opt_min = float(np.log10(min_val)) if np.isfinite(min_val) else -np.inf
                opt_max = float(np.log10(max_val)) if np.isfinite(max_val) else np.inf
                log_params.add(name)

            param_names.append(name)
            values.append(opt_value)
            lower.append(opt_min)
            upper.append(opt_max)
            self._param_settings[name] = (value, min_val, max_val, fixed, log_param)

        params0 = {name: value for name, value in zip(param_names, values, strict=True)}
        return (
            params0,
            (np.array(lower, dtype=float), np.array(upper, dtype=float)),
            param_names,
            log_params,
        )

    def _update_correlation_table(
        self,
        param_names: Sequence[str],
        correlations: np.ndarray | None,
    ) -> None:
        if correlations is None or correlations.size == 0:
            self._clear_correlation_table()
            return
        n_params = len(param_names)
        if correlations.shape != (n_params, n_params):
            self._clear_correlation_table()
            return
        self._corr_table.setRowCount(n_params)
        self._corr_table.setColumnCount(n_params)
        self._corr_table.setHorizontalHeaderLabels(list(param_names))
        self._corr_table.setVerticalHeaderLabels(list(param_names))
        for row in range(n_params):
            for col in range(n_params):
                item = QTableWidgetItem("")
                item.setFlags(
                    Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                )
                if col <= row:
                    item.setText(f"{correlations[row, col]:.3f}")
                self._corr_table.setItem(row, col, item)
        self._corr_label.setVisible(True)
        self._corr_table.setVisible(True)

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

    def _update_fit_state(self) -> None:
        fit_enabled = (
            self._model is not None
            and bool(self._datasets)
            and self._datasets_complete()
        )
        self._btn_fit.setEnabled(fit_enabled)
        selected = self._selected_datasets()
        fit_selected_enabled = (
            self._model is not None
            and bool(selected)
            and self._datasets_complete(selected)
        )
        self._btn_fit_selected.setEnabled(fit_selected_enabled)
        self._btn_reset_fit.setEnabled(bool(self._datasets) or self._last_fit_result is not None)

    def _dynamic_species(self) -> list[str]:
        if self._model is None:
            return []
        return list(self._model.dynamic_species)

    def _fixed_species(self) -> list[str]:
        if self._model is None:
            return []
        return list(self._model.fixed_species)


class SVDEFADialog(QDialog):
    """Dialog for SVD/EFA inspection and denoising."""

    def __init__(self, dataset: KineticsFitDataset, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SVD / EFA")
        self._dataset = dataset
        self._D = np.asarray(dataset.D, dtype=float)
        if self._D.ndim == 1:
            self._D = self._D.reshape(-1, 1)
        self._t = np.asarray(dataset.t, dtype=float).reshape(-1)
        self._U: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self._Vt: np.ndarray | None = None
        self._recon: np.ndarray | None = None
        self.applied_data: np.ndarray | None = None
        self._efa_cache: tuple[np.ndarray, np.ndarray] | None = None

        self._build_ui()
        self._compute_svd()
        self._update_plots()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget(self)

        self._svd_canvas = MplCanvas(self)
        svd_widget = QWidget(self)
        svd_layout = QVBoxLayout(svd_widget)
        svd_layout.addWidget(NavigationToolbar(self._svd_canvas, svd_widget))
        svd_layout.addWidget(self._svd_canvas, 1)
        self._tabs.addTab(svd_widget, "SVD")

        self._recon_canvas = MplCanvas(self)
        recon_widget = QWidget(self)
        recon_layout = QVBoxLayout(recon_widget)
        recon_layout.addWidget(NavigationToolbar(self._recon_canvas, recon_widget))
        recon_layout.addWidget(self._recon_canvas, 1)
        self._tabs.addTab(recon_widget, "Reconstruction")

        self._efa_canvas = MplCanvas(self)
        efa_widget = QWidget(self)
        efa_layout = QVBoxLayout(efa_widget)
        efa_layout.addWidget(NavigationToolbar(self._efa_canvas, efa_widget))
        efa_layout.addWidget(self._efa_canvas, 1)
        self._tabs.addTab(efa_widget, "EFA")

        layout.addWidget(self._tabs, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Components", self))
        self._components_spin = QSpinBox(self)
        max_components = max(1, min(self._D.shape[0], self._D.shape[1]))
        self._components_spin.setRange(1, max_components)
        self._components_spin.setValue(min(3, max_components))
        self._components_spin.valueChanged.connect(self._update_reconstruction)
        controls.addWidget(self._components_spin)
        controls.addStretch(1)
        self._btn_apply = QPushButton("Apply denoise", self)
        self._btn_apply.clicked.connect(self._apply_denoise)
        controls.addWidget(self._btn_apply)
        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.reject)
        controls.addWidget(btn_close)
        layout.addLayout(controls)

    def _compute_svd(self) -> None:
        self._U, self._S, self._Vt = np.linalg.svd(self._D, full_matrices=False)

    def _update_plots(self) -> None:
        self._update_svd_plot()
        self._update_reconstruction()
        self._update_efa_plot()

    def _update_svd_plot(self) -> None:
        if self._S is None:
            self._svd_canvas.show_message("SVD not available.")
            return
        self._svd_canvas.figure.clear()
        self._svd_canvas.ax = self._svd_canvas.figure.add_subplot(111)
        ax = self._svd_canvas.ax
        comps = np.arange(1, len(self._S) + 1)
        ax.semilogy(comps, self._S, marker="o")
        ax.set_xlabel("component")
        ax.set_ylabel("singular value")
        ax.set_title("Singular values")
        ax.grid(True, alpha=0.3)
        self._svd_canvas.figure.tight_layout()
        self._svd_canvas.draw_idle()

    def _update_reconstruction(self) -> None:
        if self._U is None or self._S is None or self._Vt is None:
            self._recon_canvas.show_message("SVD not available.")
            return
        n_components = int(self._components_spin.value())
        C = self._U[:, :n_components] * self._S[:n_components]
        self._recon = C @ self._Vt[:n_components, :]

        self._recon_canvas.figure.clear()
        self._recon_canvas.ax = self._recon_canvas.figure.add_subplot(111)
        ax = self._recon_canvas.ax
        indices = _select_channels(self._D.shape[1], 6)
        for idx in indices:
            line = ax.plot(self._t, self._D[:, idx], label=f"ch{idx} data")
            ax.plot(
                self._t,
                self._recon[:, idx],
                linestyle="--",
                color=line[0].get_color(),
                label=f"ch{idx} recon",
            )
        ax.set_xlabel(f"time [{self._dataset.time_unit}]")
        ax.set_ylabel(f"signal [{self._dataset.signal_unit}]")
        ax.set_title(f"Reconstruction (N={n_components})")
        if len(indices) <= 4:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._recon_canvas.figure.tight_layout()
        self._recon_canvas.draw_idle()

    def _update_efa_plot(self) -> None:
        if self._efa_cache is None:
            self._efa_cache = self._compute_efa(max_components=5, max_points=25)
        times, values = self._efa_cache
        if times.size == 0:
            self._efa_canvas.show_message("EFA not available.")
            return
        self._efa_canvas.figure.clear()
        self._efa_canvas.ax = self._efa_canvas.figure.add_subplot(111)
        ax = self._efa_canvas.ax
        for idx in range(values.shape[1]):
            ax.semilogy(times, values[:, idx], label=f"comp {idx + 1}")
        ax.set_xlabel("time index")
        ax.set_ylabel("singular value")
        ax.set_title("Evolving factor analysis (forward)")
        ax.grid(True, alpha=0.3)
        if values.shape[1] <= 5:
            ax.legend(fontsize=8)
        self._efa_canvas.figure.tight_layout()
        self._efa_canvas.draw_idle()

    def _compute_efa(self, *, max_components: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        n_time = self._D.shape[0]
        if n_time < 2:
            return np.array([], dtype=int), np.zeros((0, max_components), dtype=float)
        points = min(max_points, n_time - 1)
        indices = np.linspace(2, n_time, points, dtype=int)
        values = np.zeros((len(indices), max_components), dtype=float)
        for i, idx in enumerate(indices):
            s = np.linalg.svd(self._D[:idx, :], full_matrices=False, compute_uv=False)
            values[i, : min(max_components, len(s))] = s[:max_components]
        return indices, values

    def _apply_denoise(self) -> None:
        if self._recon is None:
            QMessageBox.warning(self, "SVD/EFA", "Reconstruction not available.")
            return
        self.applied_data = self._recon
        self.accept()


class ChannelSelectorDialog(QDialog):
    """Dialog to choose a subset of channels."""

    def __init__(self, dataset: KineticsFitDataset, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Channels")
        self.selected_indices: list[int] = []
        self._labels = dataset.channel_labels
        self._x = dataset.x

        layout = QVBoxLayout(self)
        self._list = QListWidget(self)
        for label in self._labels:
            item = QListWidgetItem(str(label))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._list.addItem(item)
        layout.addWidget(self._list, 1)

        range_box = QGroupBox("Range", self)
        range_layout = QFormLayout(range_box)
        self._min_edit = QLineEdit(range_box)
        self._max_edit = QLineEdit(range_box)
        range_layout.addRow("Min", self._min_edit)
        range_layout.addRow("Max", self._max_edit)
        layout.addWidget(range_box)

        buttons = QHBoxLayout()
        btn_all = QPushButton("All", self)
        btn_all.clicked.connect(lambda: self._set_all(True))
        buttons.addWidget(btn_all)
        btn_none = QPushButton("None", self)
        btn_none.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(btn_none)
        btn_apply = QPushButton("Apply range", self)
        btn_apply.clicked.connect(self._apply_range)
        buttons.addWidget(btn_apply)
        buttons.addStretch(1)
        btn_ok = QPushButton("OK", self)
        btn_ok.clicked.connect(self._on_accept)
        buttons.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_cancel)
        layout.addLayout(buttons)

    def _set_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for idx in range(self._list.count()):
            self._list.item(idx).setCheckState(state)

    def _apply_range(self) -> None:
        if self._x is None:
            return
        try:
            min_val = float(self._min_edit.text())
            max_val = float(self._max_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Range", "Enter numeric min/max values.")
            return
        for idx, x_val in enumerate(self._x):
            state = (
                Qt.CheckState.Checked
                if min_val <= x_val <= max_val
                else Qt.CheckState.Unchecked
            )
            self._list.item(idx).setCheckState(state)

    def _on_accept(self) -> None:
        self.selected_indices = [
            idx
            for idx in range(self._list.count())
            if self._list.item(idx).checkState() == Qt.CheckState.Checked
        ]
        self.accept()


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


def _serialize_mapping(values: object) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    output: dict[str, float] = {}
    for key, value in values.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _deserialize_mapping(values: object) -> dict[str, float] | None:
    if not isinstance(values, dict):
        return None
    output: dict[str, float] = {}
    for key, value in values.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _select_channels(channel_count: int, max_lines: int) -> list[int]:
    if channel_count <= max_lines:
        return list(range(channel_count))
    return list(np.linspace(0, channel_count - 1, max_lines, dtype=int))
