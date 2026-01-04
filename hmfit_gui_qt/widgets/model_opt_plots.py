from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
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


def _first_match(items: Iterable[str], patterns: list[str]) -> str:
    import re

    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    for item in items:
        for rx in compiled:
            if rx.search(str(item)):
                return str(item)
    return ""


def resolve_receptor_guest(
    selected_columns: list[str],
    *,
    receptor_preference: str = "",
    guest_preference: str = "",
) -> tuple[str, str]:
    """
    Port of Tauri's Auto mapping (hmfit_tauri/src/main.js -> resolveMapping).

    - If receptor/guest are explicitly selected and present in selected_columns, keep them.
    - Otherwise, guess by regex patterns, then fall back to first/second selected column.
    - Ensure receptor != guest when possible.
    """
    selected = [str(c) for c in (selected_columns or []) if str(c).strip()]

    receptor = str(receptor_preference or "").strip()
    guest = str(guest_preference or "").strip()

    if receptor and receptor not in selected:
        receptor = ""
    if guest and guest not in selected:
        guest = ""

    receptor_patterns = [
        r"^h$",
        r"^host$",
        r"^host_",
        r"_host$",
        r"^receptor$",
        r"^receptor_",
        r"_receptor$",
        r"^ligand$",
        r"^ligand_",
        r"_ligand$",
        r"^p$",
        r"^prot$",
        r"^protein$",
        r"^x1$",
    ]
    guest_patterns = [
        r"^g$",
        r"^guest$",
        r"^guest_",
        r"_guest$",
        r"^titrant$",
        r"^titrant_",
        r"_titrant$",
        r"^metal$",
        r"^metal_",
        r"_metal$",
        r"^q$",
        r"^x2$",
        r"^x\\d+$",
    ]

    if not receptor:
        receptor = _first_match(selected, receptor_patterns)
    if not guest:
        guest = _first_match(selected, guest_patterns)

    if not receptor and len(selected) >= 1:
        receptor = selected[0]
    if not guest and len(selected) >= 2:
        guest = selected[1]

    if receptor and guest and receptor == guest:
        alt = next((c for c in selected if c != receptor), "")
        guest = alt or guest

    return receptor, guest


@dataclass(frozen=True)
class ModelOptPlotsState:
    n_components: int
    n_species: int
    modelo: list[list[float]]
    non_abs_species: list[int]
    algorithm: str
    model_settings: str
    optimizer: str
    initial_k: list[float]
    bounds: list[tuple[float | None, float | None]]
    fixed_mask: list[bool]
    receptor_label: str
    guest_label: str


class ModelOptPlotsWidget(QWidget):
    """
    Shared (Spectroscopy + NMR) controls matching the Tauri workflow:
    - Sub-tabs: Model / Optimization / Plots
    - Model grid: stoichiometry matrix + non-absorbent species (row selection)
    - Optimization: algorithm/model_settings/optimizer + K table (Value/Min/Max/Fixed)
    - Plots: receptor/guest mapping + plot controls (currently UI-only)
    """

    model_defined = Signal(int, int)  # n_components, n_species

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._in_table_update = False
        self._available_conc_columns: list[str] = []
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

        self.btn_define_model = QPushButton("Define Model Dimensions", model_tab)
        self.btn_define_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_define_model.clicked.connect(self._on_define_model_clicked)
        model_layout.addWidget(self.btn_define_model)

        dims_grid = QGridLayout()
        self.spin_n_components = QSpinBox(model_tab)
        self.spin_n_components.setRange(0, 999)
        self.spin_n_components.setValue(0)
        self.spin_n_species = QSpinBox(model_tab)
        self.spin_n_species.setRange(0, 999)
        self.spin_n_species.setValue(0)
        dims_grid.addWidget(QLabel("Number of Components"), 0, 0)
        dims_grid.addWidget(self.spin_n_components, 0, 1)
        dims_grid.addWidget(QLabel("Number of Species"), 1, 0)
        dims_grid.addWidget(self.spin_n_species, 1, 1)
        model_layout.addLayout(dims_grid)

        model_layout.addWidget(QLabel("Select non-absorbent species"), 0)

        self.model_table = QTableWidget(model_tab)
        self.model_table.setColumnCount(0)
        self.model_table.setRowCount(0)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.model_table.setAlternatingRowColors(True)
        model_layout.addWidget(self.model_table, 1)

        # --- Optimization tab ---
        opt_tab = QWidget(self.tabs)
        self.tabs.addTab(opt_tab, "Optimization")
        opt_layout = QVBoxLayout(opt_tab)

        opt_form = QFormLayout()
        self.combo_algorithm = QComboBox(opt_tab)
        self.combo_algorithm.addItems(["Newton-Raphson", "Levenberg-Marquardt"])
        opt_form.addRow("Algorithm for C", self.combo_algorithm)

        self.combo_model_settings = QComboBox(opt_tab)
        self.combo_model_settings.addItems(["Free", "Step by step", "Non-cooperative"])
        self.combo_model_settings.currentIndexChanged.connect(self._on_model_settings_changed)
        opt_form.addRow("Model settings", self.combo_model_settings)

        self.combo_optimizer = QComboBox(opt_tab)
        self.combo_optimizer.addItems(
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
        opt_form.addRow("Optimizer", self.combo_optimizer)

        self.chk_multistart = QCheckBox("Multi-start", opt_tab)
        self.edit_multistart = QLineEdit(opt_tab)
        self.edit_multistart.setPlaceholderText("runs o rango de semillas (ej. 10 o 1-10)")
        self.edit_multistart.setEnabled(False)
        self.chk_multistart.toggled.connect(self.edit_multistart.setEnabled)
        multi_widget = QWidget(opt_tab)
        multi_layout = QHBoxLayout(multi_widget)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.addWidget(self.chk_multistart)
        multi_layout.addWidget(self.edit_multistart, 1)
        opt_form.addRow(multi_widget)
        opt_layout.addLayout(opt_form)

        opt_layout.addWidget(QLabel("Parameters (Initial Estimates)"), 0)

        self.params_table = QTableWidget(opt_tab)
        self.params_table.setColumnCount(5)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Fixed"])
        self.params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.params_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.params_table.setAlternatingRowColors(True)
        self.params_table.itemChanged.connect(self._on_params_item_changed)
        opt_layout.addWidget(self.params_table, 1)

        # --- Plots tab ---
        plots_tab = QWidget(self.tabs)
        self.tabs.addTab(plots_tab, "Plots")
        plots_layout = QVBoxLayout(plots_tab)

        roles_group = QGroupBox("Roles", plots_tab)
        self.roles_group = roles_group
        roles_form = QFormLayout(roles_group)
        self.combo_receptor = QComboBox(roles_group)
        self.combo_guest = QComboBox(roles_group)
        roles_form.addRow("Receptor or Ligand", self.combo_receptor)
        roles_form.addRow("Guest, Metal or Titrant", self.combo_guest)
        plots_layout.addWidget(roles_group)

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
        self.btn_export_png.setToolTip("TODO: Port Plotly export to Qt.")
        self.btn_export_csv = QPushButton("Export CSV", plots_tab)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.setToolTip("TODO: Port Plotly export to Qt.")
        export_row.addWidget(self.btn_export_png)
        export_row.addWidget(self.btn_export_csv)
        export_row.addStretch(1)
        plots_layout.addLayout(export_row)
        plots_layout.addStretch(1)

        # --- Errors tab ---
        errors_tab = QWidget(self.tabs)
        self.tabs.addTab(errors_tab, "Errors")
        self._build_errors_tab(errors_tab)

        self._reset_roles_ui()

    def _build_errors_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        controls_grid = QGridLayout()
        controls_grid.setColumnStretch(1, 1)
        controls_grid.setColumnStretch(3, 1)
        controls_grid.setColumnStretch(5, 1)
        controls_grid.setColumnStretch(7, 1)

        controls_grid.addWidget(QLabel("Error method"), 0, 0)
        self.combo_error_method = QComboBox(parent)
        self.combo_error_method.addItem("Analytical (VarPro covariance)", "analytic")
        self.combo_error_method.addItem("Bootstrap (linearized + wild)", "bootstrap_linear")
        self.combo_error_method.addItem("Bootstrap (one-step LM)", "bootstrap_onestep")
        self.combo_error_method.addItem("Bootstrap (full refit, audit)", "bootstrap_full_refit_audit")
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

        controls_grid.addWidget(QLabel("Wild type"), 1, 4)
        self.combo_error_wild = QComboBox(parent)
        self.combo_error_wild.addItem("Rademacher (+/-1)", "rademacher")
        self.combo_error_wild.addItem("Mammen", "mammen")
        controls_grid.addWidget(self.combo_error_wild, 1, 5)

        controls_grid.addWidget(QLabel("LM lambda"), 1, 6)
        self.spin_error_lambda = QDoubleSpinBox(parent)
        self.spin_error_lambda.setDecimals(6)
        self.spin_error_lambda.setRange(0.0, 1e6)
        self.spin_error_lambda.setSingleStep(1e-3)
        self.spin_error_lambda.setValue(1e-3)
        controls_grid.addWidget(self.spin_error_lambda, 1, 7)

        self.lbl_error_max_iter = QLabel("Max iter", parent)
        controls_grid.addWidget(self.lbl_error_max_iter, 2, 0)
        self.spin_error_max_iter = QSpinBox(parent)
        self.spin_error_max_iter.setRange(5, 200)
        self.spin_error_max_iter.setValue(30)
        controls_grid.addWidget(self.spin_error_max_iter, 2, 1)

        self.lbl_error_tol = QLabel("Tol", parent)
        controls_grid.addWidget(self.lbl_error_tol, 2, 2)
        self.spin_error_tol = QDoubleSpinBox(parent)
        self.spin_error_tol.setDecimals(12)
        self.spin_error_tol.setRange(1e-12, 1e-2)
        self.spin_error_tol.setSingleStep(1e-8)
        self.spin_error_tol.setValue(1e-8)
        controls_grid.addWidget(self.spin_error_tol, 2, 3)

        self.lbl_error_fail_policy = QLabel("Fail policy", parent)
        controls_grid.addWidget(self.lbl_error_fail_policy, 2, 4)
        self.combo_error_fail_policy = QComboBox(parent)
        self.combo_error_fail_policy.addItem("Skip failed replicates", "skip")
        self.combo_error_fail_policy.addItem("Stop on first failure", "stop")
        controls_grid.addWidget(self.combo_error_fail_policy, 2, 5, 1, 3)

        layout.addLayout(controls_grid)

        self.lbl_error_audit_warning = QLabel(
            "Warning: Bootstrap (full refit, audit) can take several minutes. "
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
                "Estimate log10K",
                "Median",
                "CI 2.5",
                "CI 97.5",
                "CI 16",
                "CI 84",
                "SE(log10K)",
                "%err(K)",
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

    # ---- Roles (Plots tab) ----
    def _reset_roles_ui(self) -> None:
        self.combo_receptor.blockSignals(True)
        self.combo_guest.blockSignals(True)
        self.combo_receptor.clear()
        self.combo_guest.clear()
        self.combo_receptor.addItem("Auto", "")
        self.combo_guest.addItem("Auto", "")
        self.combo_receptor.blockSignals(False)
        self.combo_guest.blockSignals(False)

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
        is_full_refit = method == "bootstrap_full_refit_audit"
        self.spin_error_b.setEnabled(is_bootstrap)
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

        # Default B depending on method (keep user's value if it differs).
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

    def _build_solver(self, algorithm: str, C_T, modelo, nas, model_settings: str):
        from hmfit_core.solvers import NewtonRaphson, LevenbergMarquardt

        algo = str(algorithm or "Newton-Raphson")
        if algo == "Newton-Raphson":
            return NewtonRaphson(C_T, modelo, nas, model_settings)
        if algo == "Levenberg-Marquardt":
            return LevenbergMarquardt(C_T, modelo, nas, model_settings)
        raise ValueError(f"Unknown algorithm: {algo}")

    def _collect_errors_options(self) -> dict[str, Any]:
        method = str(self.combo_error_method.currentData() or "analytic")
        seed = self._parse_error_seed()
        wild = str(self.combo_error_wild.currentData() or "rademacher")
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
        if method == "bootstrap_full_refit_audit":
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
            bootstrap_one_step_nmr,
            bootstrap_one_step_spectro,
            compute_errors_nmr_varpro,
            compute_errors_spectro_varpro,
            percent_error_log10K,
            wild_multipliers,
        )
        from hmfit_core.processors.nmr_processor import build_D_cols

        method = str(self.combo_error_method.currentData() or "analytic")
        seed = self._parse_error_seed()
        wild = str(self.combo_error_wild.currentData() or "rademacher")
        lam = float(self.spin_error_lambda.value())
        B = int(self.spin_error_b.value())
        include_16_84 = bool(self.chk_error_ci_16_84.isChecked())
        max_iter = int(self.spin_error_max_iter.value())
        tol = float(self.spin_error_tol.value())
        fail_policy = str(self.combo_error_fail_policy.currentData() or "skip")

        technique = str(ctx.get("technique") or "")
        k_hat_raw = ctx.get("k_hat")
        k_hat = np.asarray(k_hat_raw if k_hat_raw is not None else [], dtype=float).ravel()
        if k_hat.size == 0:
            raise ValueError("Missing fitted parameters.")

        C_T_raw = ctx.get("C_T")
        C_T = np.asarray(C_T_raw if C_T_raw is not None else [], dtype=float)
        if C_T.size == 0:
            raise ValueError("Missing concentration matrix (C_T).")

        modelo_raw = ctx.get("modelo_solver")
        modelo = np.asarray(modelo_raw if modelo_raw is not None else [], dtype=float)
        nas_raw = ctx.get("non_abs_species")
        nas = list(nas_raw) if nas_raw is not None else []
        algorithm = str(ctx.get("algorithm") or "Newton-Raphson")
        model_settings = str(ctx.get("model_settings") or "Free")
        param_names_raw = ctx.get("param_names")
        param_names = (
            list(param_names_raw)
            if param_names_raw is not None
            else [f"K{i+1}" for i in range(k_hat.size)]
        )
        fixed_mask_raw = ctx.get("fixed_mask")
        fixed_mask = (
            np.asarray(fixed_mask_raw, dtype=bool)
            if fixed_mask_raw is not None
            else np.zeros(k_hat.size, dtype=bool)
        )
        stoich_raw = ctx.get("stoichiometry_map")
        stoichiometry = None
        if stoich_raw is not None:
            try:
                stoichiometry = np.asarray(stoich_raw, dtype=float)
                if stoichiometry.size == 0:
                    stoichiometry = None
            except Exception:
                stoichiometry = None

        res = self._build_solver(algorithm, C_T, modelo, nas, model_settings)

        base_metrics = None
        samples = None
        ci_16 = None
        ci_84 = None
        corr = None
        n_success = None
        n_fail = None
        boot_summary = None
        n_fallback = None
        fallback_methods = None

        if technique == "spectro":
            Y_raw = ctx.get("Y")
            Y = np.asarray(Y_raw if Y_raw is not None else [], dtype=float)
            if Y.size == 0:
                raise ValueError("Missing spectra matrix (Y).")
            base_metrics = compute_errors_spectro_varpro(
                k_hat,
                res,
                Y,
                modelo,
                nas,
                rcond=1e-10,
                use_projector=True,
                param_names=param_names,
            )

            if method == "bootstrap_linear":
                free_idx = np.where(~fixed_mask)[0]
                if free_idx.size == 0:
                    samples = np.tile(k_hat, (B, 1))
                else:
                    boot = bootstrap_linearized_wild(
                        k_hat[free_idx],
                        np.asarray(base_metrics["J"], dtype=float)[free_idx, :],
                        base_metrics["r"],
                        B,
                        seed=seed,
                        wild=wild,
                        rcond=1e-10,
                        ridge=0.0,
                    )
                    samples = np.tile(k_hat, (B, 1))
                    samples[:, free_idx] = boot["samples"]
            elif method == "bootstrap_onestep":
                boot = bootstrap_one_step_spectro(
                    k_hat,
                    res,
                    Y,
                    modelo,
                    nas,
                    B,
                    seed=seed,
                    wild=wild,
                    lam=lam,
                    rcond=1e-10,
                    use_projector=True,
                )
                samples = boot["samples"]
            elif method == "bootstrap_full_refit_audit":
                y_fit_hat_raw = ctx.get("y_fit_hat")
                y_fit_hat = np.asarray(y_fit_hat_raw if y_fit_hat_raw is not None else [], dtype=float)
                if y_fit_hat.size == 0:
                    raise ValueError("Missing fitted spectra (y_fit) for full refit.")
                if y_fit_hat.shape != Y.shape:
                    raise ValueError(
                        f"y_fit shape {y_fit_hat.shape} does not match Y {Y.shape}."
                    )
                r_hat = Y - y_fit_hat

                def make_data_star_fn(rng, wild=wild):
                    v = wild_multipliers(rng, r_hat.size, kind=wild).reshape(r_hat.shape)
                    return y_fit_hat + r_hat * v

                refit_fn = ctx.get("refit_from_data")
                if not callable(refit_fn):
                    raise ValueError("Missing refit function for full refit.")

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
                n_success = boot_summary["n_success"]
                n_fail = boot_summary["n_fail"]
                corr = boot_summary["corr"]
                n_fallback = boot_summary.get("n_fallback")
                fallback_methods = boot_summary.get("fallback_methods")
        elif technique == "nmr":
            dq_raw = ctx.get("dq")
            dq_fit_raw = ctx.get("dq_fit")
            dq = np.asarray(dq_raw if dq_raw is not None else [], dtype=float)
            dq_fit = np.asarray(dq_fit_raw if dq_fit_raw is not None else [], dtype=float)
            if dq.size == 0:
                raise ValueError("Missing NMR data matrix (dq).")
            if dq_fit.size == 0:
                raise ValueError("Missing NMR fit matrix (dq_fit).")

            column_names_raw = ctx.get("column_names")
            signal_names_raw = ctx.get("signal_names")
            column_names = list(column_names_raw) if column_names_raw is not None else []
            signal_names = list(signal_names_raw) if signal_names_raw is not None else []
            if not signal_names:
                raise ValueError("Missing signal names for NMR errors.")

            D_cols = ctx.get("D_cols")
            if D_cols is None:
                D_cols, _ = build_D_cols(C_T, column_names, signal_names, default_idx=0)

            mask = ctx.get("mask")
            if mask is None:
                mask = np.isfinite(dq) & np.isfinite(D_cols) & (np.abs(D_cols) > 0)

            base_metrics = compute_errors_nmr_varpro(
                k_hat,
                res,
                dq,
                D_cols,
                modelo,
                nas,
                stoichiometry=stoichiometry,
                mask=mask,
                fixed_mask=fixed_mask,
                rcond=1e-10,
                rcond_cov=1e-10,
                ridge=1e-8,
                ridge_cov=0.0,
                use_projector=True,
                param_names=param_names,
            )

            if method == "bootstrap_linear":
                free_idx = np.where(~fixed_mask)[0]
                if free_idx.size == 0:
                    samples = np.tile(k_hat, (B, 1))
                else:
                    boot = bootstrap_linearized_wild(
                        k_hat[free_idx],
                        np.asarray(base_metrics["J"], dtype=float)[free_idx, :],
                        base_metrics["r"],
                        B,
                        seed=seed,
                        wild=wild,
                        rcond=1e-10,
                        ridge=0.0,
                    )
                    samples = np.tile(k_hat, (B, 1))
                    samples[:, free_idx] = boot["samples"]
            elif method == "bootstrap_onestep":
                boot = bootstrap_one_step_nmr(
                    k_hat,
                    res,
                    dq,
                    dq_fit,
                    D_cols,
                    modelo,
                    nas,
                    B,
                    stoichiometry=stoichiometry,
                    seed=seed,
                    wild=wild,
                    lam=lam,
                    rcond=1e-10,
                    use_projector=True,
                    mask=mask,
                    fixed_mask=fixed_mask,
                    rcond_cov=1e-10,
                    ridge=1e-8,
                    ridge_cov=0.0,
                )
                samples = boot["samples"]
            elif method == "bootstrap_full_refit_audit":
                dq_fit = np.asarray(dq_fit, dtype=float)
                if dq_fit.shape != dq.shape:
                    raise ValueError(
                        f"dq_fit shape {dq_fit.shape} does not match dq {dq.shape}."
                    )

                mask_full = np.asarray(mask, dtype=bool)
                mask_full &= np.isfinite(dq_fit)
                r_hat = dq - dq_fit

                def make_data_star_fn(rng, wild=wild):
                    v = wild_multipliers(rng, r_hat.size, kind=wild).reshape(r_hat.shape)
                    dq_star = dq_fit + r_hat * v
                    dq_star[~mask_full] = dq[~mask_full]
                    return dq_star

                refit_fn = ctx.get("refit_from_data")
                if not callable(refit_fn):
                    raise ValueError("Missing refit function for full refit.")

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
                n_success = boot_summary["n_success"]
                n_fail = boot_summary["n_fail"]
                corr = boot_summary["corr"]
        else:
            raise ValueError("Unknown technique for errors.")

        if base_metrics is None:
            raise ValueError("Base metrics unavailable.")

        if method == "analytic":
            se_raw = base_metrics.get("SE_log10K")
            perc_raw = base_metrics.get("percK")
            se_log10 = np.asarray(se_raw if se_raw is not None else [], dtype=float).ravel()
            perc_err = np.asarray(perc_raw if perc_raw is not None else [], dtype=float).ravel()
            z95 = 1.96
            z68 = 1.0
            median = k_hat.copy()
            ci_2p5 = k_hat - z95 * se_log10
            ci_97p5 = k_hat + z95 * se_log10
            if include_16_84:
                ci_16 = k_hat - z68 * se_log10
                ci_84 = k_hat + z68 * se_log10

            corr = self._corr_from_cov(base_metrics, k_hat.size)
        elif method == "bootstrap_full_refit_audit":
            if samples is None or samples.size == 0:
                detail = ""
                if boot_summary is not None:
                    n_fail = boot_summary.get("n_fail")
                    fail_info = boot_summary.get("fail_info") or []
                    if n_fail is not None:
                        detail = f" All {int(n_fail)} replicates failed."
                    if fail_info:
                        last_info = fail_info[-1]
                        last_err = (
                            last_info.get("error")
                            if isinstance(last_info, dict)
                            else str(last_info)
                        )
                        if last_err:
                            detail = f"{detail} Last error: {last_err}"
                raise ValueError(f"Bootstrap samples are empty.{detail}")
            if boot_summary is not None and boot_summary.get("median") is not None:
                median = np.asarray(boot_summary["median"], dtype=float)
                ci_2p5 = np.asarray(boot_summary["p2_5"], dtype=float)
                ci_97p5 = np.asarray(boot_summary["p97_5"], dtype=float)
                if include_16_84:
                    ci_16 = (
                        np.asarray(boot_summary["p16"], dtype=float)
                        if boot_summary.get("p16") is not None
                        else np.percentile(samples, 16.0, axis=0)
                    )
                    ci_84 = (
                        np.asarray(boot_summary["p84"], dtype=float)
                        if boot_summary.get("p84") is not None
                        else np.percentile(samples, 84.0, axis=0)
                    )
            else:
                median = np.median(samples, axis=0)
                ci_2p5 = np.percentile(samples, 2.5, axis=0)
                ci_97p5 = np.percentile(samples, 97.5, axis=0)
                if include_16_84:
                    ci_16 = np.percentile(samples, 16.0, axis=0)
                    ci_84 = np.percentile(samples, 84.0, axis=0)
            ddof = 1 if samples.shape[0] > 1 else 0
            se_log10 = np.std(samples, axis=0, ddof=ddof)
            perc_err, _, _ = percent_error_log10K(median, se_log10)
        else:
            if samples is None or samples.size == 0:
                detail = ""
                if boot_summary is not None:
                    n_fail = boot_summary.get("n_fail")
                    fail_info = boot_summary.get("fail_info") or []
                    if n_fail is not None:
                        detail = f" All {int(n_fail)} replicates failed."
                    if fail_info:
                        last_info = fail_info[-1]
                        last_err = (
                            last_info.get("error")
                            if isinstance(last_info, dict)
                            else str(last_info)
                        )
                        if last_err:
                            detail = f"{detail} Last error: {last_err}"
                raise ValueError(f"Bootstrap samples are empty.{detail}")
            median = np.median(samples, axis=0)
            ci_2p5 = np.percentile(samples, 2.5, axis=0)
            ci_97p5 = np.percentile(samples, 97.5, axis=0)
            if include_16_84:
                ci_16 = np.percentile(samples, 16.0, axis=0)
                ci_84 = np.percentile(samples, 84.0, axis=0)
            ddof = 1 if samples.shape[0] > 1 else 0
            se_log10 = np.std(samples, axis=0, ddof=ddof)
            perc_err, _, _ = percent_error_log10K(median, se_log10)
            corr = self._corr_from_samples(samples)

        summary = self._build_errors_summary(
            method=method,
            B=B,
            seed=seed,
            wild=wild,
            lam=lam,
            metrics=base_metrics,
            max_iter=max_iter,
            tol=tol,
            fail_policy=fail_policy,
            n_success=n_success,
            n_fail=n_fail,
            n_fallback=n_fallback,
            fallback_methods=fallback_methods,
        )

        rms_val = base_metrics.get("RMS")
        if rms_val is None:
            rms_val = base_metrics.get("rms")
        s2_val = base_metrics.get("s2")
        if s2_val is None:
            s2_val = base_metrics.get("covfit")
        dof_val = base_metrics.get("dof")

        return {
            "param_names": param_names,
            "k_hat": k_hat,
            "median": median,
            "ci_2p5": ci_2p5,
            "ci_97p5": ci_97p5,
            "ci_16": ci_16,
            "ci_84": ci_84,
            "se_log10": se_log10,
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
            "rms": rms_val,
            "s2": s2_val,
            "dof": dof_val,
        }

    def _corr_from_cov(self, metrics: dict[str, Any], p_total: int) -> np.ndarray:
        cov_full = np.zeros((p_total, p_total), dtype=float)
        if "Cov_log10K" in metrics:
            cov = np.asarray(metrics["Cov_log10K"], dtype=float)
            cov_full = cov
        elif "Cov_log10K_free" in metrics:
            cov_free_raw = metrics.get("Cov_log10K_free")
            cov_free = np.asarray(cov_free_raw if cov_free_raw is not None else [], dtype=float)
            free_idx = np.arange(p_total)
            if "SE_log10K" in metrics:
                se_log10 = np.asarray(metrics["SE_log10K"], dtype=float).ravel()
                free_idx = np.where(se_log10 > 0)[0]
            if cov_free.size > 0:
                cov_full = np.zeros((p_total, p_total), dtype=float)
                cov_full[np.ix_(free_idx, free_idx)] = cov_free

        d = np.sqrt(np.clip(np.diag(cov_full), 1e-30, None))
        denom = np.outer(d, d)
        corr = np.divide(cov_full, denom, where=denom > 0)
        corr = np.clip(corr, -1.0, 1.0)
        for i in range(p_total):
            corr[i, i] = 1.0
        return corr

    def _corr_from_samples(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] < 2:
            return np.eye(samples.shape[1], dtype=float)
        corr = np.corrcoef(samples, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        for i in range(corr.shape[0]):
            corr[i, i] = 1.0
        return corr

    def _build_errors_summary(
        self,
        *,
        method: str,
        B: int,
        seed: int | None,
        wild: str,
        lam: float,
        metrics: dict[str, Any],
        max_iter: int | None = None,
        tol: float | None = None,
        fail_policy: str | None = None,
        n_success: int | None = None,
        n_fail: int | None = None,
        n_fallback: int | None = None,
        fallback_methods: list[str] | None = None,
    ) -> str:
        method_labels = {
            "analytic": "Analytical (VarPro covariance)",
            "bootstrap_linear": "Bootstrap (linearized + wild)",
            "bootstrap_onestep": "Bootstrap (one-step LM)",
            "bootstrap_full_refit_audit": "Bootstrap (full refit, audit)",
        }
        lines = [f"Method: {method_labels.get(method, method)}"]
        if method != "analytic":
            seed_txt = "random" if seed is None else str(seed)
            if method == "bootstrap_full_refit_audit":
                lines.append("method=full_refit_audit")
                lines.append(f"B={B}, seed={seed_txt}, wild_type={wild}")
                if max_iter is not None and tol is not None:
                    lines.append(f"max_iter={int(max_iter)}, tol={float(tol):.3g}")
                if fail_policy:
                    lines.append(f"fail_policy={fail_policy}")
                if n_success is not None and n_fail is not None:
                    lines.append(f"n_success={int(n_success)}, n_fail={int(n_fail)}")
                if n_fallback:
                    methods = ", ".join(fallback_methods or [])
                    suffix = f" [{methods}]" if methods else ""
                    lines.append(f"fallback_used={int(n_fallback)}{suffix}")
            else:
                lines.append(f"B={B}, seed={seed_txt}, wild={wild}, lambda={lam:g}")

        rms = metrics.get("RMS")
        if rms is None:
            rms = metrics.get("rms")
        s2 = metrics.get("s2")
        if s2 is None:
            s2 = metrics.get("covfit")
        dof = metrics.get("dof")
        if rms is not None:
            lines.append(f"RMS={float(rms):.6g}")
        if s2 is not None:
            lines.append(f"s2={float(s2):.6g}")
        if dof is not None:
            lines.append(f"dof={int(dof)}")

        diag = metrics.get("stability_diag") or {}
        if isinstance(diag, dict) and diag:
            status = diag.get("status")
            summary = diag.get("diag_summary")
            if status in ("warn", "critical") and summary:
                lines.append(f"Note: {summary}")

        return "\n".join(lines)

    def _apply_errors_output(self, output: dict[str, Any]) -> None:
        param_names_raw = output.get("param_names")
        k_hat_raw = output.get("k_hat")
        median_raw = output.get("median")
        ci_2p5_raw = output.get("ci_2p5")
        ci_97p5_raw = output.get("ci_97p5")
        param_names = list(param_names_raw) if param_names_raw is not None else []
        k_hat = np.asarray(k_hat_raw if k_hat_raw is not None else [], dtype=float)
        median = np.asarray(median_raw if median_raw is not None else [], dtype=float)
        ci_2p5 = np.asarray(ci_2p5_raw if ci_2p5_raw is not None else [], dtype=float)
        ci_97p5 = np.asarray(ci_97p5_raw if ci_97p5_raw is not None else [], dtype=float)
        ci_16 = output.get("ci_16")
        ci_84 = output.get("ci_84")
        se_log10_raw = output.get("se_log10")
        perc_err_raw = output.get("perc_err")
        se_log10 = np.asarray(se_log10_raw if se_log10_raw is not None else [], dtype=float)
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
            self._set_error_cell(row, 7, self._fmt_num(se_log10, row))
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
            param_names_raw = output.get("param_names")
            param_names = list(param_names_raw) if param_names_raw is not None else []
            df = pd.DataFrame(
                {
                    "Parameter": param_names,
                    "Estimate log10K": output.get("k_hat"),
                    "Median": output.get("median"),
                    "CI 2.5": output.get("ci_2p5"),
                    "CI 97.5": output.get("ci_97p5"),
                    "CI 16": output.get("ci_16"),
                    "CI 84": output.get("ci_84"),
                    "SE(log10K)": output.get("se_log10"),
                    "%err(K)": output.get("perc_err"),
                }
            )
            method_raw = str(output.get("method") or "")
            method_tag = "full_refit_audit" if method_raw == "bootstrap_full_refit_audit" else method_raw
            lam_val = output.get("lam") if method_raw == "bootstrap_onestep" else None
            wild_val = output.get("wild") if method_raw != "analytic" else None
            seed_raw = output.get("seed") if method_raw != "analytic" else None
            seed_val = "random" if seed_raw is None and method_raw != "analytic" else seed_raw
            b_val = output.get("B") if method_raw != "analytic" else None
            max_iter_val = output.get("max_iter") if method_raw == "bootstrap_full_refit_audit" else None
            tol_val = output.get("tol") if method_raw == "bootstrap_full_refit_audit" else None
            n_success_val = output.get("n_success") if method_raw == "bootstrap_full_refit_audit" else None
            n_fail_val = output.get("n_fail") if method_raw == "bootstrap_full_refit_audit" else None

            meta_rows = [
                ("method", method_tag),
                ("B", b_val),
                ("seed", seed_val),
                ("wild_type", wild_val),
                ("lambda", lam_val),
                ("max_iter", max_iter_val),
                ("tol", tol_val),
                ("dof", output.get("dof")),
                ("RMS", output.get("rms")),
                ("s2", output.get("s2")),
                ("n_success", n_success_val),
                ("n_fail", n_fail_val),
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

    def set_available_conc_columns(self, selected_columns: list[str]) -> None:
        self._available_conc_columns = [str(c) for c in (selected_columns or []) if str(c).strip()]
        prev_receptor = str(self.combo_receptor.currentData() or "")
        prev_guest = str(self.combo_guest.currentData() or "")

        self.combo_receptor.blockSignals(True)
        self.combo_guest.blockSignals(True)
        self.combo_receptor.clear()
        self.combo_guest.clear()
        self.combo_receptor.addItem("Auto", "")
        self.combo_guest.addItem("Auto", "")
        for col in self._available_conc_columns:
            self.combo_receptor.addItem(col, col)
            self.combo_guest.addItem(col, col)
        self.combo_receptor.blockSignals(False)
        self.combo_guest.blockSignals(False)

        def _restore(combo: QComboBox, value: str) -> None:
            ix = combo.findData(value)
            if ix >= 0:
                combo.setCurrentIndex(ix)

        _restore(self.combo_receptor, prev_receptor)
        _restore(self.combo_guest, prev_guest)

    def resolve_roles(self) -> tuple[str, str]:
        receptor_pref = str(self.combo_receptor.currentData() or "").strip()
        guest_pref = str(self.combo_guest.currentData() or "").strip()
        return resolve_receptor_guest(
            list(self._available_conc_columns),
            receptor_preference=receptor_pref,
            guest_preference=guest_pref,
        )

    # ---- Model tab ----
    def _on_define_model_clicked(self) -> None:
        n_components = int(self.spin_n_components.value())
        n_species = int(self.spin_n_species.value())
        if n_components <= 0 or n_species <= 0:
            QMessageBox.warning(self, "Invalid model", "Enter Number of Components and Species (>0).")
            return
        self.set_model_dimensions(n_components, n_species)
        self.model_defined.emit(n_components, n_species)

    def set_model_dimensions(self, n_components: int, n_species: int) -> None:
        n_components = max(0, int(n_components))
        n_species = max(0, int(n_species))

        self.spin_n_components.blockSignals(True)
        self.spin_n_species.blockSignals(True)
        self.spin_n_components.setValue(n_components)
        self.spin_n_species.setValue(n_species)
        self.spin_n_components.blockSignals(False)
        self.spin_n_species.blockSignals(False)

        total_rows = n_components + n_species
        self.model_table.clear()
        self.model_table.setRowCount(total_rows)
        self.model_table.setColumnCount(n_components)
        self.model_table.setHorizontalHeaderLabels([f"C{i+1}" for i in range(n_components)])
        self.model_table.setVerticalHeaderLabels([f"sp{i+1}" for i in range(total_rows)])

        for row in range(total_rows):
            for col in range(n_components):
                if row < n_components:
                    value = "1.0" if col == row else "0.0"
                else:
                    value = "0"
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.model_table.setItem(row, col, item)

        self.model_table.resizeColumnsToContents()
        self._sync_param_row_count_from_model()

    def get_modelo(self) -> list[list[float]]:
        rows = self.model_table.rowCount()
        cols = self.model_table.columnCount()
        data: list[list[float]] = []
        for r in range(rows):
            row_vals: list[float] = []
            for c in range(cols):
                item = self.model_table.item(r, c)
                row_vals.append(_coerce_float(item.text() if item is not None else "", default=0.0))
            data.append(row_vals)
        return data

    def set_modelo(self, modelo: list[list[float]]) -> None:
        rows = self.model_table.rowCount()
        cols = self.model_table.columnCount()
        for r in range(min(rows, len(modelo))):
            row_data = modelo[r] or []
            for c in range(min(cols, len(row_data))):
                item = self.model_table.item(r, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.model_table.setItem(r, c, item)
                item.setText(str(_coerce_float(row_data[c], default=0.0)))

    def get_non_abs_species(self) -> list[int]:
        rows = sorted({idx.row() for idx in self.model_table.selectionModel().selectedRows()})
        return [int(r) for r in rows]

    def set_non_abs_species(self, indices: list[int]) -> None:
        selection = self.model_table.selectionModel()
        if selection is None:
            return
        selection.clearSelection()
        wanted = {int(x) for x in (indices or []) if int(x) >= 0}
        for row in sorted(wanted):
            if row < 0 or row >= self.model_table.rowCount():
                continue
            # Add to selection (do not replace previous rows).
            idx = self.model_table.model().index(row, 0)
            selection.select(idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    # ---- Optimization tab ----
    def _n_constants(self) -> int:
        n_species = int(self.spin_n_species.value())
        if n_species <= 0:
            return 0
        if self.combo_model_settings.currentText() == "Non-cooperative":
            return 1
        return n_species

    def _sync_param_row_count_from_model(self) -> None:
        self._set_param_row_count(self._n_constants())

    def _on_model_settings_changed(self) -> None:
        first = self._read_param_row(0) if self.params_table.rowCount() > 0 else None
        self._set_param_row_count(self._n_constants())
        if first is not None and self.params_table.rowCount() > 0:
            self._write_param_row(0, first)

    def _set_param_row_count(self, n: int) -> None:
        n = max(0, int(n))
        self._in_table_update = True
        try:
            current = self.params_table.rowCount()
            if current == n:
                return
            if current < n:
                for _ in range(n - current):
                    self._append_param_row()
            else:
                for row in range(current - 1, n - 1, -1):
                    self.params_table.removeRow(row)
        finally:
            self._in_table_update = False

        # Renumber
        for row in range(self.params_table.rowCount()):
            it = self.params_table.item(row, 0)
            if it is not None:
                it.setText(f"K{row + 1}")

        self.params_table.resizeColumnsToContents()

    def _append_param_row(self) -> None:
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)

        name_item = QTableWidgetItem(f"K{row + 1}")
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.params_table.setItem(row, 0, name_item)

        for col in (1, 2, 3):
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.params_table.setItem(row, col, item)

        fixed_item = QTableWidgetItem("")
        fixed_item.setFlags(
            (fixed_item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
        )
        fixed_item.setCheckState(Qt.CheckState.Unchecked)
        fixed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_table.setItem(row, 4, fixed_item)

    def _on_params_item_changed(self, item: QTableWidgetItem) -> None:
        if self._in_table_update:
            return
        if item is None:
            return
        row = int(item.row())
        col = int(item.column())
        if row < 0 or row >= self.params_table.rowCount():
            return

        fixed_item = self.params_table.item(row, 4)
        is_fixed = fixed_item is not None and fixed_item.checkState() == Qt.CheckState.Checked

        if col in (1, 4):  # Value changed or Fixed toggled
            value_item = self.params_table.item(row, 1)
            val = _coerce_float_or_none(value_item.text() if value_item is not None else "")

            if is_fixed and val is not None:
                self._in_table_update = True
                try:
                    min_item = self.params_table.item(row, 2)
                    max_item = self.params_table.item(row, 3)
                    if min_item is None:
                        min_item = QTableWidgetItem("")
                        self.params_table.setItem(row, 2, min_item)
                    if max_item is None:
                        max_item = QTableWidgetItem("")
                        self.params_table.setItem(row, 3, max_item)
                    min_item.setText(str(val))
                    max_item.setText(str(val))
                finally:
                    self._in_table_update = False

            # Disable / enable bounds editing based on Fixed checkbox
            for bounds_col in (2, 3):
                bounds_item = self.params_table.item(row, bounds_col)
                if bounds_item is None:
                    continue
                flags = bounds_item.flags()
                if is_fixed:
                    bounds_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    bounds_item.setFlags(flags | Qt.ItemFlag.ItemIsEditable)

    def _read_param_row(self, row: int) -> dict[str, Any]:
        def txt(c: int) -> str:
            it = self.params_table.item(row, c)
            return str(it.text() if it is not None else "")

        fixed_it = self.params_table.item(row, 4)
        fixed = fixed_it is not None and fixed_it.checkState() == Qt.CheckState.Checked
        return {"value": txt(1), "min": txt(2), "max": txt(3), "fixed": fixed}

    def _write_param_row(self, row: int, data: dict[str, Any]) -> None:
        self._in_table_update = True
        try:
            for col, key in ((1, "value"), (2, "min"), (3, "max")):
                it = self.params_table.item(row, col)
                if it is None:
                    it = QTableWidgetItem("")
                    self.params_table.setItem(row, col, it)
                it.setText(str(data.get(key, "")))

            fixed_it = self.params_table.item(row, 4)
            if fixed_it is not None:
                fixed_it.setCheckState(Qt.CheckState.Checked if data.get("fixed") else Qt.CheckState.Unchecked)
        finally:
            self._in_table_update = False

        # Trigger fixed semantics
        fixed_it = self.params_table.item(row, 4)
        if fixed_it is not None:
            self._on_params_item_changed(fixed_it)

    def get_optimization(self) -> tuple[list[float], list[tuple[float | None, float | None]], list[bool]]:
        initial_k: list[float] = []
        bounds: list[tuple[float | None, float | None]] = []
        fixed_mask: list[bool] = []

        n = self.params_table.rowCount()
        for row in range(n):
            value_item = self.params_table.item(row, 1)
            min_item = self.params_table.item(row, 2)
            max_item = self.params_table.item(row, 3)
            fixed_item = self.params_table.item(row, 4)

            val = _coerce_float_or_none(value_item.text() if value_item is not None else "")
            if val is None:
                val = 1.0  # matches Tauri behavior
            initial_k.append(float(val))

            min_v = _coerce_float_or_none(min_item.text() if min_item is not None else "")
            max_v = _coerce_float_or_none(max_item.text() if max_item is not None else "")

            fixed = fixed_item is not None and fixed_item.checkState() == Qt.CheckState.Checked
            fixed_mask.append(bool(fixed))

            if fixed:
                min_v = float(val)
                max_v = float(val)
            bounds.append((min_v, max_v))

        return initial_k, bounds, fixed_mask

    def set_optimization(
        self,
        *,
        algorithm: str | None = None,
        model_settings: str | None = None,
        optimizer: str | None = None,
        initial_k: list[float] | None = None,
        bounds: list[list[float | None] | tuple[float | None, float | None] | dict[str, Any]] | None = None,
        fixed_mask: list[bool] | None = None,
    ) -> None:
        if algorithm:
            ix = self.combo_algorithm.findText(str(algorithm))
            if ix >= 0:
                self.combo_algorithm.setCurrentIndex(ix)
        if model_settings:
            ix = self.combo_model_settings.findText(str(model_settings))
            if ix >= 0:
                self.combo_model_settings.setCurrentIndex(ix)
        if optimizer:
            ix = self.combo_optimizer.findText(str(optimizer))
            if ix >= 0:
                self.combo_optimizer.setCurrentIndex(ix)

        if initial_k is None and bounds is None and fixed_mask is None:
            return

        n = self.params_table.rowCount()
        initial_k = list(initial_k or [])
        fixed_mask = list(fixed_mask or [])
        bounds = list(bounds or [])

        self._in_table_update = True
        try:
            for row in range(n):
                if row < len(initial_k):
                    it = self.params_table.item(row, 1)
                    if it is None:
                        it = QTableWidgetItem("")
                        self.params_table.setItem(row, 1, it)
                    it.setText(str(_coerce_float(initial_k[row], default=1.0)))

                if row < len(bounds):
                    b = bounds[row]
                    if isinstance(b, dict):
                        b_min = b.get("min")
                        b_max = b.get("max")
                    else:
                        b_min = b[0] if len(b) >= 1 else None  # type: ignore[index]
                        b_max = b[1] if len(b) >= 2 else None  # type: ignore[index]

                    for col, val in ((2, b_min), (3, b_max)):
                        it = self.params_table.item(row, col)
                        if it is None:
                            it = QTableWidgetItem("")
                            self.params_table.setItem(row, col, it)
                        it.setText("" if val is None else str(val))

                if row < len(fixed_mask):
                    it = self.params_table.item(row, 4)
                    if it is not None:
                        it.setCheckState(Qt.CheckState.Checked if fixed_mask[row] else Qt.CheckState.Unchecked)
        finally:
            self._in_table_update = False

        # Apply fixed semantics to all rows
        for row in range(n):
            it = self.params_table.item(row, 4)
            if it is not None:
                self._on_params_item_changed(it)

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
            self.edit_multistart.clear()

    # ---- Export / Import ----
    def collect_state(self) -> ModelOptPlotsState:
        initial_k, bounds, fixed_mask = self.get_optimization()
        receptor_label, guest_label = self.resolve_roles()
        return ModelOptPlotsState(
            n_components=int(self.spin_n_components.value()),
            n_species=int(self.spin_n_species.value()),
            modelo=self.get_modelo(),
            non_abs_species=self.get_non_abs_species(),
            algorithm=str(self.combo_algorithm.currentText()),
            model_settings=str(self.combo_model_settings.currentText()),
            optimizer=str(self.combo_optimizer.currentText()),
            initial_k=initial_k,
            bounds=bounds,
            fixed_mask=fixed_mask,
            receptor_label=receptor_label,
            guest_label=guest_label,
        )

    def apply_state(self, state: ModelOptPlotsState) -> None:
        self.set_model_dimensions(state.n_components, state.n_species)
        self.set_modelo(state.modelo)
        self.set_non_abs_species(state.non_abs_species)
        self.set_optimization(
            algorithm=state.algorithm,
            model_settings=state.model_settings,
            optimizer=state.optimizer,
            initial_k=state.initial_k,
            bounds=state.bounds,
            fixed_mask=state.fixed_mask,
        )

        # Restore roles if possible
        self.set_available_conc_columns(self._available_conc_columns)
        for combo, raw in ((self.combo_receptor, state.receptor_label), (self.combo_guest, state.guest_label)):
            ix = combo.findData(raw)
            if ix >= 0:
                combo.setCurrentIndex(ix)

    def reset(self) -> None:
        self.spin_n_components.setValue(0)
        self.spin_n_species.setValue(0)
        self.model_table.clear()
        self.model_table.setRowCount(0)
        self.model_table.setColumnCount(0)

        self.combo_algorithm.setCurrentIndex(0)
        self.combo_model_settings.setCurrentIndex(0)
        self.combo_optimizer.setCurrentIndex(0)
        self.params_table.setRowCount(0)
        self.chk_multistart.setChecked(False)
        self.edit_multistart.clear()

        self._available_conc_columns = []
        self._reset_roles_ui()
        self.set_errors_context(None)
