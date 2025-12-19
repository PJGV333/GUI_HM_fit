from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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

        self._reset_roles_ui()

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

        self._available_conc_columns = []
        self._reset_roles_ui()
