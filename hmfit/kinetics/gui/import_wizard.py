"""Import wizard for kinetics datasets."""

from __future__ import annotations

import copy
import numpy as np
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QRadioButton,
)

from ..data.fit_dataset import KineticsFitDataset, TechniqueType
from ..data.loaders import load_matrix_file, load_xlsx
from .dataset_editor import DatasetEditorDialog, DatasetEditorWidget
from hmfit_gui_qt.widgets.channel_spec import ChannelSpecWidget


class ImportWizard(QDialog):
    """Wizard dialog to import kinetics datasets."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        dynamic_species: Sequence[str] | None = None,
        fixed_species: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Dataset")
        self._dynamic_species = list(dynamic_species or [])
        self._fixed_species = list(fixed_species or [])
        self._dataset: KineticsFitDataset | None = None
        self._datasets: list[KineticsFitDataset] = []
        self._paths: list[str] = []
        self._raw_t = None
        self._raw_D = None
        self._raw_x = None
        self._raw_labels: list[str] = []
        self._raw_shift_x = None
        self._selected_channel_parents: list[str] = []
        self._ui_ready = False

        self._build_ui()
        self._update_nav()

    @property
    def datasets(self) -> list[KineticsFitDataset]:
        return list(self._datasets)

    @property
    def dataset(self) -> KineticsFitDataset | None:
        return self._datasets[0] if self._datasets else None

    # ---- UI ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget(self)
        layout.addWidget(self._stack, 1)

        self._stack.addWidget(self._build_step_type())
        self._stack.addWidget(self._build_step_file())
        self._stack.addWidget(self._build_step_channels())
        self._stack.addWidget(self._build_step_metadata())

        nav_layout = QHBoxLayout()
        nav_layout.addStretch(1)
        self._btn_back = QPushButton("Back", self)
        self._btn_back.clicked.connect(self._on_back)
        nav_layout.addWidget(self._btn_back)
        self._btn_next = QPushButton("Next", self)
        self._btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self._btn_next)
        self._btn_finish = QPushButton("Finish", self)
        self._btn_finish.clicked.connect(self._on_finish)
        nav_layout.addWidget(self._btn_finish)
        self._btn_cancel = QPushButton("Cancel", self)
        self._btn_cancel.clicked.connect(self.reject)
        nav_layout.addWidget(self._btn_cancel)
        layout.addLayout(nav_layout)
        self._apply_technique_ui()
        self._ui_ready = True

    def _build_step_type(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        group = QGroupBox("Select data type", widget)
        group_layout = QVBoxLayout(group)
        self._technique_group = QButtonGroup(group)

        self._technique_buttons: dict[TechniqueType, QRadioButton] = {}
        options = [
            ("spec_full", "Spectroscopy: full spectrum (time x lambda)"),
            ("spec_channels", "Spectroscopy: few channels"),
            ("nmr_integrals", "NMR: integrals (time x peaks)"),
            ("nmr_full", "NMR: full spectrum (time x ppm)"),
        ]
        for technique, label in options:
            button = QRadioButton(label, group)
            self._technique_group.addButton(button)
            self._technique_buttons[technique] = button
            group_layout.addWidget(button)
            button.toggled.connect(self._on_technique_changed)
        self._technique_buttons["spec_channels"].setChecked(True)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _build_step_file(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)

        file_row = QHBoxLayout()
        self._path_edit = QLineEdit(widget)
        self._path_edit.setPlaceholderText("Select data file(s)")
        file_row.addWidget(self._path_edit, 1)
        btn_browse = QPushButton("Browse...", widget)
        btn_browse.clicked.connect(self._on_browse)
        file_row.addWidget(btn_browse)
        layout.addLayout(file_row)

        options_layout = QFormLayout()
        self._sheet_label = QLabel("Sheet", widget)
        self._sheet_combo = QComboBox(widget)
        self._sheet_combo.currentIndexChanged.connect(self._reload_data)
        options_layout.addRow(self._sheet_label, self._sheet_combo)

        self._conc_sheet_label = QLabel("Concentration sheet (optional)", widget)
        self._conc_sheet_combo = QComboBox(widget)
        self._conc_sheet_combo.currentIndexChanged.connect(self._update_nav)
        options_layout.addRow(self._conc_sheet_label, self._conc_sheet_combo)

        self._shift_sheet_label = QLabel("Shift sheet", widget)
        self._shift_sheet_combo = QComboBox(widget)
        self._shift_sheet_combo.currentIndexChanged.connect(self._reload_data)
        options_layout.addRow(self._shift_sheet_label, self._shift_sheet_combo)

        self._header_check = QCheckBox("Header row", widget)
        self._header_check.setChecked(True)
        self._header_check.stateChanged.connect(self._reload_data)
        options_layout.addRow("", self._header_check)

        self._transpose_check = QCheckBox("Transpose", widget)
        self._transpose_check.stateChanged.connect(self._reload_data)
        options_layout.addRow("", self._transpose_check)

        self._time_col_spin = QSpinBox(widget)
        self._time_col_spin.setMinimum(0)
        self._time_col_spin.valueChanged.connect(self._reload_data)
        options_layout.addRow("Time column", self._time_col_spin)

        self._delimiter_edit = QLineEdit(widget)
        self._delimiter_edit.setPlaceholderText("Auto")
        self._delimiter_edit.textChanged.connect(self._reload_data)
        options_layout.addRow("Delimiter", self._delimiter_edit)

        layout.addLayout(options_layout)

        self._preview = QTableWidget(widget)
        self._preview.setColumnCount(0)
        self._preview.setRowCount(0)
        layout.addWidget(self._preview, 1)

        return widget

    def _build_step_channels(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self._channels_stack = QStackedWidget(widget)

        list_page = QWidget(self._channels_stack)
        list_layout = QVBoxLayout(list_page)
        self._channels_list = QListWidget(list_page)
        self._channels_list.itemChanged.connect(self._on_channel_item_changed)
        list_layout.addWidget(self._channels_list, 1)

        actions_layout = QHBoxLayout()
        btn_all = QPushButton("All", list_page)
        btn_all.clicked.connect(lambda: self._set_all_channels(True))
        actions_layout.addWidget(btn_all)
        btn_none = QPushButton("None", list_page)
        btn_none.clicked.connect(lambda: self._set_all_channels(False))
        actions_layout.addWidget(btn_none)
        actions_layout.addStretch(1)

        self._range_min = QLineEdit(list_page)
        self._range_min.setPlaceholderText("min")
        self._range_max = QLineEdit(list_page)
        self._range_max.setPlaceholderText("max")
        btn_range = QPushButton("Apply range", list_page)
        btn_range.clicked.connect(self._apply_range)
        actions_layout.addWidget(self._range_min)
        actions_layout.addWidget(self._range_max)
        actions_layout.addWidget(btn_range)
        list_layout.addLayout(actions_layout)

        efa_row = QHBoxLayout()
        efa_row.addWidget(QLabel("EFA Eigenvalues", list_page))
        self._efa_check = QCheckBox("EFA", list_page)
        self._efa_check.setChecked(True)
        efa_row.addWidget(self._efa_check)
        self._efa_spin = QSpinBox(list_page)
        self._efa_spin.setRange(0, 999)
        self._efa_spin.setValue(0)
        efa_row.addWidget(self._efa_spin)
        efa_row.addStretch(1)
        list_layout.addLayout(efa_row)

        self._channels_stack.addWidget(list_page)

        spec_page = QWidget(self._channels_stack)
        spec_layout = QVBoxLayout(spec_page)
        self._channel_spec_widget = ChannelSpecWidget(spec_page)
        spec_layout.addWidget(self._channel_spec_widget, 1)
        spec_btns = QHBoxLayout()
        btn_spec_all = QPushButton("Select all", spec_page)
        btn_spec_all.clicked.connect(lambda: self._channel_spec_widget.set_all_checked(True))
        spec_btns.addWidget(btn_spec_all)
        btn_spec_none = QPushButton("Select none", spec_page)
        btn_spec_none.clicked.connect(lambda: self._channel_spec_widget.set_all_checked(False))
        spec_btns.addWidget(btn_spec_none)
        spec_btns.addStretch(1)
        spec_layout.addLayout(spec_btns)
        self._channels_stack.addWidget(spec_page)

        layout.addWidget(self._channels_stack, 1)
        return widget

    def _build_step_metadata(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self._metadata_editor = DatasetEditorWidget(
            widget,
            dynamic_species=self._dynamic_species,
            fixed_species=self._fixed_species,
        )
        layout.addWidget(self._metadata_editor)
        self._metadata_warning = QLabel(
            "Initial concentrations (y0) are required. "
            "Validate the mechanism to enable y0/fixed fields. "
            "Missing values will be requested when you finish.",
            widget,
        )
        self._metadata_warning.setStyleSheet("color: #B00020;")
        self._metadata_warning.setWordWrap(True)
        self._metadata_warning.setVisible(False)
        layout.addWidget(self._metadata_warning)
        return widget

    def _on_technique_changed(self) -> None:
        if not self._ui_ready:
            return
        self._apply_technique_ui()
        self._reload_data()
        self._update_nav()

    def _apply_technique_ui(self) -> None:
        technique = self._current_technique()
        is_spec = technique in {"spec_full", "spec_channels"}
        self._sheet_label.setText("Spectra sheet" if is_spec else "Integrals sheet")

        show_conc = technique == "spec_full"
        self._conc_sheet_label.setVisible(show_conc)
        self._conc_sheet_combo.setVisible(show_conc)

        show_shift = technique == "nmr_full"
        self._shift_sheet_label.setVisible(show_shift)
        self._shift_sheet_combo.setVisible(show_shift)

        if technique == "nmr_full":
            self._channels_stack.setCurrentIndex(1)
        else:
            self._channels_stack.setCurrentIndex(0)

        show_efa = technique == "spec_full"
        self._efa_check.setVisible(show_efa)
        self._efa_spin.setVisible(show_efa)
        if not show_efa:
            self._efa_check.setChecked(False)
            self._efa_spin.setValue(0)

        if self._channel_spec_widget is not None:
            self._channel_spec_widget.update_parent_options(list(self._dynamic_species))

    # ---- Navigation ----
    def _on_back(self) -> None:
        self._stack.setCurrentIndex(max(0, self._stack.currentIndex() - 1))
        self._update_nav()

    def _on_next(self) -> None:
        self._stack.setCurrentIndex(min(self._stack.count() - 1, self._stack.currentIndex() + 1))
        self._update_nav()

    def _on_finish(self) -> None:
        try:
            self._build_datasets()
        except Exception as exc:
            QMessageBox.warning(self, "Import error", str(exc))
            return
        if not self._datasets:
            QMessageBox.warning(self, "Import error", "No datasets imported.")
            return
        if any(
            not self._metadata_editor.is_complete(dataset)
            for dataset in self._datasets
        ):
            template = copy.deepcopy(self._datasets[0])
            dialog = DatasetEditorDialog(
                template,
                dynamic_species=self._dynamic_species,
                fixed_species=self._fixed_species,
                parent=self,
            )
            dialog.setWindowTitle("Complete Metadata")
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            updated = dialog.dataset
            for dataset in self._datasets:
                dataset.temperature = updated.temperature
                dataset.time_unit = updated.time_unit
                dataset.x_unit = updated.x_unit
                dataset.signal_unit = updated.signal_unit
                dataset.y0 = dict(updated.y0) if updated.y0 is not None else None
                dataset.fixed_conc = (
                    dict(updated.fixed_conc) if updated.fixed_conc is not None else None
                )
        self.accept()

    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        is_last = idx == self._stack.count() - 1
        self._btn_back.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < self._stack.count() - 1)
        if is_last:
            has_dataset = self._dataset is not None
            complete = self._metadata_complete() if has_dataset else False
            self._btn_finish.setEnabled(has_dataset)
            self._metadata_warning.setVisible(has_dataset and not complete)
        else:
            self._btn_finish.setEnabled(False)
            self._metadata_warning.setVisible(False)

    def _metadata_complete(self) -> bool:
        if self._dataset is None:
            return False
        self._metadata_editor.apply_to_dataset(self._dataset)
        if not self._dynamic_species and not self._fixed_species:
            return False
        return self._metadata_editor.is_complete(self._dataset)

    # ---- Data loading ----
    def _on_browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv *.tsv *.txt *.xlsx *.xls);;All Files (*)",
        )
        if not paths:
            return
        self._paths = list(paths)
        display = paths[0] if len(paths) == 1 else f"{len(paths)} files selected"
        self._path_edit.setText(display)
        suffix = Path(paths[0]).suffix.lower()
        if suffix == ".tsv":
            self._delimiter_edit.setText("\\t")
        elif suffix in {".csv", ".txt"}:
            self._delimiter_edit.setText("")
        self._populate_sheets(paths[0])
        self._reload_data()

    def _populate_sheets(self, path: str) -> None:
        self._sheet_combo.clear()
        self._conc_sheet_combo.clear()
        self._shift_sheet_combo.clear()
        if Path(path).suffix.lower() in {".xlsx", ".xls"}:
            import pandas as pd

            xls = pd.ExcelFile(path)
            names = [str(name) for name in xls.sheet_names]
            self._sheet_combo.addItems(names)
            self._conc_sheet_combo.addItems([""] + names)
            self._shift_sheet_combo.addItems([""] + names)
            self._sheet_combo.setEnabled(True)
            self._conc_sheet_combo.setEnabled(True)
            self._shift_sheet_combo.setEnabled(True)
        else:
            self._sheet_combo.setEnabled(False)
            self._conc_sheet_combo.setEnabled(False)
            self._shift_sheet_combo.setEnabled(False)

    def _reload_data(self) -> None:
        path = self._current_path()
        if not path:
            return

        try:
            t, D, x, labels = self._load_data(path)
            if self._current_technique() == "nmr_full":
                shift_sheet = self._shift_sheet_combo.currentText().strip()
                shift_axis = self._load_shift_axis(path, shift_sheet, len(labels))
                if shift_axis is not None:
                    x = shift_axis
                    self._raw_shift_x = shift_axis
        except Exception as exc:
            self._preview.setRowCount(0)
            self._preview.setColumnCount(0)
            QMessageBox.warning(self, "Load error", str(exc))
            return

        self._raw_t = t
        self._raw_D = D
        self._raw_x = x
        self._raw_labels = labels

        self._update_preview(t, D, labels)
        self._populate_channels(labels)
        self._time_col_spin.setMaximum(max(len(labels), 1))
        self._initialize_metadata(path, t, D, x, labels)
        self._update_nav()

    def _current_path(self) -> str:
        if self._paths:
            return self._paths[0]
        return self._path_edit.text().strip()

    def _load_data(self, path: str):
        settings = self._loader_settings(path)
        if settings["kind"] == "xlsx":
            return load_xlsx(
                path,
                sheet=settings["sheet"],
                transpose=settings["transpose"],
                time_col=settings["time_col"],
                header=settings["header"],
            )
        return load_matrix_file(
            path,
            delimiter=settings["delimiter"],
            transpose=settings["transpose"],
            time_col=settings["time_col"],
            header=settings["header"],
        )

    def _load_shift_axis(self, path: str, sheet: str, expected: int) -> np.ndarray | None:
        if not sheet:
            return None
        if Path(path).suffix.lower() not in {".xlsx", ".xls"}:
            return None
        try:
            import pandas as pd
        except Exception:
            return None
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=0)
        except Exception:
            return None
        if expected <= 0:
            return None
        labels = [str(c) for c in df.columns]
        axis = _parse_axis(labels)
        if axis is not None and axis.size == expected:
            return axis
        col = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        if col.size == expected:
            return np.asarray(col.to_numpy(), dtype=float)
        return None

    def _update_preview(self, t, D, labels) -> None:
        max_rows = min(50, D.shape[0])
        max_cols = min(10, D.shape[1])

        self._preview.setRowCount(max_rows)
        self._preview.setColumnCount(max_cols + 1)
        headers = ["t"] + labels[:max_cols]
        self._preview.setHorizontalHeaderLabels(headers)

        for row in range(max_rows):
            self._preview.setItem(row, 0, QTableWidgetItem(f"{t[row]:.6g}"))
            for col in range(max_cols):
                self._preview.setItem(
                    row, col + 1, QTableWidgetItem(f"{D[row, col]:.6g}")
                )

    def _populate_channels(self, labels: Sequence[str]) -> None:
        technique = self._current_technique()
        if technique == "nmr_full":
            self._channel_spec_widget.set_channels([str(label) for label in labels])
            self._channel_spec_widget.update_parent_options(list(self._dynamic_species))
            return

        self._channels_list.blockSignals(True)
        self._channels_list.clear()
        for label in labels:
            item = QListWidgetItem(str(label))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._channels_list.addItem(item)
        self._channels_list.blockSignals(False)

    def _initialize_metadata(
        self,
        path: str,
        t,
        D,
        x,
        labels,
    ) -> None:
        technique = self._current_technique()
        name = Path(path).stem
        time_unit, x_unit, signal_unit = _default_units(technique)
        settings = self._loader_settings(path)
        dataset = KineticsFitDataset(
            t=t,
            D=D,
            x=x,
            channel_labels=list(labels),
            technique=technique,
            time_unit=time_unit,
            x_unit=x_unit,
            signal_unit=signal_unit,
            name=name,
            source_path=str(path),
            y0=None,
            loader_kind=settings["kind"],
            loader_delimiter=settings["delimiter"],
            loader_header=settings["header"],
            loader_transpose=settings["transpose"],
            loader_time_col=settings["time_col"],
            loader_sheet=settings["sheet"],
            loader_conc_sheet=self._conc_sheet_combo.currentText().strip() or None,
            loader_shift_sheet=self._shift_sheet_combo.currentText().strip() or None,
            channel_indices=list(range(len(labels))),
            efa_enabled=bool(self._efa_check.isChecked()),
            efa_eigen=int(self._efa_spin.value()),
        )
        self._dataset = dataset
        self._metadata_editor.set_dataset(dataset)

    # ---- Channels ----
    def _set_all_channels(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._channels_list.blockSignals(True)
        for idx in range(self._channels_list.count()):
            self._channels_list.item(idx).setCheckState(state)
        self._channels_list.blockSignals(False)

    def _apply_range(self) -> None:
        if self._raw_x is None:
            return
        try:
            min_val = float(self._range_min.text())
            max_val = float(self._range_max.text())
        except ValueError:
            QMessageBox.warning(self, "Range error", "Enter numeric min/max values.")
            return
        self._channels_list.blockSignals(True)
        for idx, x_val in enumerate(self._raw_x):
            item = self._channels_list.item(idx)
            if item is None:
                continue
            if min_val <= x_val <= max_val:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        self._channels_list.blockSignals(False)

    def _on_channel_item_changed(self, _item: QListWidgetItem) -> None:
        pass

    # ---- Final dataset ----
    def _build_datasets(self) -> None:
        path_list = self._paths or [self._path_edit.text().strip()]
        path_list = [path for path in path_list if path]
        if not path_list:
            raise ValueError("No data file selected.")
        indices = self._selected_channel_indices()
        if not indices:
            raise ValueError("Select at least one channel.")
        parents = list(self._selected_channel_parents) if self._selected_channel_parents else None
        datasets: list[KineticsFitDataset] = []
        for path in path_list:
            dataset = self._build_dataset_for_path(
                path,
                indices,
                parents=parents,
                multi=len(path_list) > 1,
            )
            datasets.append(dataset)
        self._datasets = datasets

    def _build_dataset_for_path(
        self,
        path: str,
        indices: list[int],
        *,
        parents: list[str] | None = None,
        multi: bool,
    ) -> KineticsFitDataset:
        t, D, x, labels = self._load_data(path)
        if self._current_technique() == "nmr_full":
            shift_sheet = self._shift_sheet_combo.currentText().strip()
            shift_axis = self._load_shift_axis(path, shift_sheet, len(labels))
            if shift_axis is not None:
                x = shift_axis
        if any(idx >= len(labels) or idx < 0 for idx in indices):
            raise ValueError(f"Channel indices out of range for {path}.")
        D_sel = D[:, indices]
        labels_sel = [labels[idx] for idx in indices]
        x_sel = x[indices] if x is not None else None
        technique = self._current_technique()
        name = Path(path).stem
        time_unit, x_unit, signal_unit = _default_units(technique)
        settings = self._loader_settings(path)
        dataset = KineticsFitDataset(
            t=t,
            D=D_sel,
            x=x_sel,
            channel_labels=list(labels_sel),
            technique=technique,
            time_unit=time_unit,
            x_unit=x_unit,
            signal_unit=signal_unit,
            name=name,
            source_path=str(path),
            y0=None,
            loader_kind=settings["kind"],
            loader_delimiter=settings["delimiter"],
            loader_header=settings["header"],
            loader_transpose=settings["transpose"],
            loader_time_col=settings["time_col"],
            loader_sheet=settings["sheet"],
            loader_conc_sheet=self._conc_sheet_combo.currentText().strip() or None,
            loader_shift_sheet=self._shift_sheet_combo.currentText().strip() or None,
            channel_indices=list(indices),
            efa_enabled=bool(self._efa_check.isChecked()),
            efa_eigen=int(self._efa_spin.value()),
        )
        if parents:
            dataset.channel_parents = list(parents)
        self._metadata_editor.apply_to_dataset(dataset)
        if multi:
            dataset.name = name
        return dataset

    def _selected_channel_indices(self) -> list[int]:
        technique = self._current_technique()
        if technique == "nmr_full":
            selected = self._channel_spec_widget.get_selected_channels()
            label_to_index = {str(label): i for i, label in enumerate(self._raw_labels)}
            indices: list[int] = []
            parents: list[str] = []
            for entry in selected:
                name = str(entry.get("col_name") or "")
                if name not in label_to_index:
                    continue
                indices.append(label_to_index[name])
                parents.append(str(entry.get("parent") or ""))
            self._selected_channel_parents = parents
            return indices

        indices = []
        for idx in range(self._channels_list.count()):
            item = self._channels_list.item(idx)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                indices.append(idx)
        self._selected_channel_parents = []
        return indices

    def _current_technique(self) -> TechniqueType:
        for technique, button in self._technique_buttons.items():
            if button.isChecked():
                return technique
        return "spec_channels"

    def _loader_settings(self, path: str) -> dict[str, object]:
        suffix = Path(path).suffix.lower()
        header = self._header_check.isChecked()
        transpose = self._transpose_check.isChecked()
        time_col = self._time_col_spin.value()
        delimiter = self._delimiter_edit.text()
        if delimiter == "\\t":
            delimiter = "\t"
        if not delimiter:
            delimiter = None
        if suffix in {".xlsx", ".xls"}:
            sheet = self._sheet_combo.currentText() or "data"
            kind = "xlsx"
        else:
            sheet = None
            kind = "matrix"
        return {
            "kind": kind,
            "delimiter": delimiter,
            "transpose": transpose,
            "time_col": time_col,
            "header": header,
            "sheet": sheet,
        }


def _default_units(technique: TechniqueType) -> tuple[str, str, str]:
    if technique in {"spec_full", "spec_channels"}:
        return "s", "nm", "a.u."
    return "s", "ppm", "a.u."


def _parse_axis(labels: Sequence[str]) -> np.ndarray | None:
    axis: list[float] = []
    for label in labels:
        try:
            axis.append(float(label))
        except ValueError:
            return None
    return np.asarray(axis, dtype=float)
