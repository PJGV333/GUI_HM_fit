"""Import wizard for kinetics datasets."""

from __future__ import annotations

import copy
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

    def _build_step_type(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        group = QGroupBox("Select data type", widget)
        group_layout = QVBoxLayout(group)
        self._technique_group = QButtonGroup(group)

        self._technique_buttons: dict[TechniqueType, QRadioButton] = {}
        options = [
            ("spec_full", "Spectroscopy: full spectrum (time × λ)"),
            ("spec_channels", "Spectroscopy: few channels"),
            ("nmr_integrals", "NMR: integrals (time × peaks)"),
            ("nmr_full", "NMR: full spectrum (time × ppm)"),
        ]
        for technique, label in options:
            button = QRadioButton(label, group)
            self._technique_group.addButton(button)
            self._technique_buttons[technique] = button
            group_layout.addWidget(button)
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
        btn_browse = QPushButton("Browse…", widget)
        btn_browse.clicked.connect(self._on_browse)
        file_row.addWidget(btn_browse)
        layout.addLayout(file_row)

        options_layout = QFormLayout()
        self._sheet_combo = QComboBox(widget)
        self._sheet_combo.currentIndexChanged.connect(self._reload_data)
        options_layout.addRow("Sheet", self._sheet_combo)

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

        self._channels_list = QListWidget(widget)
        self._channels_list.itemChanged.connect(self._on_channel_item_changed)
        layout.addWidget(self._channels_list, 1)

        actions_layout = QHBoxLayout()
        btn_all = QPushButton("All", widget)
        btn_all.clicked.connect(lambda: self._set_all_channels(True))
        actions_layout.addWidget(btn_all)
        btn_none = QPushButton("None", widget)
        btn_none.clicked.connect(lambda: self._set_all_channels(False))
        actions_layout.addWidget(btn_none)
        actions_layout.addStretch(1)

        self._range_min = QLineEdit(widget)
        self._range_min.setPlaceholderText("min")
        self._range_max = QLineEdit(widget)
        self._range_max.setPlaceholderText("max")
        btn_range = QPushButton("Apply range", widget)
        btn_range.clicked.connect(self._apply_range)
        actions_layout.addWidget(self._range_min)
        actions_layout.addWidget(self._range_max)
        actions_layout.addWidget(btn_range)

        layout.addLayout(actions_layout)
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
            "Complete initial concentrations before finishing.", widget
        )
        self._metadata_warning.setStyleSheet("color: #B00020;")
        self._metadata_warning.setWordWrap(True)
        self._metadata_warning.setVisible(False)
        layout.addWidget(self._metadata_warning)
        return widget

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
            complete = self._metadata_complete()
            self._btn_finish.setEnabled(complete)
            self._metadata_warning.setVisible(not complete)
        else:
            self._btn_finish.setEnabled(False)
            self._metadata_warning.setVisible(False)

    def _metadata_complete(self) -> bool:
        if self._dataset is None:
            return True
        self._metadata_editor.apply_to_dataset(self._dataset)
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
        if Path(path).suffix.lower() in {".xlsx", ".xls"}:
            import pandas as pd

            xls = pd.ExcelFile(path)
            self._sheet_combo.addItems([str(name) for name in xls.sheet_names])
            self._sheet_combo.setEnabled(True)
        else:
            self._sheet_combo.setEnabled(False)

    def _reload_data(self) -> None:
        path = self._current_path()
        if not path:
            return

        try:
            t, D, x, labels = self._load_data(path)
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
        self._time_col_spin.setMaximum(len(labels))
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
            channel_indices=list(range(len(labels))),
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
        datasets: list[KineticsFitDataset] = []
        for path in path_list:
            dataset = self._build_dataset_for_path(path, indices, multi=len(path_list) > 1)
            datasets.append(dataset)
        self._datasets = datasets

    def _build_dataset_for_path(
        self, path: str, indices: list[int], *, multi: bool
    ) -> KineticsFitDataset:
        t, D, x, labels = self._load_data(path)
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
            channel_indices=list(indices),
        )
        self._metadata_editor.apply_to_dataset(dataset)
        if multi:
            dataset.name = name
        return dataset

    def _selected_channel_indices(self) -> list[int]:
        indices = []
        for idx in range(self._channels_list.count()):
            item = self._channels_list.item(idx)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                indices.append(idx)
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
