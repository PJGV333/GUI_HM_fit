"""Dataset editor widgets for kinetics GUI."""

from __future__ import annotations

from typing import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..data.fit_dataset import KineticsFitDataset


class DatasetEditorWidget(QWidget):
    """Editor for dataset metadata and initial conditions."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        dynamic_species: Sequence[str] | None = None,
        fixed_species: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._dynamic_species = list(dynamic_species or [])
        self._fixed_species = list(fixed_species or [])
        self._y0_fields: dict[str, QDoubleSpinBox] = {}
        self._fixed_fields: dict[str, QDoubleSpinBox] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._form = QFormLayout()
        layout.addLayout(self._form)

        self.name_edit = QLineEdit(self)
        self._form.addRow("Name", self.name_edit)

        self.temperature_spin = QDoubleSpinBox(self)
        self.temperature_spin.setDecimals(6)
        self.temperature_spin.setRange(-1e6, 1e6)
        self.temperature_spin.setValue(298.15)
        self._form.addRow("Temperature", self.temperature_spin)

        self.time_unit_edit = QLineEdit(self)
        self.time_unit_edit.setText("s")
        self.time_unit_edit.setToolTip("Time unit: typically seconds (s).")
        self._form.addRow("Time unit", self.time_unit_edit)

        self.x_unit_edit = QLineEdit(self)
        self.x_unit_edit.setToolTip("X unit: wavelength (nm) or ppm for NMR.")
        self._form.addRow("X unit", self.x_unit_edit)

        self.signal_unit_edit = QLineEdit(self)
        self.signal_unit_edit.setToolTip("Signal unit: absorbance, intensity, or a.u.")
        self._form.addRow("Signal unit", self.signal_unit_edit)

        self._y0_label = QLabel("y0 (dynamic species)", self)
        self._y0_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._y0_label.setToolTip(
            "Initial concentrations (y0) are required for all dynamic species."
        )
        self._form.addRow(self._y0_label)

        self._fixed_label = QLabel("fixed_conc (fixed species)", self)
        self._fixed_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._fixed_label.setToolTip(
            "Fixed concentrations are required when the mechanism defines fixed species."
        )
        self._form.addRow(self._fixed_label)

        self._empty_species_label = QLabel(
            "Load a mechanism to edit y0/fixed species.", self
        )
        self._form.addRow(self._empty_species_label)

        self._rebuild_species_fields()

    def set_species(
        self, dynamic_species: Sequence[str], fixed_species: Sequence[str]
    ) -> None:
        self._dynamic_species = list(dynamic_species)
        self._fixed_species = list(fixed_species)
        self._rebuild_species_fields()

    def set_dataset(self, dataset: KineticsFitDataset) -> None:
        self.name_edit.setText(dataset.name)
        self.temperature_spin.setValue(float(dataset.temperature))
        self.time_unit_edit.setText(dataset.time_unit)
        self.x_unit_edit.setText(dataset.x_unit)
        self.signal_unit_edit.setText(dataset.signal_unit)

        y0 = dataset.y0 or {}
        for species, field in self._y0_fields.items():
            field.setValue(float(y0.get(species, 0.0)))

        fixed = dataset.fixed_conc or {}
        for species, field in self._fixed_fields.items():
            field.setValue(float(fixed.get(species, 0.0)))

    def apply_to_dataset(self, dataset: KineticsFitDataset) -> None:
        dataset.name = self.name_edit.text().strip() or dataset.name
        dataset.temperature = float(self.temperature_spin.value())
        dataset.time_unit = self.time_unit_edit.text().strip()
        dataset.x_unit = self.x_unit_edit.text().strip()
        dataset.signal_unit = self.signal_unit_edit.text().strip()

        if self._dynamic_species:
            dataset.y0 = {
                species: float(field.value())
                for species, field in self._y0_fields.items()
            }
        else:
            dataset.y0 = dataset.y0

        if self._fixed_species:
            dataset.fixed_conc = {
                species: float(field.value())
                for species, field in self._fixed_fields.items()
            }
        else:
            dataset.fixed_conc = dataset.fixed_conc

    def is_complete(self, dataset: KineticsFitDataset) -> bool:
        if self._dynamic_species:
            if dataset.y0 is None:
                return False
            if any(species not in dataset.y0 for species in self._dynamic_species):
                return False
        if self._fixed_species:
            if dataset.fixed_conc is None:
                return False
            if any(species not in dataset.fixed_conc for species in self._fixed_species):
                return False
        return True

    def _rebuild_species_fields(self) -> None:
        for field in list(self._y0_fields.values()) + list(self._fixed_fields.values()):
            self._form.removeRow(field)
            field.deleteLater()
        self._y0_fields.clear()
        self._fixed_fields.clear()

        if not self._dynamic_species and not self._fixed_species:
            self._empty_species_label.setVisible(True)
            return

        self._empty_species_label.setVisible(False)

        if self._dynamic_species:
            for species in self._dynamic_species:
                spin = _make_spinbox(self)
                self._y0_fields[species] = spin
                self._form.addRow(f"y0[{species}]", spin)
        if self._fixed_species:
            for species in self._fixed_species:
                spin = _make_spinbox(self)
                self._fixed_fields[species] = spin
                self._form.addRow(f"fixed[{species}]", spin)


class DatasetEditorDialog(QDialog):
    """Dialog wrapper for dataset metadata editing."""

    def __init__(
        self,
        dataset: KineticsFitDataset,
        *,
        dynamic_species: Sequence[str] | None = None,
        fixed_species: Sequence[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Dataset")
        self._dataset = dataset
        self.editor = DatasetEditorWidget(
            self,
            dynamic_species=dynamic_species,
            fixed_species=fixed_species,
        )
        self.editor.set_dataset(dataset)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        self.editor.apply_to_dataset(self._dataset)
        super().accept()

    @property
    def dataset(self) -> KineticsFitDataset:
        return self._dataset


def _make_spinbox(parent: QWidget) -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setDecimals(6)
    spin.setRange(-1e9, 1e9)
    spin.setValue(0.0)
    return spin
