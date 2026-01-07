from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from hmfit.kinetics.gui.main import KineticsMainWidget


class KineticsTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(KineticsMainWidget(self))
