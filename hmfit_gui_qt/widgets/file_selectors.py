from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class FileSelector(QWidget):
    path_changed = Signal(str)

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label, self)
        layout.addWidget(self._label)

        self._edit = QLineEdit(self)
        self._edit.textChanged.connect(self.path_changed.emit)
        layout.addWidget(self._edit, 1)

        btn = QPushButton("Browse…", self)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

    def path(self) -> str:
        return self._edit.text().strip()

    def set_path(self, path: str) -> None:
        self._edit.setText(str(path or ""))

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel file", "", "Excel (*.xlsx)")
        if not path:
            return
        self.set_path(path)
