from __future__ import annotations

from PySide6 import QtGui
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class LogConsole(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._autoscroll = True

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(10)
        self.setFont(font)
        self.setStyleSheet(
            """
QPlainTextEdit {
  background: #0f111a;
  color: #e6e6e6;
  border: 1px solid #333;
}
"""
        )

    def set_autoscroll(self, enabled: bool) -> None:
        self._autoscroll = bool(enabled)

    def append_text(self, text: str) -> None:
        self.appendPlainText(str(text))
        if self._autoscroll:
            self.moveCursor(QTextCursor.MoveOperation.End)
            self.ensureCursorVisible()
