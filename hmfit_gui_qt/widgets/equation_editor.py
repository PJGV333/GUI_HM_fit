from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from hmfit_core.utils.graph_gui_parser import parse_multiline_equilibria


class EquationEditorWidget(QtWidgets.QWidget):
    model_parsed = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_graph = None
        self._build_ui()
        self._build_debounce_timer()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setPlaceholderText(
            "Ejemplo:\n"
            "H + G <=> HG ; 4.5\n"
            "H + G <=> HG ; logB=4.5"
        )
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.editor.setFont(font)
        layout.addWidget(self.editor, 1)

        self.lbl_status = QtWidgets.QLabel("Escribe ecuaciones para validar el modelo.", self)
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    def _build_debounce_timer(self) -> None:
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(500)
        self._debounce_timer.timeout.connect(self._validate_and_parse)
        self.editor.textChanged.connect(self._on_text_changed)

    @QtCore.Slot()
    def _on_text_changed(self) -> None:
        self._debounce_timer.start()

    @QtCore.Slot()
    def _validate_and_parse(self) -> None:
        text = self.editor.toPlainText()
        try:
            graph, solver_inputs = parse_multiline_equilibria(text)
            self._last_graph = graph
            self.lbl_status.setText("Modelo valido.")
            self.lbl_status.setStyleSheet("color: #26a269;")
            self.model_parsed.emit(solver_inputs)
        except Exception as exc:
            self.lbl_status.setText(str(exc))
            self.lbl_status.setStyleSheet("color: #c01c28;")

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(str(text or ""))

    def text(self) -> str:
        return self.editor.toPlainText()
