from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from hmfit_core.utils.graph_gui_parser import parse_multiline_equilibria


class _ParserWorkerSignals(QtCore.QObject):
    parsed_finished = QtCore.Signal(int, object, dict)
    parsed_error = QtCore.Signal(int, str)


class _ParserWorker(QtCore.QRunnable):
    def __init__(self, seq: int, text: str, graph_ref: object | None = None) -> None:
        super().__init__()
        self.seq = int(seq)
        self.text = str(text or "")
        self.graph_ref = graph_ref
        self.signals = _ParserWorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            graph, solver_inputs = parse_multiline_equilibria(self.text)
            # Fuerza el cálculo de rutas globales en el hilo worker.
            graph.resolve_global_pathways()
            self.signals.parsed_finished.emit(self.seq, graph, solver_inputs)
        except Exception as exc:
            self.signals.parsed_error.emit(self.seq, str(exc))


class EquationEditorWidget(QtWidgets.QWidget):
    model_parsed = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_graph: Any = None
        self._parse_seq = 0
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._active_workers: dict[int, _ParserWorker] = {}
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
        self._parse_seq += 1
        seq = self._parse_seq

        self.lbl_status.setText("Analizando modelo…")
        self.lbl_status.setStyleSheet("color: #1c71d8;")

        worker = _ParserWorker(seq=seq, text=text, graph_ref=self._last_graph)
        worker.signals.parsed_finished.connect(
            self._on_worker_parsed_finished, QtCore.Qt.ConnectionType.QueuedConnection
        )
        worker.signals.parsed_error.connect(
            self._on_worker_parsed_error, QtCore.Qt.ConnectionType.QueuedConnection
        )
        self._active_workers[seq] = worker
        self._thread_pool.start(worker)

    @QtCore.Slot(int, object, dict)
    def _on_worker_parsed_finished(self, seq: int, graph: object, solver_inputs: dict) -> None:
        self._active_workers.pop(int(seq), None)
        if int(seq) != int(self._parse_seq):
            return
        self._last_graph = graph
        self.lbl_status.setText("Modelo valido.")
        self.lbl_status.setStyleSheet("color: #26a269;")
        self.model_parsed.emit(dict(solver_inputs))

    @QtCore.Slot(int, str)
    def _on_worker_parsed_error(self, seq: int, message: str) -> None:
        self._active_workers.pop(int(seq), None)
        if int(seq) != int(self._parse_seq):
            return
        self.lbl_status.setText(str(message))
        self.lbl_status.setStyleSheet("color: #c01c28;")

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)

    def text(self) -> str:
        return self.editor.toPlainText()

    def get_text(self) -> str:
        return self.editor.toPlainText().strip()

    def clear(self) -> None:
        self.editor.clear()
