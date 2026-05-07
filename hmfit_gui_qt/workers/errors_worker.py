from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class ErrorsWorker(QThread):
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, run_fn: Callable[..., Any], *, payload: dict[str, Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._run_fn = run_fn
        self._payload = dict(payload)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.requestInterruption()

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested or self.isInterruptionRequested())

    def run(self) -> None:
        try:
            from hmfit_core.utils.errors import BootstrapCancelled
        except Exception:  # pragma: no cover - optional at runtime
            BootstrapCancelled = None

        try:
            def _progress(payload: object) -> None:
                self.progress.emit(payload)

            res = self._run_fn(self._payload, progress_cb=_progress, cancel_cb=self._is_cancelled)
            self.result.emit(res)
        except Exception as exc:
            if BootstrapCancelled is not None and isinstance(exc, BootstrapCancelled):
                self.cancelled.emit()
            else:
                self.error.emit(str(exc))

    def thread(self) -> QThread:
        return self
