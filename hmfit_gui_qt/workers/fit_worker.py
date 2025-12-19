from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot


class FitWorker(QObject):
    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, run_fn: Callable[..., Any], *, config: dict[str, Any], parent: QObject | None = None) -> None:
        super().__init__(None)
        self._run_fn = run_fn
        self._config = dict(config)
        self._cancel_requested = False
        self._thread = QThread(parent)
        self.moveToThread(self._thread)

        self._thread.started.connect(self.run)
        self.finished.connect(self._thread.quit)
        self.finished.connect(self.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    def start(self) -> None:
        self._thread.start()

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    @Slot()
    def run(self) -> None:
        try:
            from hmfit_core.api import FitCancelled
        except Exception:  # pragma: no cover - optional at runtime
            FitCancelled = None

        try:
            def _progress(msg: str) -> None:
                self.progress.emit(str(msg))

            res = self._run_fn(self._config, progress_cb=_progress, cancel=self._is_cancelled)
            if isinstance(res, dict) and res.get("error"):
                raise RuntimeError(str(res.get("error")))
            if isinstance(res, dict) and not res.get("success", True):
                raise RuntimeError("Fit failed.")
            self.result.emit(res)
        except Exception as exc:
            if FitCancelled is not None and isinstance(exc, FitCancelled):
                self.progress.emit(str(exc) or "Fit cancelled.")
            else:
                self.error.emit(str(exc))
        finally:
            self.finished.emit()
