from __future__ import annotations

import time
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
        try:
            throttle_ms = float(self._config.get("_progress_throttle_ms", 80.0))
        except (TypeError, ValueError):
            throttle_ms = 80.0
        self._progress_interval_s = max(0.0, throttle_ms / 1000.0)
        try:
            batch_size = int(self._config.get("_progress_batch_size", 20))
        except (TypeError, ValueError):
            batch_size = 20
        self._progress_batch_size = max(1, batch_size)
        self._progress_buffer: list[str] = []
        self._last_progress_emit = 0.0
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

    @staticmethod
    def _is_critical_message(msg: str) -> bool:
        text = str(msg or "").strip().lower()
        if not text:
            return False
        critical_tokens = ("error", "cancel", "cancelled", "critical", "traceback", "failed")
        return any(tok in text for tok in critical_tokens)

    def _flush_progress(self, *, force: bool = False) -> None:
        if not self._progress_buffer:
            return
        now = time.monotonic()
        if (not force) and (self._progress_interval_s > 0.0) and ((now - self._last_progress_emit) < self._progress_interval_s):
            return
        payload = "\n".join(self._progress_buffer)
        self._progress_buffer.clear()
        self._last_progress_emit = now
        self.progress.emit(payload)

    def _emit_progress(self, msg: str, *, immediate: bool = False) -> None:
        text = str(msg)
        if immediate or self._is_critical_message(text):
            self._flush_progress(force=True)
            self.progress.emit(text)
            self._last_progress_emit = time.monotonic()
            return
        self._progress_buffer.append(text)
        if len(self._progress_buffer) >= self._progress_batch_size:
            self._flush_progress(force=True)
            return
        self._flush_progress(force=False)

    @Slot()
    def run(self) -> None:
        try:
            from hmfit_core.api import FitCancelled
        except Exception:  # pragma: no cover - optional at runtime
            FitCancelled = None

        try:
            def _progress(msg: str) -> None:
                self._emit_progress(str(msg), immediate=False)

            res = self._run_fn(self._config, progress_cb=_progress, cancel=self._is_cancelled)
            if isinstance(res, dict) and res.get("error"):
                raise RuntimeError(str(res.get("error")))
            if isinstance(res, dict) and not res.get("success", True):
                raise RuntimeError("Fit failed.")
            self._flush_progress(force=True)
            self.result.emit(res)
        except Exception as exc:
            self._flush_progress(force=True)
            if FitCancelled is not None and isinstance(exc, FitCancelled):
                self._emit_progress(str(exc) or "Fit cancelled.", immediate=True)
            else:
                self.error.emit(str(exc))
        finally:
            self._flush_progress(force=True)
            self.finished.emit()
