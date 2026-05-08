from __future__ import annotations

from typing import Any, Callable

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hmfit_gui_qt.workers.errors_worker import ErrorsWorker


class ErrorAnalysisWidget(QWidget):
    output_ready = Signal(object)

    def __init__(
        self,
        compute_payload_fn: Callable[..., dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._compute_payload_fn = compute_payload_fn
        self._errors_context: dict[str, Any] | None = None
        self._errors_last_output: dict[str, Any] | None = None
        self._errors_worker: ErrorsWorker | None = None
        self._results_ci_columns: tuple[int | None, int | None] = (None, None)
        self._build_ui()
        self._refresh_idle_state()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls_grid = QGridLayout()
        controls_grid.setColumnStretch(1, 1)
        controls_grid.setColumnStretch(3, 1)
        controls_grid.setColumnStretch(5, 1)
        controls_grid.setColumnStretch(7, 1)

        controls_grid.addWidget(QLabel("Error method"), 0, 0)
        self.combo_error_method = QComboBox(self)
        self.combo_error_method.addItem("Analytical covariance", "analytic")
        self.combo_error_method.addItem("Bootstrap (linearized + wild)", "bootstrap_linear")
        self.combo_error_method.addItem("Bootstrap (one-step LM)", "bootstrap_onestep")
        self.combo_error_method.addItem("Bootstrap (full refit, audit)", "bootstrap_full_refit_audit")
        self.combo_error_method.currentIndexChanged.connect(self._on_error_method_changed)
        controls_grid.addWidget(self.combo_error_method, 0, 1, 1, 7)

        controls_grid.addWidget(QLabel("B (replicates)"), 1, 0)
        self.spin_error_b = QSpinBox(self)
        self.spin_error_b.setRange(50, 5000)
        self.spin_error_b.setValue(500)
        controls_grid.addWidget(self.spin_error_b, 1, 1)

        controls_grid.addWidget(QLabel("Seed"), 1, 2)
        self.edit_error_seed = QLineEdit(self)
        self.edit_error_seed.setPlaceholderText("optional")
        controls_grid.addWidget(self.edit_error_seed, 1, 3)

        controls_grid.addWidget(QLabel("Wild type"), 1, 4)
        self.combo_error_wild = QComboBox(self)
        self.combo_error_wild.addItem("Rademacher (+/-1)", "rademacher")
        self.combo_error_wild.addItem("Mammen", "mammen")
        controls_grid.addWidget(self.combo_error_wild, 1, 5)

        controls_grid.addWidget(QLabel("LM lambda"), 1, 6)
        self.spin_error_lambda = QDoubleSpinBox(self)
        self.spin_error_lambda.setDecimals(6)
        self.spin_error_lambda.setRange(0.0, 1e6)
        self.spin_error_lambda.setSingleStep(1e-3)
        self.spin_error_lambda.setValue(1e-3)
        controls_grid.addWidget(self.spin_error_lambda, 1, 7)

        self.lbl_error_max_iter = QLabel("Max iter", self)
        controls_grid.addWidget(self.lbl_error_max_iter, 2, 0)
        self.spin_error_max_iter = QSpinBox(self)
        self.spin_error_max_iter.setRange(5, 200)
        self.spin_error_max_iter.setValue(30)
        controls_grid.addWidget(self.spin_error_max_iter, 2, 1)

        self.lbl_error_tol = QLabel("Tol", self)
        controls_grid.addWidget(self.lbl_error_tol, 2, 2)
        self.spin_error_tol = QDoubleSpinBox(self)
        self.spin_error_tol.setDecimals(12)
        self.spin_error_tol.setRange(1e-12, 1e-2)
        self.spin_error_tol.setSingleStep(1e-8)
        self.spin_error_tol.setValue(1e-8)
        controls_grid.addWidget(self.spin_error_tol, 2, 3)

        self.lbl_error_fail_policy = QLabel("Fail policy", self)
        controls_grid.addWidget(self.lbl_error_fail_policy, 2, 4)
        self.combo_error_fail_policy = QComboBox(self)
        self.combo_error_fail_policy.addItem("Skip failed replicates", "skip")
        self.combo_error_fail_policy.addItem("Stop on first failure", "stop")
        controls_grid.addWidget(self.combo_error_fail_policy, 2, 5, 1, 3)

        layout.addLayout(controls_grid)

        self.lbl_error_audit_warning = QLabel(
            "Warning: Bootstrap (full refit, audit) can take several minutes. "
            "Use Cancel to stop the computation if needed.",
            self,
        )
        self.lbl_error_audit_warning.setWordWrap(True)
        self.lbl_error_audit_warning.setVisible(False)
        layout.addWidget(self.lbl_error_audit_warning)

        flags_row = QHBoxLayout()
        self.chk_error_ci_16_84 = QCheckBox("Include 16/84 percentiles", self)
        self.chk_error_ci_16_84.setChecked(False)
        self.chk_error_ci_16_84.toggled.connect(self._apply_errors_ci_visibility)
        flags_row.addWidget(self.chk_error_ci_16_84)

        self.chk_error_show_corr = QCheckBox("Show correlation matrix", self)
        self.chk_error_show_corr.setChecked(True)
        self.chk_error_show_corr.toggled.connect(self._apply_errors_corr_visibility)
        flags_row.addWidget(self.chk_error_show_corr)

        flags_row.addStretch(1)
        self.btn_error_compute = QPushButton("Compute", self)
        self.btn_error_compute.clicked.connect(self._on_error_compute_clicked)
        flags_row.addWidget(self.btn_error_compute)
        self.btn_error_export = QPushButton("Export XLSX", self)
        self.btn_error_export.setEnabled(False)
        self.btn_error_export.clicked.connect(self._on_error_export_clicked)
        flags_row.addWidget(self.btn_error_export)
        layout.addLayout(flags_row)

        progress_row = QHBoxLayout()
        self.lbl_error_status = QLabel("", self)
        progress_row.addWidget(self.lbl_error_status)
        self.progress_error = QProgressBar(self)
        self.progress_error.setVisible(False)
        self.progress_error.setTextVisible(True)
        self.progress_error.setFormat("Working...")
        progress_row.addWidget(self.progress_error, 1)
        self.btn_error_cancel = QPushButton("Cancel", self)
        self.btn_error_cancel.setEnabled(False)
        self.btn_error_cancel.clicked.connect(self._on_error_cancel_clicked)
        progress_row.addWidget(self.btn_error_cancel)
        layout.addLayout(progress_row)

        layout.addWidget(QLabel("Results", self))
        self.table_error_results = QTableWidget(self)
        self.table_error_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_error_results.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table_error_results.setAlternatingRowColors(True)
        self.table_error_results.verticalHeader().setVisible(False)
        layout.addWidget(self.table_error_results, 1)

        self.lbl_error_extra = QLabel("Derived tables", self)
        self.lbl_error_extra.setVisible(False)
        layout.addWidget(self.lbl_error_extra)
        self.tabs_error_extra = QTabWidget(self)
        self.tabs_error_extra.setVisible(False)
        layout.addWidget(self.tabs_error_extra)

        self.lbl_error_corr = QLabel("Correlation matrix", self)
        layout.addWidget(self.lbl_error_corr)
        self.table_error_corr = QTableWidget(self)
        self.table_error_corr.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_error_corr.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table_error_corr.setAlternatingRowColors(True)
        layout.addWidget(self.table_error_corr)

        layout.addWidget(QLabel("Summary", self))
        self.text_error_summary = QPlainTextEdit(self)
        self.text_error_summary.setReadOnly(True)
        self.text_error_summary.setMinimumHeight(90)
        layout.addWidget(self.text_error_summary)

        self._set_default_results_headers()
        self._apply_errors_ci_visibility()
        self._apply_errors_corr_visibility()
        self._on_error_method_changed()

    def _set_default_results_headers(self) -> None:
        columns = [
            "Parameter",
            "Estimate log10K",
            "Median",
            "CI 2.5",
            "CI 97.5",
            "CI 16",
            "CI 84",
            "SE(log10K)",
            "%err(K)",
        ]
        self.table_error_results.setColumnCount(len(columns))
        self.table_error_results.setHorizontalHeaderLabels(columns)
        header = self.table_error_results.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, len(columns)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self._results_ci_columns = (5, 6)

    def set_supported_methods(self, supported_methods: set[str] | list[str], *, default: str = "analytic") -> None:
        supported = {str(method) for method in supported_methods}
        model = self.combo_error_method.model()
        for idx in range(self.combo_error_method.count()):
            method = str(self.combo_error_method.itemData(idx) or "")
            item = model.item(idx)
            if item is not None:
                item.setEnabled(method in supported)
        current_method = str(self.combo_error_method.currentData() or "")
        if current_method not in supported:
            default_index = self.combo_error_method.findData(default)
            if default_index >= 0:
                self.combo_error_method.setCurrentIndex(default_index)
            else:
                for idx in range(self.combo_error_method.count()):
                    method = str(self.combo_error_method.itemData(idx) or "")
                    if method in supported:
                        self.combo_error_method.setCurrentIndex(idx)
                        break

    def has_errors_context(self) -> bool:
        return bool(self._errors_context)

    def last_output(self) -> dict[str, Any] | None:
        return self._errors_last_output

    def set_errors_context(self, context: dict[str, Any] | None, *, auto_compute: bool = False) -> None:
        self._errors_context = context
        self._errors_last_output = None
        self.btn_error_export.setEnabled(False)
        self._clear_errors_tables()
        self._refresh_idle_state()
        if context and auto_compute:
            self.combo_error_method.setCurrentIndex(0)
            self._on_error_compute_clicked()

    def compute_now(self) -> dict[str, Any]:
        if not self._errors_context:
            raise ValueError("Run a fit first.")
        options = self._collect_errors_options()
        payload = {"ctx": dict(self._errors_context), "options": options}
        output = self._compute_errors_payload(payload)
        self._apply_errors_output(output)
        return output

    def _refresh_idle_state(self) -> None:
        running = self._errors_worker is not None
        has_context = bool(self._errors_context)
        controls = [
            self.combo_error_method,
            self.spin_error_b,
            self.edit_error_seed,
            self.combo_error_wild,
            self.spin_error_lambda,
            self.spin_error_max_iter,
            self.spin_error_tol,
            self.combo_error_fail_policy,
            self.chk_error_ci_16_84,
            self.chk_error_show_corr,
        ]
        for widget in controls:
            widget.setEnabled((not running) and has_context)
        self.btn_error_compute.setEnabled((not running) and has_context)
        self.btn_error_cancel.setEnabled(running)
        self.btn_error_export.setEnabled((not running) and self._errors_last_output is not None)

    def _on_error_method_changed(self) -> None:
        method = str(self.combo_error_method.currentData() or "")
        is_bootstrap = method.startswith("bootstrap")
        is_onestep = method == "bootstrap_onestep"
        is_full_refit = method == "bootstrap_full_refit_audit"
        self.spin_error_b.setEnabled(is_bootstrap and self._errors_worker is None and bool(self._errors_context))
        self.combo_error_wild.setEnabled(is_bootstrap and self._errors_worker is None and bool(self._errors_context))
        self.spin_error_lambda.setEnabled(is_onestep and self._errors_worker is None and bool(self._errors_context))
        self.lbl_error_max_iter.setVisible(is_full_refit)
        self.spin_error_max_iter.setVisible(is_full_refit)
        self.lbl_error_tol.setVisible(is_full_refit)
        self.spin_error_tol.setVisible(is_full_refit)
        self.lbl_error_fail_policy.setVisible(is_full_refit)
        self.combo_error_fail_policy.setVisible(is_full_refit)
        self.lbl_error_audit_warning.setVisible(is_full_refit)

        if is_full_refit and self.spin_error_b.value() in (300, 500):
            self.spin_error_b.setValue(50)
        if is_onestep and self.spin_error_b.value() in (50, 500):
            self.spin_error_b.setValue(300)
        if (not is_onestep) and (not is_full_refit) and self.spin_error_b.value() in (50, 300):
            self.spin_error_b.setValue(500)

    def _apply_errors_ci_visibility(self) -> None:
        show = bool(self.chk_error_ci_16_84.isChecked())
        for column in self._results_ci_columns:
            if column is not None and 0 <= column < self.table_error_results.columnCount():
                self.table_error_results.setColumnHidden(column, not show)

    def _apply_errors_corr_visibility(self) -> None:
        show = bool(self.chk_error_show_corr.isChecked())
        self.lbl_error_corr.setVisible(show)
        self.table_error_corr.setVisible(show)

    def _parse_error_seed(self) -> int | None:
        raw = str(self.edit_error_seed.text() or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError("Seed must be an integer.") from exc

    def _collect_errors_options(self) -> dict[str, Any]:
        return {
            "method": str(self.combo_error_method.currentData() or "analytic"),
            "seed": self._parse_error_seed(),
            "wild": str(self.combo_error_wild.currentData() or "rademacher"),
            "lam": float(self.spin_error_lambda.value()),
            "B": int(self.spin_error_b.value()),
            "include_16_84": bool(self.chk_error_ci_16_84.isChecked()),
            "max_iter": int(self.spin_error_max_iter.value()),
            "tol": float(self.spin_error_tol.value()),
            "fail_policy": str(self.combo_error_fail_policy.currentData() or "skip"),
        }

    def _on_error_compute_clicked(self) -> None:
        if self._errors_worker is not None:
            QMessageBox.information(self, "Errors", "An errors computation is already running.")
            return
        if not self._errors_context:
            QMessageBox.warning(self, "Errors", "Run a fit first.")
            return
        try:
            options = self._collect_errors_options()
        except Exception as exc:
            QMessageBox.warning(self, "Errors", str(exc))
            return
        payload = {"ctx": dict(self._errors_context), "options": options}
        self._start_errors_worker(payload)

    def _start_errors_worker(self, payload: dict[str, Any]) -> None:
        self._errors_last_output = None
        self.btn_error_export.setEnabled(False)
        self._set_errors_busy(True, payload.get("options") or {})
        self._errors_worker = ErrorsWorker(self._compute_errors_payload, payload=payload, parent=self)
        worker_thread = self._errors_worker.thread()
        worker_thread.finished.connect(self._on_error_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._errors_worker.progress.connect(self._on_error_progress, Qt.ConnectionType.QueuedConnection)
        self._errors_worker.result.connect(self._on_error_worker_result, Qt.ConnectionType.QueuedConnection)
        self._errors_worker.error.connect(self._on_error_worker_error, Qt.ConnectionType.QueuedConnection)
        self._errors_worker.cancelled.connect(self._on_error_worker_cancelled, Qt.ConnectionType.QueuedConnection)
        self._errors_worker.finished.connect(self._on_error_worker_finished, Qt.ConnectionType.QueuedConnection)
        self._errors_worker.start()

    def _compute_errors_payload(
        self,
        payload: dict[str, Any],
        *,
        progress_cb=None,
        cancel_cb=None,
    ) -> dict[str, Any]:
        return self._compute_payload_fn(payload, progress_cb=progress_cb, cancel_cb=cancel_cb)

    def _set_errors_busy(self, running: bool, options: dict[str, Any]) -> None:
        if running:
            self._configure_errors_progress(options)
            self.lbl_error_status.setText("Computing errors...")
        else:
            self.lbl_error_status.setText("")
            self.progress_error.setVisible(False)
        self._refresh_idle_state()

    def _configure_errors_progress(self, options: dict[str, Any]) -> None:
        method = str(options.get("method") or "")
        self.progress_error.setVisible(True)
        if method == "bootstrap_full_refit_audit":
            total = max(int(options.get("B") or 0), 1)
            self.progress_error.setRange(0, total)
            self.progress_error.setValue(0)
            self.progress_error.setFormat("Bootstrap %v/%m")
        else:
            self.progress_error.setRange(0, 0)
            self.progress_error.setFormat("Working...")

    def _on_error_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            current = payload.get("current")
            total = payload.get("total")
            n_success = payload.get("n_success")
            n_fail = payload.get("n_fail")
            if total is not None:
                try:
                    total_int = int(total)
                    current_int = int(current or 0)
                except Exception:
                    return
                if total_int > 0:
                    self.progress_error.setRange(0, total_int)
                    self.progress_error.setValue(current_int)
                    if n_success is not None and n_fail is not None:
                        self.progress_error.setFormat(
                            f"Bootstrap {current_int}/{total_int} (ok {int(n_success)}, fail {int(n_fail)})"
                        )
            return
        if isinstance(payload, str):
            self.lbl_error_status.setText(str(payload))

    def _on_error_cancel_clicked(self) -> None:
        if self._errors_worker is None:
            return
        self._errors_worker.request_cancel()
        self.lbl_error_status.setText("Cancelling...")

    def _on_error_worker_result(self, output: dict[str, Any]) -> None:
        self._apply_errors_output(output)

    def _on_error_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Errors", str(message))

    def _on_error_worker_cancelled(self) -> None:
        self.text_error_summary.setPlainText("Cancelled by user.")
        self.lbl_error_status.setText("Cancelled.")

    def _on_error_worker_finished(self) -> None:
        self._set_errors_busy(False, {})

    def _on_error_thread_finished(self) -> None:
        self._errors_worker = None
        self._refresh_idle_state()

    def _apply_errors_output(self, output: dict[str, Any]) -> None:
        if output.get("results_columns") and output.get("results_rows") is not None:
            columns = [str(value) for value in (output.get("results_columns") or [])]
            rows = list(output.get("results_rows") or [])
            self._populate_results_table(columns, rows)
        else:
            self._populate_legacy_results(output)

        corr = output.get("corr")
        param_names = [str(value) for value in (output.get("param_names") or [])]
        if corr is not None and self.chk_error_show_corr.isChecked():
            self._populate_corr_table(np.asarray(corr, dtype=float), param_names)
        else:
            self._clear_corr_table()

        self._render_extra_tables(list(output.get("extra_tables") or []))
        self.text_error_summary.setPlainText(str(output.get("summary") or ""))
        self._errors_last_output = output
        self._refresh_idle_state()
        self.output_ready.emit(output)

    def _populate_results_table(self, columns: list[str], rows: list[Any]) -> None:
        self.table_error_results.clear()
        self.table_error_results.setColumnCount(len(columns))
        self.table_error_results.setHorizontalHeaderLabels(columns)
        self.table_error_results.setRowCount(len(rows))
        ci16_col = None
        ci84_col = None
        for idx, name in enumerate(columns):
            normalized = name.strip().lower().replace("%", "").replace(".", "").replace(" ", "")
            if normalized in {"ci16", "p16"}:
                ci16_col = idx
            if normalized in {"ci84", "p84"}:
                ci84_col = idx
        self._results_ci_columns = (ci16_col, ci84_col)
        header = self.table_error_results.horizontalHeader()
        if columns:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, len(columns)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        for row_idx, row in enumerate(rows):
            values = list(row) if isinstance(row, (list, tuple)) else [row]
            for col_idx, value in enumerate(values[: len(columns)]):
                align_left = col_idx == 0
                self._set_table_cell(self.table_error_results, row_idx, col_idx, value, align_left=align_left)
        self._apply_errors_ci_visibility()

    def _populate_legacy_results(self, output: dict[str, Any]) -> None:
        self._set_default_results_headers()
        param_names = list(output.get("param_names") or [])
        k_hat = np.asarray(output.get("k_hat") or [], dtype=float)
        median = np.asarray(output.get("median") or [], dtype=float)
        ci_2p5 = np.asarray(output.get("ci_2p5") or [], dtype=float)
        ci_97p5 = np.asarray(output.get("ci_97p5") or [], dtype=float)
        ci_16 = output.get("ci_16")
        ci_84 = output.get("ci_84")
        se_log10 = np.asarray(output.get("se_log10") or [], dtype=float)
        perc_err = np.asarray(output.get("perc_err") or [], dtype=float)
        self.table_error_results.setRowCount(len(param_names))
        for row, name in enumerate(param_names):
            self._set_table_cell(self.table_error_results, row, 0, str(name), align_left=True)
            self._set_table_cell(self.table_error_results, row, 1, self._fmt_num(k_hat, row))
            self._set_table_cell(self.table_error_results, row, 2, self._fmt_num(median, row))
            self._set_table_cell(self.table_error_results, row, 3, self._fmt_num(ci_2p5, row))
            self._set_table_cell(self.table_error_results, row, 4, self._fmt_num(ci_97p5, row))
            self._set_table_cell(self.table_error_results, row, 5, self._fmt_num(ci_16, row))
            self._set_table_cell(self.table_error_results, row, 6, self._fmt_num(ci_84, row))
            self._set_table_cell(self.table_error_results, row, 7, self._fmt_num(se_log10, row))
            self._set_table_cell(self.table_error_results, row, 8, self._fmt_num(perc_err, row))
        self._apply_errors_ci_visibility()

    def _render_extra_tables(self, extra_tables: list[dict[str, Any]]) -> None:
        self.tabs_error_extra.clear()
        has_tables = False
        for table_def in extra_tables:
            name = str(table_def.get("name") or "Table")
            columns = [str(value) for value in (table_def.get("columns") or [])]
            rows = list(table_def.get("rows") or [])
            row_headers = [str(value) for value in (table_def.get("row_headers") or [])]
            table = QTableWidget(self.tabs_error_extra)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            table.setAlternatingRowColors(True)
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)
            table.setRowCount(len(rows))
            if row_headers and len(row_headers) == len(rows):
                table.setVerticalHeaderLabels(row_headers)
            else:
                table.verticalHeader().setVisible(False)
            header = table.horizontalHeader()
            for col in range(len(columns)):
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch if col else QHeaderView.ResizeMode.ResizeToContents)
            for row_idx, row in enumerate(rows):
                values = list(row) if isinstance(row, (list, tuple)) else [row]
                for col_idx, value in enumerate(values[: len(columns)]):
                    self._set_table_cell(table, row_idx, col_idx, value, align_left=(col_idx == 0))
            self.tabs_error_extra.addTab(table, name)
            has_tables = True
        self.lbl_error_extra.setVisible(has_tables)
        self.tabs_error_extra.setVisible(has_tables)

    def _set_table_cell(
        self,
        table: QTableWidget,
        row: int,
        col: int,
        value: Any,
        *,
        align_left: bool = False,
    ) -> None:
        text = str(value) if value is not None else ""
        if isinstance(value, (float, np.floating)):
            text = "" if not np.isfinite(float(value)) else f"{float(value):.6g}"
        item = QTableWidgetItem(text)
        if align_left:
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
        else:
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        table.setItem(row, col, item)

    def _fmt_num(self, arr: Any, idx: int) -> str:
        if arr is None:
            return ""
        try:
            val = float(np.asarray(arr).ravel()[idx])
        except Exception:
            return ""
        if not np.isfinite(val):
            return ""
        return f"{val:.6g}"

    def _populate_corr_table(self, corr: np.ndarray, param_names: list[str]) -> None:
        if corr.size == 0:
            self._clear_corr_table()
            return
        p = corr.shape[0]
        self.table_error_corr.clear()
        self.table_error_corr.setRowCount(p)
        self.table_error_corr.setColumnCount(p)
        self.table_error_corr.setHorizontalHeaderLabels(param_names)
        self.table_error_corr.setVerticalHeaderLabels(param_names)
        self.table_error_corr.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(p):
            for j in range(p):
                val = corr[i, j]
                text = "" if not np.isfinite(val) else f"{val:.3f}"
                self._set_table_cell(self.table_error_corr, i, j, text)

    def _clear_corr_table(self) -> None:
        self.table_error_corr.clear()
        self.table_error_corr.setRowCount(0)
        self.table_error_corr.setColumnCount(0)

    def _clear_errors_tables(self) -> None:
        self.table_error_results.setRowCount(0)
        self._clear_corr_table()
        self.tabs_error_extra.clear()
        self.tabs_error_extra.setVisible(False)
        self.lbl_error_extra.setVisible(False)
        self.text_error_summary.clear()

    def _on_error_export_clicked(self) -> None:
        if not self._errors_last_output:
            QMessageBox.information(self, "Errors", "Compute errors first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export errors", "errors.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not str(path).lower().endswith(".xlsx"):
            path = f"{path}.xlsx"
        try:
            import pandas as pd

            output = self._errors_last_output
            frames = output.get("export_frames")
            with pd.ExcelWriter(path) as writer:
                if isinstance(frames, dict) and frames:
                    for sheet_name, data in frames.items():
                        df = self._to_frame(data)
                        if df is None or df.empty:
                            continue
                        use_index = str(sheet_name) in {"Covariance", "Correlation"}
                        df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=use_index)
                else:
                    param_names = list(output.get("param_names") or [])
                    df = pd.DataFrame(
                        {
                            "Parameter": param_names,
                            "Estimate log10K": output.get("k_hat"),
                            "Median": output.get("median"),
                            "CI 2.5": output.get("ci_2p5"),
                            "CI 97.5": output.get("ci_97p5"),
                            "CI 16": output.get("ci_16"),
                            "CI 84": output.get("ci_84"),
                            "SE(log10K)": output.get("se_log10"),
                            "%err(K)": output.get("perc_err"),
                        }
                    )
                    df.to_excel(writer, sheet_name="Results", index=False)
                    corr = output.get("corr")
                    if corr is not None:
                        pd.DataFrame(corr, index=param_names, columns=param_names).to_excel(
                            writer,
                            sheet_name="Correlation",
                        )
                summary = str(output.get("summary") or "").strip()
                if summary:
                    pd.DataFrame({"summary": summary.splitlines()}).to_excel(
                        writer,
                        sheet_name="Summary",
                        index=False,
                    )
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _to_frame(self, data: Any):
        import pandas as pd

        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            return pd.DataFrame(data)
        return pd.DataFrame(data)
