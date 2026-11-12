# SPDX-License-Identifier: GPL-3.0-or-later
"""Spin box for quantities that span many orders of magnitude (concentrations)."""
from __future__ import annotations

import math

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox, QWidget


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """
    ``QDoubleSpinBox`` that keeps the value the user typed.

    A plain ``QDoubleSpinBox`` rounds its value to ``decimals()`` digits after the point: with
    six decimals an initial concentration of 2.5e-7 M became 0 and 1.5e-6 M became 2e-6 M
    without any message. Here the value is stored unrounded, shown with up to
    ``significant`` significant digits (``1e-06``, ``0.00125``, ``2.5e-07``) and typed in plain
    or scientific notation with a decimal point. The arrows change the value by 10 %.

    It is still a ``QDoubleSpinBox``, so ``value()``, ``setValue()``, ``valueChanged`` and
    ``isinstance(..., QDoubleSpinBox)`` checks keep working.
    """

    def __init__(self, parent: QWidget | None = None, *, significant: int = 10) -> None:
        super().__init__(parent)
        self._significant = max(1, int(significant))
        # Qt rounds the stored value to this many decimals; 323 is the largest Qt accepts
        # and is beyond the resolution of a double, i.e. no rounding.
        self.setDecimals(323)
        self.setRange(0.0, 1.0e12)
        self.setKeyboardTracking(False)

    # ---- text <-> value ----
    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt API)
        if value == 0.0:
            return "0"
        return format(float(value), f".{self._significant}g")

    def valueFromText(self, text: str) -> float:  # noqa: N802 (Qt API)
        try:
            return float(self._strip(text))
        except ValueError:
            return float(self.value())

    def validate(self, text: str, pos: int):  # noqa: D401 (Qt API)
        body = self._strip(text)
        if body in {"", "+", "-", ".", "+.", "-."}:
            return (QValidator.State.Intermediate, text, pos)
        try:
            value = float(body)
        except ValueError:
            lowered = body.lower()
            # "2.5e", "2.5e-", "2.5e+" are on their way to a valid number
            for tail in ("e", "e-", "e+"):
                if lowered.endswith(tail):
                    try:
                        float(lowered[: -len(tail)])
                    except ValueError:
                        break
                    return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)
        if not math.isfinite(value):
            return (QValidator.State.Invalid, text, pos)
        if value < self.minimum() or value > self.maximum():
            return (QValidator.State.Intermediate, text, pos)
        return (QValidator.State.Acceptable, text, pos)

    def fixup(self, text: str) -> str:
        return self.textFromValue(self.value())

    def stepBy(self, steps: int) -> None:  # noqa: N802 (Qt API)
        value = float(self.value())
        if value == 0.0:
            value = float(self.singleStep()) * steps if steps > 0 else 0.0
        else:
            value = value * (1.1 ** steps)
        self.setValue(min(max(value, self.minimum()), self.maximum()))

    def _strip(self, text: str) -> str:
        body = str(text or "")
        prefix, suffix = self.prefix(), self.suffix()
        if prefix and body.startswith(prefix):
            body = body[len(prefix):]
        if suffix and body.endswith(suffix):
            body = body[: -len(suffix)]
        return body.strip()


__all__ = ["ScientificDoubleSpinBox"]
