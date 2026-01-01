from __future__ import annotations

import math
import re
from typing import Any, Iterable

_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$")
DEFAULT_CHANNEL_TOL = 0.5


class ChannelSpecResult(set):
    def __init__(
        self,
        iterable: Iterable[float] | None = None,
        *,
        mode: str = "custom",
        errors: list[str] | None = None,
        mapping_lines: list[str] | None = None,
    ) -> None:
        super().__init__(iterable or [])
        self.mode = mode
        self.errors = list(errors or [])
        self.mapping_lines = list(mapping_lines or [])


def parse_channels_spec(spec: str) -> dict[str, Any]:
    """
    Parse a Channels specification string (ported from hmfit_tauri/src/main.js).

    Supported (case-insensitive):
    - "all"
    - "250-400"
    - "386, 485, 512.5"
    - combinations like "250-400, 485, 510-520"
    """
    raw = str(spec or "").strip()
    if not raw:
        return {
            "mode": "custom",
            "tokens": [],
            "errors": ["Channels is empty. Use 'All' or a list like 450,650."],
        }

    if raw.lower() == "all":
        return {"mode": "all", "tokens": [], "errors": []}

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    tokens: list[dict[str, Any]] = []
    errors: list[str] = []

    for part in parts:
        m = _RANGE_RE.match(part)
        if m:
            try:
                a = float(m.group(1))
                b = float(m.group(2))
            except Exception:
                errors.append(f"Invalid range: '{part}'.")
                continue
            if not math.isfinite(a) or not math.isfinite(b):
                errors.append(f"Invalid range: '{part}'.")
                continue
            tokens.append({"type": "range", "min": a, "max": b, "raw": part})
            continue

        try:
            v = float(part)
        except Exception:
            errors.append(f"Invalid channel value: '{part}'.")
            continue
        if not math.isfinite(v):
            errors.append(f"Invalid channel value: '{part}'.")
            continue
        tokens.append({"type": "value", "value": v, "raw": part})

    if not tokens and not errors:
        errors.append("No channels parsed. Use 'All' or values like 450,650.")

    return {"mode": "custom", "tokens": tokens, "errors": errors}


def resolve_channels(
    tokens: list[dict[str, Any]],
    axis_values: list[float],
    *,
    tol: float = DEFAULT_CHANNEL_TOL,
) -> dict[str, Any]:
    """
    Resolve parsed tokens onto concrete axis values.

    - For value tokens, picks the nearest axis value and requires diff <= tol.
    - For range tokens, selects all axis values within [min,max] (inclusive).
    """
    axis: list[float] = []
    for v in axis_values or []:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            axis.append(f)

    if not axis:
        return {
            "resolved": [],
            "mapping_lines": [],
            "errors": ["Axis not loaded yet. Select a Spectra sheet first."],
        }

    errors: list[str] = []
    mapping_lines: list[str] = []
    resolved_set: set[float] = set()

    def nearest(target: float) -> tuple[float | None, float]:
        best_val: float | None = None
        best_diff = float("inf")
        for a in axis:
            d = abs(a - target)
            if d < best_diff:
                best_val = a
                best_diff = d
        return best_val, best_diff

    for tok in tokens or []:
        ttype = tok.get("type")
        if ttype == "value":
            raw = tok.get("raw") or str(tok.get("value"))
            try:
                target = float(tok.get("value"))
            except Exception:
                errors.append(f"Invalid channel value: '{raw}'.")
                continue
            nearest_val, diff = nearest(target)
            if nearest_val is None or not math.isfinite(diff):
                errors.append(f"No axis values available to resolve '{raw}'.")
                continue
            if diff > float(tol):
                errors.append(f"'{raw}' is not within tolerance (tol={tol}) of any axis value.")
                continue
            resolved_set.add(nearest_val)
            mapping_lines.append(f"{raw} → {nearest_val}")
            continue

        if ttype == "range":
            raw = str(tok.get("raw") or "")
            try:
                lo = float(tok.get("min"))
                hi = float(tok.get("max"))
            except Exception:
                errors.append(f"Invalid range: '{raw}'.")
                continue
            lo2 = min(lo, hi)
            hi2 = max(lo, hi)
            in_range = [v for v in axis if lo2 <= v <= hi2]
            if not in_range:
                errors.append(f"Range '{raw}' matched 0 axis values.")
                continue
            for v in in_range:
                resolved_set.add(v)
            mapping_lines.append(f"{raw} → {len(in_range)} channels")
            continue

        errors.append(f"Unsupported token '{tok.get('raw', '')}'.")

    resolved = [v for v in axis if v in resolved_set]
    return {
        "resolved": resolved,
        "mapping_lines": mapping_lines,
        "errors": errors,
    }


from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QLabel,
    QScrollArea,
)

class ChannelItem(QWidget):
    """A row in the ChannelSpecWidget representing one potential signal."""
    toggled = Signal(bool)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        self.checkbox = QCheckBox(name)
        self.checkbox.toggled.connect(self.toggled.emit)
        layout.addWidget(self.checkbox, 1)
        
        self.combo_parent = QComboBox()
        self.combo_parent.setFixedWidth(150)
        self.combo_parent.addItem("Auto (1:1)")
        self.combo_parent.currentIndexChanged.connect(self._update_style)
        layout.addWidget(self.combo_parent)
        
        # Color feedback
        self.setAutoFillBackground(True)
        self._update_style()

    def _update_style(self):
        p = self.palette()
        if self.combo_parent.currentIndex() > 0: # Not "Auto (1:1)"
            # Subtle highlight (e.g., light blue/green)
            p.setColor(self.backgroundRole(), Qt.GlobalColor.darkCyan)
            self.setStyleSheet("background-color: rgba(0, 255, 255, 30); border-radius: 4px;")
        else:
            self.setStyleSheet("")
        self.setPalette(p)

    def set_parent_options(self, options: list[str]):
        current = self.combo_parent.currentText()
        self.combo_parent.blockSignals(True)
        self.combo_parent.clear()
        self.combo_parent.addItems(["Auto (1:1)"] + options + ["Mezcla"])
        
        index = self.combo_parent.findText(current)
        if index >= 0:
            self.combo_parent.setCurrentIndex(index)
        else:
            self.combo_parent.setCurrentIndex(0)
        self.combo_parent.blockSignals(False)
        self._update_style()

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool):
        self.checkbox.setChecked(checked)

    def get_data(self) -> dict[str, str]:
        return {
            "col_name": self.name,
            "parent": self.combo_parent.currentText()
        }

    def set_data(self, data: dict[str, str]):
        self.checkbox.setChecked(True)
        idx = self.combo_parent.findText(data.get("parent", ""))
        if idx >= 0:
            self.combo_parent.setCurrentIndex(idx)


class ChannelSpecWidget(QWidget):
    """Widget to select multiple signals and assign their parents."""
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[ChannelItem] = []
        self._component_names: list[str] = []
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.addStretch(1)
        self.scroll.setWidget(self.container)
        
        layout.addWidget(self.scroll)

    def set_channels(self, names: list[str]):
        # Clear existing
        for item in self._items:
            item.setParent(None)
            item.deleteLater()
        self._items = []
        
        # Add new
        # Remove stretch temporarily
        if self.container_layout.count() > 0:
            last_item = self.container_layout.takeAt(self.container_layout.count() - 1)
        
        for name in names:
            item = ChannelItem(name)
            item.toggled.connect(lambda _: self.selectionChanged.emit())
            self._items.append(item)
            self.container_layout.addWidget(item)
            item.set_parent_options(self._component_names)
            
        self.container_layout.addStretch(1)

    def update_parent_options(self, component_names: list[str]):
        self._component_names = component_names
        for item in self._items:
            item.set_parent_options(component_names)

    def get_selected_channels(self) -> list[dict[str, str]]:
        return [item.get_data() for item in self._items if item.is_checked()]

    def set_all_checked(self, checked: bool):
        for item in self._items:
            item.set_checked(checked)
        self.selectionChanged.emit()

    def set_selected_channels(self, selected_names: list[str]):
        names_set = set(selected_names)
        for item in self._items:
            item.set_checked(item.name in names_set)
        self.selectionChanged.emit()

    def set_selected_data(self, selected_data: list[dict[str, str]]):
        data_map = {d["col_name"]: d for d in selected_data}
        for item in self._items:
            if item.name in data_map:
                item.set_data(data_map[item.name])
            else:
                item.set_checked(False)
        self.selectionChanged.emit()


def parse_channel_spec(
    spec: str,
    available: list[float],
    *,
    tol: float = DEFAULT_CHANNEL_TOL,
) -> ChannelSpecResult:
    parsed = parse_channels_spec(spec)
    mode = str(parsed.get("mode") or "custom")
    if mode == "all":
        return ChannelSpecResult(list(available or []), mode="all")

    resolved_info = resolve_channels(list(parsed.get("tokens") or []), list(available or []), tol=tol)
    errors = list(parsed.get("errors") or []) + list(resolved_info.get("errors") or [])
    resolved = list(resolved_info.get("resolved") or [])
    mapping_lines = list(resolved_info.get("mapping_lines") or [])
    return ChannelSpecResult(resolved, mode="custom", errors=errors, mapping_lines=mapping_lines)
