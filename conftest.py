"""Repository-wide pytest configuration.

Tests in modules that import PySide6 or ``hmfit_gui_qt`` are marked ``gui``
automatically so CI can run the non-GUI subset on platforms without a usable
Qt offscreen stack (``pytest -m "not gui"``). No test is skipped or removed by
this hook; it only adds a marker.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_GUI_TOKENS = ("PySide6", "hmfit_gui_qt")
_cache: dict[str, bool] = {}


def _module_uses_gui(path: Path) -> bool:
    key = str(path)
    if key not in _cache:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        _cache[key] = any(token in text for token in _GUI_TOKENS)
    return _cache[key]


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(getattr(item, "path", item.fspath)))
        if _module_uses_gui(path):
            item.add_marker(pytest.mark.gui)
