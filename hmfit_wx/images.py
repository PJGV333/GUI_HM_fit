from __future__ import annotations

import base64
from typing import Optional


def bitmap_from_base64_png(png_base64: str):
    import wx

    if not png_base64:
        return None

    raw = base64.b64decode(png_base64)
    try:
        stream = wx.MemoryInputStream(raw)
        img = wx.Image(stream, wx.BITMAP_TYPE_PNG)
        return wx.Bitmap(img)
    except Exception:
        return None


def as_bool_cell(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v in {"1", "true", "yes", "y", "t"}


def as_float_or_none(value: str) -> Optional[float]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None

