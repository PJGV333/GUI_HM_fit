from __future__ import annotations

import math
import re
from typing import Any

_RANGE_RE = re.compile(r"^(-?\\d+(?:\\.\\d+)?)\\s*-\\s*(-?\\d+(?:\\.\\d+)?)$")


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
            "errors": ["Channels is empty. Use 'all' or a list like 450,650."],
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
        errors.append("No channels parsed. Use 'all' or values like 450,650.")

    return {"mode": "custom", "tokens": tokens, "errors": errors}


def resolve_channels(tokens: list[dict[str, Any]], axis_values: list[float], *, tol: float = 1e-6) -> dict[str, Any]:
    """
    Resolve parsed tokens onto concrete axis values.

    - For value tokens, picks the nearest axis value and requires diff <= tol.
    - For range tokens, selects all axis values within [min,max] (inclusive).
    """
    axis = [float(v) for v in (axis_values or []) if v is not None and math.isfinite(float(v))]
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
            d = abs(float(a) - float(target))
            if d < best_diff:
                best_val = a
                best_diff = d
        return best_val, best_diff

    for tok in tokens or []:
        ttype = tok.get("type")
        if ttype == "value":
            target = tok.get("value")
            raw = tok.get("raw") or str(target)
            if target is None:
                errors.append(f"Invalid channel value: '{raw}'.")
                continue
            best, diff = nearest(float(target))
            if best is None or not math.isfinite(diff):
                errors.append(f"No axis values available to resolve '{raw}'.")
                continue
            if diff > float(tol):
                errors.append(f"'{raw}' is not within tolerance (tol={tol}) of any axis value.")
                continue
            resolved_set.add(float(best))
            mapping_lines.append(f"{raw} → {best}")
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
                resolved_set.add(float(v))
            mapping_lines.append(f"{raw} → {len(in_range)} channels")
            continue

        errors.append(f"Unsupported token '{tok.get('raw', '')}'.")

    resolved = [v for v in axis if v in resolved_set]
    return {"resolved": resolved, "mapping_lines": mapping_lines, "errors": errors}
