"""Readable validation of parameter tables before a fit is launched.

The numerical core keeps raising ``ValueError`` for invalid bounds, NaN values and,
in kinetics, initial values outside the bounds. This module turns the same
conditions into messages that name the parameter and the field, say what is wrong
and how to fix it, e.g.::

    k2: the lower bound (0.5) is greater than the upper bound (0.1).
    Swap them, or set both equal to fix the parameter.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from PySide6.QtCore import QCoreApplication

_CTX = "ParameterValidation"


def _tr(text: str) -> str:
    return QCoreApplication.translate(_CTX, text)


def parse_number(text: Any) -> float | None | str:
    """float for numbers, None for an empty field, the raw text when it is not a number."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    raw = str(text).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


def _fmt(value: float) -> str:
    return f"{value:g}"


def parameter_issues(
    name: str,
    value: Any,
    lower: Any,
    upper: Any,
    *,
    fixed: bool = False,
    require_inside: bool = False,
    value_required: bool = True,
) -> list[str]:
    """Problems with one parameter row. ``value``/``lower``/``upper`` may be numbers,
    None (empty field / no limit) or raw strings (as typed by the user)."""
    issues: list[str] = []
    fields = (
        (_tr("initial value"), parse_number(value), value_required),
        (_tr("lower bound"), parse_number(lower), False),
        (_tr("upper bound"), parse_number(upper), False),
    )
    numbers: list[float | None] = []
    for label, parsed, required in fields:
        if isinstance(parsed, str):
            issues.append(
                _tr("{name}: the {field} ('{text}') is not a number. Enter a number"
                    " (use '.' as decimal separator)").format(name=name, field=label, text=parsed)
                + ("." if required else _tr(", or leave it empty for no limit."))
            )
            numbers.append(None)
        elif parsed is not None and math.isnan(parsed):
            issues.append(_tr("{name}: the {field} is NaN. Enter a number.").format(name=name, field=label))
            numbers.append(None)
        elif parsed is None and required:
            issues.append(_tr("{name}: the {field} is empty. Enter a number.").format(name=name, field=label))
            numbers.append(None)
        else:
            numbers.append(parsed)
    val, lo, hi = numbers
    if fixed:
        return issues
    if lo is not None and hi is not None and lo > hi:
        issues.append(
            _tr("{name}: the lower bound ({lo}) is greater than the upper bound ({hi}). "
                "Swap them, or set both equal to fix the parameter.").format(name=name, lo=_fmt(lo), hi=_fmt(hi))
        )
    elif require_inside and val is not None and ((lo is not None and val < lo) or (hi is not None and val > hi)):
        issues.append(
            _tr("{name}: the initial value ({val}) is outside its bounds [{lo}, {hi}]. "
                "Move it inside the interval or widen the bounds.").format(
                name=name, val=_fmt(val),
                lo=_fmt(lo) if lo is not None else "-inf", hi=_fmt(hi) if hi is not None else "+inf")
        )
    return issues


def raise_if_issues(issues: Iterable[str]) -> None:
    issues = [str(i) for i in issues if str(i).strip()]
    if issues:
        raise ValueError(_tr("Please correct the parameter table:") + "\n- " + "\n- ".join(issues))
