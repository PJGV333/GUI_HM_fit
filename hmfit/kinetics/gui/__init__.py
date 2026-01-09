"""GUI components for the kinetics module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import KineticsMainWidget

__all__ = ["KineticsMainWidget"]


def __getattr__(name: str):
    if name == "KineticsMainWidget":
        from .main import KineticsMainWidget

        return KineticsMainWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
