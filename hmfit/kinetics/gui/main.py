"""Main GUI for the kinetics module (Qt wrapper)."""

from __future__ import annotations

from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab


class KineticsMainWidget(KineticsTab):
    """Compatibility wrapper for the unified Qt kinetics tab."""
