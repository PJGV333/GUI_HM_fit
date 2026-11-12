# SPDX-License-Identifier: GPL-3.0-or-later
"""Small layout helpers shared by the module tabs."""
from __future__ import annotations

from PySide6.QtWidgets import QScrollArea, QSplitter


def fit_left_panel(splitter: QSplitter, scroll: QScrollArea, *, max_fraction: float = 0.55) -> None:
    """
    Give the left (scrollable) panel the width its content needs.

    A ``QScrollArea`` reports a capped size hint, so the splitter used to give the panel
    about 540 px while its content needs about 700 px: buttons and table columns were cut
    off behind a horizontal scroll bar. The panel never takes more than ``max_fraction`` of
    the window, so the plot keeps its space on small screens.
    """
    content = scroll.widget()
    sizes = splitter.sizes()
    total = int(sum(sizes))
    if content is None or total <= 0 or len(sizes) < 2:
        return
    needed = (
        content.minimumSizeHint().width()
        + scroll.verticalScrollBar().sizeHint().width()
        + 2 * scroll.frameWidth()
        + 4
    )
    left = max(sizes[0], min(int(needed), int(max_fraction * total)))
    splitter.setSizes([left, total - left] + [0] * (len(sizes) - 2))


__all__ = ["fit_left_panel"]
