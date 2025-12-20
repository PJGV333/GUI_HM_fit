from __future__ import annotations

from typing import Any


def build_nmr_plot_sources(result: dict[str, Any]) -> dict[str, Any]:
    plot_data = result.get("plotData", {}).get("nmr") or {}
    return plot_data if isinstance(plot_data, dict) else {}
