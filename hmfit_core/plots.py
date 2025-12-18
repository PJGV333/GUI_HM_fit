from __future__ import annotations

import base64
import io
from typing import Any, Dict, List


def figure_from_png_base64(png_base64: str, *, title: str | None = None):
    from matplotlib.figure import Figure

    raw = base64.b64decode(png_base64)
    img = None
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        import matplotlib.image as mpimg

        img = mpimg.imread(io.BytesIO(raw), format="png")

    fig = Figure(figsize=(6.0, 4.0), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def figures_from_graphs(graphs: Dict[str, Any]) -> List[Any]:
    figs: List[Any] = []
    for key, b64 in (graphs or {}).items():
        if not b64:
            continue
        try:
            figs.append(figure_from_png_base64(str(b64), title=str(key)))
        except Exception:
            continue
    return figs
