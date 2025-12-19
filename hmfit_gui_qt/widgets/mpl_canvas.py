from __future__ import annotations

import base64
import io

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        if parent is not None:
            self.setParent(parent)

    def clear(self) -> None:
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.draw_idle()

    def show_image_base64(self, png_base64: str, *, title: str | None = None) -> None:
        raw = base64.b64decode(str(png_base64))
        img = None
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception:
            import matplotlib.image as mpimg

            img = mpimg.imread(io.BytesIO(raw), format="png")

        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis("off")
        if title:
            self.ax.set_title(str(title))
        self.figure.tight_layout()
        self.draw_idle()


class NavigationToolbar(NavigationToolbar2QT):
    pass

