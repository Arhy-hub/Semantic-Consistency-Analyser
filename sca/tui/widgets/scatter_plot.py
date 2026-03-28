"""ASCII UMAP scatter plot widget."""

from __future__ import annotations

import numpy as np
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text

# Colours per cluster index (cycles)
_COLORS = ["#00d7d7", "#ffffff", "#aaaaaa", "#666666", "#00aaaa", "#dddddd"]


class ScatterPlot(Widget):
    BORDER_TITLE = "umap scatter"
    DEFAULT_CSS = """
    ScatterPlot {
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        height: 100%;
        overflow: hidden;
        background: #0a0a0a;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._coords: np.ndarray | None = None   # shape (n, 2)
        self._labels: np.ndarray | None = None    # shape (n,)
        self._error: str | None = None

    def set_error(self, msg: str) -> None:
        self._error = msg
        self.refresh()

    def update(self, coords: np.ndarray, labels: np.ndarray) -> None:
        self._coords = coords
        self._labels = labels
        self.border_subtitle = f"n={len(coords)}"
        self.refresh()

    def render(self) -> RenderResult:
        if self._error is not None:
            return Text(self._error, style="dim #ff5555")
        if self._coords is None:
            return Text("waiting for umap…\n(requires umap-learn)", style="dim #555555")

        h = self.size.height
        w = self.size.width
        coords = self._coords
        labels = self._labels

        x = coords[:, 0]
        y = coords[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0

        # Build character grid — background dots for reference
        grid_char  = [["·"] * w for _ in range(h)]
        grid_style = [["#1a1a1a"] * w for _ in range(h)]

        for i, (px, py) in enumerate(zip(x, y)):
            col = int((px - x_min) / x_range * (w - 1))
            row = int((1.0 - (py - y_min) / y_range) * (h - 1))
            col = max(0, min(w - 1, col))
            row = max(0, min(h - 1, row))
            lbl = int(labels[i]) if labels is not None else 0
            grid_char[row][col]  = "●"
            grid_style[row][col] = _COLORS[lbl % len(_COLORS)]

        out = Text(no_wrap=True, overflow="fold")
        for r in range(h):
            for c in range(w):
                out.append(grid_char[r][c], style=grid_style[r][c])
            if r < h - 1:
                out.append("\n")
        return out
