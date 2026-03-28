"""Similarity heatmap widget — fills the tile with axes and legend."""

from __future__ import annotations

import numpy as np
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text

# Left margin width (chars) reserved for row-axis labels
_LEFT = 4
# Rows reserved at the bottom: 1 col-axis + 1 legend bar + 1 legend labels
_BOTTOM = 3

# Continuous colour ramp: similarity 0 → 1 maps to dark → cyan
_RAMP = [
    (0.00, "#1a1a1a"),
    (0.35, "#2e2e2e"),
    (0.55, "#555555"),
    (0.75, "#999999"),
    (0.92, "#00d7d7"),
]


def _cell_style(value: float) -> str:
    """Interpolate through the ramp for a similarity value in [-1, 1]."""
    v = max(0.0, min(1.0, (value + 1.0) / 2.0))
    for i in range(len(_RAMP) - 1):
        lo_v, lo_c = _RAMP[i]
        hi_v, hi_c = _RAMP[i + 1]
        if v <= hi_v:
            # Return the closer bucket colour (no true colour blending needed)
            return hi_c if (v - lo_v) >= (hi_v - v) else lo_c
    return _RAMP[-1][1]


def _label_row(n: int, width: int) -> Text:
    """Build a column-axis label row of exactly `width` chars."""
    buf = [" "] * width
    # Place labels at a handful of evenly-spaced positions
    ticks = _tick_positions(n, width)
    for col, sample_idx in ticks:
        lbl = str(sample_idx + 1)
        for k, ch in enumerate(lbl):
            if col + k < width:
                buf[col + k] = ch
    t = Text(no_wrap=True, overflow="fold")
    t.append("".join(buf), style="#555555")
    return t


def _tick_positions(n: int, width: int, max_ticks: int = 8) -> list[tuple[int, int]]:
    """Return (col_pixel, sample_index) pairs for axis tick labels."""
    if n <= 1:
        return [(0, 0)]
    step = max(1, n // min(max_ticks, width // 4))
    positions = []
    for idx in range(0, n, step):
        col = int(idx * width / n)
        positions.append((col, idx))
    # Always include last sample
    last_col = width - len(str(n))
    if positions[-1][1] != n - 1 and last_col >= 0:
        positions.append((last_col, n - 1))
    return positions


class SimilarityHeatmap(Widget):
    """
    Pairwise cosine similarity matrix, scaled to fill the widget tile.

    Layout (inside border):
        ┌──────────────────────────────┐
        │ row │  matrix cells          │  ← plot_h rows
        │  #  │                        │
        ├─────┴────────────────────────┤
        │     col-axis labels          │  ← 1 row
        │     legend gradient bar      │  ← 1 row
        │     0.0      0.5      1.0    │  ← 1 row
        └──────────────────────────────┘
    """

    BORDER_TITLE = "similarity"

    DEFAULT_CSS = """
    SimilarityHeatmap {
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        height: 100%;
        overflow: hidden;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._matrix: np.ndarray | None = None
        self._n = 0

    def update_matrix(self, matrix: np.ndarray) -> None:
        self._matrix = matrix
        self._n = matrix.shape[0]
        self.border_subtitle = f"n={self._n}"
        self.refresh()

    def render(self) -> RenderResult:
        if self._matrix is None or self._n == 0:
            return Text("waiting for samples…", style="dim #555555")

        h = self.size.height
        w = self.size.width
        n = self._n

        plot_h = max(1, h - _BOTTOM)
        plot_w = max(1, w - _LEFT)

        out = Text(no_wrap=True, overflow="fold")

        # ── row-axis tick interval ────────────────────────────────────────
        row_tick_step = max(1, plot_h // 6)

        # ── matrix rows ──────────────────────────────────────────────────
        for row in range(plot_h):
            i = min(int(row * n / plot_h), n - 1)

            # Left axis label
            if row % row_tick_step == 0:
                out.append(f"{i + 1:>3} ", style="#555555")
            else:
                out.append("    ")

            # Matrix cells
            for col in range(plot_w):
                j = min(int(col * n / plot_w), n - 1)
                val = float(self._matrix[i, j])
                out.append("█", style=_cell_style(val))

            out.append("\n")

        # ── column-axis labels ────────────────────────────────────────────
        out.append(" " * _LEFT)
        out.append_text(_label_row(n, plot_w))
        out.append("\n")

        # ── legend gradient bar ───────────────────────────────────────────
        out.append(" " * _LEFT)
        for col in range(plot_w):
            v = col / max(plot_w - 1, 1)
            out.append("█", style=_cell_style(2 * v - 1))  # [-1,1] → ramp
        out.append("\n")

        # ── legend value labels ───────────────────────────────────────────
        out.append(" " * _LEFT)
        lo_lbl = "0.0"
        mid_lbl = "0.5"
        hi_lbl = "1.0"
        mid_pos = (plot_w - len(mid_lbl)) // 2
        hi_pos = plot_w - len(hi_lbl)
        legend_line = [" "] * plot_w
        for k, ch in enumerate(lo_lbl):
            legend_line[k] = ch
        for k, ch in enumerate(mid_lbl):
            if mid_pos + k < plot_w:
                legend_line[mid_pos + k] = ch
        for k, ch in enumerate(hi_lbl):
            if hi_pos + k < plot_w:
                legend_line[hi_pos + k] = ch
        out.append("".join(legend_line), style="#555555")

        return out
