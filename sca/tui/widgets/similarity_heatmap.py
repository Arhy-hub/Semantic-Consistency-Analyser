"""Similarity heatmap — display-agnostic matrix with centred axes and entropy sidebar."""

from __future__ import annotations

import numpy as np
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text

# ── Layout constants ──────────────────────────────────────────────────────
_LEFT   = 5   # row axis:  "  1 │"
_BOTTOM = 3   # col axis, legend bar, legend values

# ── 2-colour gradient: dark background → accent cyan ─────────────────────
_LO: tuple[int, int, int] = (10,   10,  10)   # #0a0a0a
_HI: tuple[int, int, int] = ( 0,  215, 215)   # #00d7d7

# Pre-compute 256 evenly-spaced steps so every possible terminal colour
# value maps to a distinct, equally-sized band.
_STEPS = 256
_PALETTE: list[str] = []
for _i in range(_STEPS):
    _t = _i / (_STEPS - 1)
    _r = int(_LO[0] + (_HI[0] - _LO[0]) * _t)
    _g = int(_LO[1] + (_HI[1] - _LO[1]) * _t)
    _b = int(_LO[2] + (_HI[2] - _LO[2]) * _t)
    _PALETTE.append(f"#{_r:02x}{_g:02x}{_b:02x}")


def _cell_style(value: float) -> str:
    """Map a similarity value in [-1, 1] to a palette colour."""
    v = max(0.0, min(1.0, (value + 1.0) / 2.0))
    return _PALETTE[int(v * (_STEPS - 1))]



def _row_label_map(n: int, plot_h: int) -> dict[int, str]:
    """
    Map display-row indices to sample-number labels.

    Each label is placed at the midpoint row of its sample's pixel block,
    so it sits visually centred in the block regardless of up/down scaling.
    """
    result: dict[int, str] = {}
    # Aim for ~plot_h/3 labels; never more than n
    step = max(1, n // max(plot_h // 3, 1))
    last_row = -999
    for i in range(0, n, step):
        mid = int((i + 0.5) * plot_h / n)
        if mid - last_row >= 2:
            result[mid] = str(i + 1)
            last_row = mid
    # Always include n (last sample) if there's room
    last_mid = int((n - 0.5) * plot_h / n)
    if last_mid - last_row >= 2 and (n - 1) % step != 0:
        result[last_mid] = str(n)
    return result


def _col_axis(n: int, plot_w: int) -> Text:
    """
    Build the column-axis label row.

    Each label is centred at the midpoint column of its sample's pixel block.
    """
    buf = [" "] * plot_w
    step = max(1, n // max(plot_w // 4, 1))
    last_end = -999
    for i in range(0, n, step):
        mid = int((i + 0.5) * plot_w / n)
        lbl = str(i + 1)
        start = mid - len(lbl) // 2
        end = start + len(lbl)
        if start >= 0 and end <= plot_w and start > last_end:
            for k, ch in enumerate(lbl):
                buf[start + k] = ch
            last_end = end
    # Always include n
    last_mid = int((n - 0.5) * plot_w / n)
    lbl = str(n)
    start = last_mid - len(lbl) // 2
    end = start + len(lbl)
    if start >= 0 and end <= plot_w and start > last_end:
        for k, ch in enumerate(lbl):
            buf[start + k] = ch

    t = Text(no_wrap=True, overflow="fold")
    t.append("".join(buf), style="#555555")
    return t


class SimilarityHeatmap(Widget):
    """
    Display-agnostic n×n matrix heatmap.

    The app decides what measure to pass (cosine sim, cluster agreement, …).
    The measure is chosen via the --measure CLI flag.

    Layout inside border:
        row │  matrix cells
        axis│
        ────┴───────────────────────
             sample axis labels
             legend gradient
             0.0      0.5      1.0
    """

    BORDER_TITLE = "cosine sim"

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

    def update_matrix(self, matrix: np.ndarray, title: str = "cosine sim") -> None:
        self._matrix = matrix
        self._n = matrix.shape[0]
        self.border_title = title
        self.border_subtitle = f"n={self._n}"
        self.refresh()

    # ── render ─────────────────────────────────────────────────────────────

    def render(self) -> RenderResult:
        if self._matrix is None or self._n == 0:
            return Text("waiting for samples…", style="dim #555555")

        h = self.size.height
        w = self.size.width
        n = self._n

        plot_h = max(1, h - _BOTTOM)
        plot_w = max(1, w - _LEFT)

        row_labels = _row_label_map(n, plot_h)

        out = Text(no_wrap=True, overflow="fold")

        # ── matrix rows ───────────────────────────────────────────────────
        for row in range(plot_h):
            i = min(int(row * n / plot_h), n - 1)

            # Left axis (3-char label, right-justified, then space + "│")
            lbl = row_labels.get(row, "")
            out.append(f"{lbl:>3} ", style="#555555")
            out.append("│", style="#333333")

            # Matrix cells
            for col in range(plot_w):
                j = min(int(col * n / plot_w), n - 1)
                out.append("█", style=_cell_style(float(self._matrix[i, j])))

            out.append("\n")

        # ── column axis ───────────────────────────────────────────────────
        out.append(" " * _LEFT)
        out.append_text(_col_axis(n, plot_w))
        out.append("\n")

        # ── legend gradient ───────────────────────────────────────────────
        out.append(" " * _LEFT)
        for col in range(plot_w):
            v = col / max(plot_w - 1, 1)
            out.append("█", style=_cell_style(2 * v - 1))
        out.append("\n")

        # ── legend value labels ───────────────────────────────────────────
        out.append(" " * _LEFT)
        lo, mid, hi = "0.0", "0.5", "1.0"
        mid_pos = (plot_w - len(mid)) // 2
        hi_pos  = plot_w - len(hi)
        legend  = [" "] * plot_w
        for k, ch in enumerate(lo):
            legend[k] = ch
        for k, ch in enumerate(mid):
            if mid_pos + k < plot_w:
                legend[mid_pos + k] = ch
        for k, ch in enumerate(hi):
            if hi_pos + k < plot_w:
                legend[hi_pos + k] = ch
        out.append("".join(legend), style="#555555")

        return out
