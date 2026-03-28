"""Pairwise cosine similarity histogram widget."""

from __future__ import annotations

import numpy as np
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text

_BLOCKS = " ▁▂▃▄▅▆▇█"
_N_BINS = 20   # bins across [0, 1]


class SimilarityHistogram(Widget):
    """
    Histogram of off-diagonal pairwise cosine similarity values.

    Peaks near 1.0 → consistent outputs.
    Spread out / bimodal → inconsistent or multi-modal outputs.
    """

    BORDER_TITLE = "similarity distribution"

    DEFAULT_CSS = """
    SimilarityHistogram {
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
        self._values: np.ndarray | None = None   # 1-D array of similarity values

    def update_matrix(self, sim_matrix: np.ndarray) -> None:
        n = sim_matrix.shape[0]
        if n < 2:
            return
        # Upper triangle, excluding diagonal
        idx = np.triu_indices(n, k=1)
        self._values = sim_matrix[idx].astype(float)
        mean = float(self._values.mean())
        self.border_subtitle = f"mean={mean:.3f}  n={len(self._values)}"
        self.refresh()

    def render(self) -> RenderResult:
        if self._values is None or len(self._values) == 0:
            return Text("waiting for samples…", style="dim #555555")

        h = self.size.height
        w = self.size.width

        _LEFT   = 5   # y-axis label width
        _BOTTOM = 2   # x-axis labels + mean line

        plot_h = max(1, h - _BOTTOM)
        plot_w = max(1, w - _LEFT)

        n_bins = min(_N_BINS, plot_w)
        counts, edges = np.histogram(self._values, bins=n_bins, range=(0.0, 1.0))
        max_count = int(counts.max()) if counts.max() > 0 else 1

        # Each bin gets plot_w // n_bins columns; remainder distributed left→right
        bin_widths = [plot_w // n_bins] * n_bins
        for i in range(plot_w % n_bins):
            bin_widths[i] += 1

        # ── build grid ────────────────────────────────────────────────────
        grid: list[list[tuple[str, str]]] = [
            [(" ", "#0a0a0a")] * plot_w for _ in range(plot_h)
        ]

        col = 0
        for bi, (count, bw) in enumerate(zip(counts, bin_widths)):
            fill_rows = int(count / max_count * plot_h)
            frac      = (count / max_count * plot_h) - fill_rows
            partial   = _BLOCKS[int(frac * 8)]

            for row in range(plot_h):
                depth = plot_h - 1 - row
                if depth < fill_rows:
                    char = "█"
                elif depth == fill_rows and partial != " ":
                    char = partial
                else:
                    char = " "
                if char != " ":
                    for c in range(col, col + bw):
                        if c < plot_w:
                            grid[row][c] = (char, "#555555")
            col += bw

        # ── render rows with y-axis ───────────────────────────────────────
        out = Text(no_wrap=True, overflow="fold")

        label_rows: dict[int, str] = {
            0:            str(max_count),
            plot_h // 2:  str(max_count // 2),
            plot_h - 1:   "0",
        }

        for row in range(plot_h):
            lbl = label_rows.get(row, "")
            out.append(f"{lbl:>4} ", style="#555555")
            for c in range(plot_w):
                char, style = grid[row][c]
                out.append(char, style=style)
            out.append("\n")

        # ── x-axis labels: 0.0, 0.5, 1.0 ────────────────────────────────
        out.append(" " * _LEFT)
        lo, mid, hi = "0.0", "0.5", "1.0"
        x_row = [" "] * plot_w
        mid_pos = (plot_w - len(mid)) // 2
        hi_pos  = plot_w - len(hi)
        for k, ch in enumerate(lo):
            x_row[k] = ch
        for k, ch in enumerate(mid):
            if mid_pos + k < plot_w:
                x_row[mid_pos + k] = ch
        for k, ch in enumerate(hi):
            if hi_pos + k < plot_w:
                x_row[hi_pos + k] = ch
        out.append("".join(x_row), style="#555555")
        out.append("\n")

        # ── mean marker ───────────────────────────────────────────────────
        out.append(" " * _LEFT)
        mean = float(self._values.mean())
        mean_col = int(mean * (plot_w - 1))
        marker_row = [" "] * plot_w
        if 0 <= mean_col < plot_w:
            marker_row[mean_col] = "▲"
        lbl = f"{mean:.2f}"
        lbl_start = max(0, min(mean_col - len(lbl) // 2, plot_w - len(lbl)))
        for k, ch in enumerate(lbl):
            if lbl_start + k < plot_w:
                marker_row[lbl_start + k] = ch
        out.append("".join(marker_row), style="#00d7d7")

        return out
