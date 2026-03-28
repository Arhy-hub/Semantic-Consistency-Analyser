"""Cluster bar chart widget — member counts per cluster with entropy annotation."""

from __future__ import annotations

import math

import numpy as np
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text

from sca.core.clustering import Cluster

_BAR_COLOR = "#555555"
_BLOCKS = " ▁▂▃▄▅▆▇█"   # 9 levels, index 0 = empty


def _entropy(clusters: list[Cluster]) -> float:
    total = sum(len(c.members) for c in clusters)
    if total == 0:
        return 0.0
    h = 0.0
    for c in clusters:
        p = len(c.members) / total
        if p > 0:
            h -= p * math.log(p)
    return h


class ClusterBars(Widget):
    """
    Vertical bar chart — one bar per cluster, height ∝ member count.

    Layout:
        count axis │  bars
                   ├──────────────────
                     cluster labels
                     entropy annotation
    """

    BORDER_TITLE = "cluster distribution"

    DEFAULT_CSS = """
    ClusterBars {
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
        self._clusters: list[Cluster] = []

    def update_clusters(self, clusters: list[Cluster]) -> None:
        self._clusters = clusters
        n_total = sum(len(c.members) for c in clusters)
        h = _entropy(clusters)
        self.border_subtitle = f"SE={h:.2f}  n={n_total}"
        self.refresh()

    def render(self) -> RenderResult:
        if not self._clusters:
            return Text("waiting for clusters…", style="dim #555555")

        h = self.size.height
        w = self.size.width
        clusters = self._clusters
        k = len(clusters)

        _LEFT   = 4   # y-axis label width: " 99 "
        _BOTTOM = 2   # cluster id row + entropy row

        plot_h = max(1, h - _BOTTOM)
        plot_w = max(1, w - _LEFT)

        max_count = max(len(c.members) for c in clusters)
        bar_w = max(1, plot_w // k)

        # ── build grid ────────────────────────────────────────────────────
        # grid[row][col] = (char, style)
        grid: list[list[tuple[str, str]]] = [
            [(" ", "#0a0a0a")] * plot_w for _ in range(plot_h)
        ]

        for ci, cluster in enumerate(clusters):
            col_start = ci * bar_w
            col_end   = min(col_start + bar_w, plot_w)
            if col_start >= plot_w:
                break

            fill_rows = int(len(cluster.members) / max_count * plot_h)
            for row in range(plot_h):
                # row 0 = top, row plot_h-1 = bottom
                depth = plot_h - 1 - row   # depth from bottom (0 = bottom row)
                if depth < fill_rows:
                    char = "█"
                elif depth == fill_rows:
                    # partial block for smooth top
                    frac = (len(cluster.members) / max_count * plot_h) - fill_rows
                    char = _BLOCKS[int(frac * 8)]
                else:
                    char = " "

                if char != " ":
                    for col in range(col_start, col_end):
                        grid[row][col] = (char, _BAR_COLOR)

        # ── y-axis labels ─────────────────────────────────────────────────
        out = Text(no_wrap=True, overflow="fold")

        # Decide which rows to label (top, mid, bottom)
        label_rows: dict[int, str] = {
            0:            str(max_count),
            plot_h // 2:  str(max_count // 2),
            plot_h - 1:   "0",
        }

        for row in range(plot_h):
            lbl = label_rows.get(row, "")
            out.append(f"{lbl:>3} ", style="#555555")
            for col in range(plot_w):
                char, style = grid[row][col]
                out.append(char, style=style)
            out.append("\n")

        # ── cluster id axis ───────────────────────────────────────────────
        out.append(" " * _LEFT)
        id_row = [" "] * plot_w
        for ci, cluster in enumerate(clusters):
            col_mid = ci * bar_w + bar_w // 2
            lbl = str(cluster.id)
            start = col_mid - len(lbl) // 2
            for ki, ch in enumerate(lbl):
                if 0 <= start + ki < plot_w:
                    id_row[start + ki] = ch
        out.append("".join(id_row), style="#555555")
        out.append("\n")

        # ── entropy annotation ────────────────────────────────────────────
        out.append(" " * _LEFT)
        h_val   = _entropy(clusters)
        n_total = sum(len(c.members) for c in clusters)
        # max entropy = log(k) when all clusters equal size
        h_max   = math.log(k) if k > 1 else 1.0
        bar_len = max(0, min(plot_w - 18, plot_w - 18))
        filled  = int(h_val / h_max * bar_len) if h_max > 0 else 0
        out.append("SE ", style="#555555")
        out.append("█" * filled, style="#00d7d7")
        out.append("░" * (bar_len - filled), style="#333333")
        out.append(f"  {h_val:.3f} / {h_max:.3f}", style="#555555")

        return out
