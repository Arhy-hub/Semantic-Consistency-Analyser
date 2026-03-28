"""Similarity heatmap widget — ASCII/Unicode block character rendering."""

from __future__ import annotations

import numpy as np
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text
from rich.panel import Panel


# Unicode block characters mapped to similarity intensity (low → high)
BLOCKS = " ░▒▓█"


def _sim_to_block(value: float) -> str:
    """Map a similarity value in [-1, 1] to a Unicode block character."""
    # Normalize from [-1, 1] to [0, 1]
    normalized = (value + 1.0) / 2.0
    normalized = max(0.0, min(1.0, normalized))
    idx = int(normalized * (len(BLOCKS) - 1))
    return BLOCKS[idx]


def _sim_to_color(value: float) -> str:
    """Map similarity value to a Rich color string."""
    # Normalize from [-1, 1] to [0, 1]
    normalized = (value + 1.0) / 2.0
    normalized = max(0.0, min(1.0, normalized))
    if normalized >= 0.8:
        return "bright_green"
    elif normalized >= 0.6:
        return "green"
    elif normalized >= 0.4:
        return "yellow"
    elif normalized >= 0.2:
        return "red"
    else:
        return "bright_red"


class SimilarityHeatmap(Widget):
    """
    Renders the pairwise cosine similarity matrix as a Unicode block heatmap.

    Updates incrementally as each new sample arrives.
    """

    DEFAULT_CSS = """
    SimilarityHeatmap {
        border: solid $primary;
        height: 100%;
        overflow: hidden;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._matrix: np.ndarray | None = None
        self._n = 0

    def update_matrix(self, matrix: np.ndarray) -> None:
        """Update the similarity matrix and refresh the display."""
        self._matrix = matrix
        self._n = matrix.shape[0]
        self.refresh()

    def render(self) -> RenderResult:
        if self._matrix is None or self._n == 0:
            return Panel(
                "[dim]Waiting for samples...[/dim]",
                title="Similarity Heatmap",
                border_style="blue",
            )

        # Determine how many rows/cols fit in the widget
        # Each cell is 1 character wide + 1 space = 2 chars
        max_dim = min(self._n, 40)  # cap at 40 for readability
        step = max(1, self._n // max_dim)

        lines = Text()
        indices = list(range(0, self._n, step))

        for i in indices:
            for j in indices:
                val = float(self._matrix[i, j])
                block = _sim_to_block(val)
                color = _sim_to_color(val)
                lines.append(block, style=color)
            lines.append("\n")

        return Panel(
            lines,
            title=f"Similarity Heatmap (n={self._n})",
            border_style="blue",
        )
