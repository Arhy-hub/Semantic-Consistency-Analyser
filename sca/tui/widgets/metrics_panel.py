"""Metrics panel widget."""

from __future__ import annotations

import math

from textual.widget import Widget
from textual.app import RenderResult
from rich.table import Table
from rich.text import Text

from sca.core.metrics import Metrics

BAR_WIDTH = 16

_SPARK = "▁▂▃▄▅▆▇█"


def _bar(value: float) -> Text:
    """Greyscale progress bar: filled portion in white, empty in dark."""
    filled = int(max(0.0, min(1.0, value)) * BAR_WIDTH)
    t = Text(no_wrap=True)
    t.append("█" * filled, style="white")
    t.append("░" * (BAR_WIDTH - filled), style="#333333")
    return t


def _sparkline(values: list[float], width: int) -> Text:
    if not values:
        return Text("")
    data = values[-width:]
    lo, hi = min(data), max(data)
    t = Text(no_wrap=True)
    for v in data:
        idx = int((v - lo) / (hi - lo) * (len(_SPARK) - 1)) if hi > lo else 0
        t.append(_SPARK[idx], style="#00d7d7")
    return t


class MetricsPanel(Widget):

    BORDER_TITLE = "metrics"

    DEFAULT_CSS = """
    MetricsPanel {
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        height: 100%;
        overflow: hidden;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metrics: Metrics | None = None
        self._sample_count: int = 0
        self._entropy_history: list[float] = []

    def update_metrics(self, metrics: Metrics, sample_count: int = 0, entropy_history: list[float] | None = None) -> None:
        self._metrics = metrics
        self._sample_count = sample_count
        if entropy_history is not None:
            self._entropy_history = entropy_history
        self.refresh()

    def render(self) -> RenderResult:
        if self._metrics is None:
            return Text("waiting for samples…", style="dim #555555")

        m = self._metrics

        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("label", style="#888888", min_width=26)
        table.add_column("value", style="white", min_width=8)
        table.add_column("bar")

        table.add_row("samples collected", str(self._sample_count), Text(""))

        table.add_row(
            "mean pairwise similarity",
            f"{m.mean_pairwise_similarity:.4f}",
            _bar(m.mean_pairwise_similarity),
        )

        max_entropy = math.log(max(m.cluster_count, 2))
        norm_entropy = m.semantic_entropy / max_entropy if max_entropy > 0 else 0.0
        table.add_row(
            "semantic entropy",
            f"{m.semantic_entropy:.4f}",
            _bar(norm_entropy),
        )

        # Sparkline row for entropy history
        table.add_row(
            "  entropy history",
            "",
            _sparkline(self._entropy_history, BAR_WIDTH * 2),
        )

        # cluster count: bar fills relative to sample count (1 cluster → empty, n clusters → full)
        max_clusters = max(self._sample_count, 1)
        table.add_row(
            "cluster count",
            str(m.cluster_count),
            _bar(m.cluster_count / max_clusters),
        )

        table.add_row(
            "silhouette score",
            f"{m.silhouette_score:.4f}",
            _bar((m.silhouette_score + 1) / 2),
        )

        # centroid dist. variance: unbounded — squash with tanh
        table.add_row(
            "centroid dist. variance",
            f"{m.centroid_distance_variance:.6f}",
            _bar(math.tanh(m.centroid_distance_variance * 10)),
        )

        if m.entailment_rate is not None:
            table.add_row(
                "entailment rate",
                f"{m.entailment_rate:.4f}",
                _bar(m.entailment_rate),
            )

        return table
