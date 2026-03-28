"""Metrics panel widget — displays current consistency metrics."""

from __future__ import annotations

from textual.widget import Widget
from textual.app import RenderResult
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from sca.core.metrics import Metrics


def _bar(value: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar for a [0, 1] value."""
    filled = int(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


class MetricsPanel(Widget):
    """
    Displays computed semantic consistency metrics.

    Updates as metrics are recomputed after each new sample.
    """

    DEFAULT_CSS = """
    MetricsPanel {
        border: solid $primary;
        height: 100%;
        overflow: hidden;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metrics: Metrics | None = None
        self._sample_count: int = 0

    def update_metrics(self, metrics: Metrics, sample_count: int = 0) -> None:
        """Update displayed metrics."""
        self._metrics = metrics
        self._sample_count = sample_count
        self.refresh()

    def render(self) -> RenderResult:
        if self._metrics is None:
            return Panel(
                "[dim]Waiting for samples...[/dim]",
                title="Metrics",
                border_style="blue",
            )

        m = self._metrics
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold cyan", min_width=28)
        table.add_column("Value", style="white")
        table.add_column("Bar", style="green")

        table.add_row(
            "Samples",
            str(self._sample_count),
            "",
        )
        table.add_row(
            "Mean Pairwise Similarity",
            f"{m.mean_pairwise_similarity:.4f}",
            _bar(m.mean_pairwise_similarity),
        )

        # Entropy: normalize by log(cluster_count) for display
        import math
        max_entropy = math.log(max(m.cluster_count, 2))
        norm_entropy = m.semantic_entropy / max_entropy if max_entropy > 0 else 0.0
        table.add_row(
            "Semantic Entropy",
            f"{m.semantic_entropy:.4f}",
            _bar(norm_entropy),
        )
        table.add_row(
            "Cluster Count",
            str(m.cluster_count),
            "",
        )
        table.add_row(
            "Silhouette Score",
            f"{m.silhouette_score:.4f}",
            _bar((m.silhouette_score + 1) / 2),  # [-1,1] → [0,1]
        )
        table.add_row(
            "Centroid Dist. Variance",
            f"{m.centroid_distance_variance:.6f}",
            "",
        )
        if m.entailment_rate is not None:
            table.add_row(
                "Entailment Rate",
                f"{m.entailment_rate:.4f}",
                _bar(m.entailment_rate),
            )

        return Panel(table, title="Metrics", border_style="blue")
