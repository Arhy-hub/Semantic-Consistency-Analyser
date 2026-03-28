"""Cluster panel widget — shows semantic clusters with summaries."""

from __future__ import annotations

from textual.widget import Widget
from textual.app import RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sca.core.clustering import Cluster


class ClusterPanel(Widget):
    """
    Displays semantic clusters with member counts and auto-generated summaries.

    Populates once enough samples exist to cluster (min 5 by default).
    """

    DEFAULT_CSS = """
    ClusterPanel {
        border: solid $primary;
        height: 100%;
        overflow: hidden;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._clusters: list[Cluster] = []

    def update_clusters(self, clusters: list[Cluster]) -> None:
        """Update the displayed cluster list."""
        self._clusters = clusters
        self.refresh()

    def render(self) -> RenderResult:
        if not self._clusters:
            return Panel(
                "[dim]Waiting for enough samples to cluster...[/dim]",
                title="Clusters",
                border_style="blue",
            )

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("#", style="bold cyan", min_width=3)
        table.add_column("N", style="yellow", min_width=4)
        table.add_column("Summary", style="white")

        for cluster in self._clusters[:20]:  # show up to 20 clusters
            summary = cluster.summary or "(no summary yet)"
            # Truncate long summaries
            if len(summary) > 60:
                summary = summary[:57] + "..."
            table.add_row(
                str(cluster.id),
                str(len(cluster.members)),
                summary,
            )

        total = sum(len(c.members) for c in self._clusters)
        content = Text()
        content.append(f"Total clusters: {len(self._clusters)}  |  Total samples: {total}\n\n",
                       style="bold")

        panel_content = Text.assemble(
            (f"Total clusters: {len(self._clusters)}  |  Total samples: {total}\n\n",
             "bold dim"),
        )

        from rich.console import Group  # noqa: PLC0415
        return Panel(
            Group(
                Text(
                    f"Total: {len(self._clusters)} clusters, {total} samples",
                    style="bold dim",
                ),
                table,
            ),
            title="Clusters",
            border_style="blue",
        )
