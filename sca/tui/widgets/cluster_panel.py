"""Cluster panel widget."""

from __future__ import annotations

from textual.widget import Widget
from textual.app import RenderResult
from rich.table import Table
from rich.text import Text

from sca.core.clustering import Cluster


class ClusterPanel(Widget):

    BORDER_TITLE = "clusters"

    DEFAULT_CSS = """
    ClusterPanel {
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
        self._clusters: list[Cluster] = []

    def update_clusters(self, clusters: list[Cluster]) -> None:
        self._clusters = clusters
        total = sum(len(c.members) for c in clusters)
        self.border_subtitle = f"{len(clusters)} clusters · {total} samples"
        self.refresh()

    def render(self) -> RenderResult:
        if not self._clusters:
            return Text("waiting for enough samples…", style="dim #555555")

        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("n", style="#555555", min_width=3)
        table.add_column("size", style="white", min_width=4)
        table.add_column("summary", style="#cccccc")

        for cluster in self._clusters[:30]:
            summary = cluster.summary or "—"
            if len(summary) > 55:
                summary = summary[:52] + "…"
            table.add_row(str(cluster.id), str(len(cluster.members)), summary)

        return table
