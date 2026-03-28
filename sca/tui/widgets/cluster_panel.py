"""Cluster panel widget — clickable cluster list."""

from __future__ import annotations

from textual.app import RenderResult
from textual.containers import ScrollableContainer
from textual.widget import Widget
from rich.text import Text

from sca.core.clustering import Cluster


class ClusterItem(Widget):
    """A single clickable cluster row."""

    DEFAULT_CSS = """
    ClusterItem {
        height: auto;
        padding: 0 1;
        border-bottom: solid #1a1a1a;
    }
    ClusterItem:hover {
        background: #1a1a1a;
    }
    """

    def __init__(self, cluster: Cluster, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cluster = cluster

    def render(self) -> RenderResult:
        c = self._cluster
        summary = c.summary or "—"
        # Truncate for display; full text available in modal
        if len(summary) > 60:
            summary = summary[:57] + "…"
        t = Text(no_wrap=True)
        t.append(f"{c.id:>2}", style="#555555")
        t.append("  ")
        t.append(f"{len(c.members):>3}", style="white")
        t.append("  ")
        t.append(summary, style="#cccccc")
        return t

    def on_click(self) -> None:
        from sca.tui.widgets.cluster_modal import ClusterModal
        self.app.push_screen(ClusterModal(self._cluster))


class ClusterPanel(ScrollableContainer):

    BORDER_TITLE = "clusters"

    DEFAULT_CSS = """
    ClusterPanel {
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        height: 100%;
        overflow-y: scroll;
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
        # Remove old items and remount fresh ones
        for child in self.query(ClusterItem):
            child.remove()
        for cluster in clusters:
            self.mount(ClusterItem(cluster))
