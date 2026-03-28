"""Modal screen showing full details for a single cluster."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Label, Button
from textual.containers import ScrollableContainer, Vertical

from sca.core.clustering import Cluster


class ClusterModal(ModalScreen):

    DEFAULT_CSS = """
    ClusterModal {
        align: center middle;
    }
    ClusterModal > Vertical {
        width: 80%;
        height: 85%;
        background: #111111;
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        padding: 1 2;
    }
    ClusterModal Label.summary {
        color: #00d7d7;
        padding: 0 0 1 0;
    }
    ClusterModal Label.divider {
        color: #333333;
        padding: 0 0 1 0;
    }
    ClusterModal Label.member {
        color: #cccccc;
        border-bottom: solid #1a1a1a;
        padding: 0 0 1 0;
        margin-bottom: 1;
    }
    ClusterModal Label.member-index {
        color: #555555;
    }
    ClusterModal Button {
        margin-top: 1;
        background: #1a1a1a;
        border: solid #333333;
        color: #888888;
        width: 10;
    }
    """

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, cluster: Cluster) -> None:
        super().__init__()
        self._cluster = cluster

    def compose(self) -> ComposeResult:
        c = self._cluster
        with Vertical() as v:
            v.border_title = f"cluster {c.id}  ·  {len(c.members)} samples"
            with ScrollableContainer():
                yield Label(c.summary or "—", classes="summary")
                yield Label("─" * 40, classes="divider")
                for i, member in enumerate(c.members):
                    yield Label(f"[#555555]{i + 1:>2}[/]  {member}", classes="member")
            yield Button("close [esc]", id="close")

    def on_button_pressed(self) -> None:
        self.dismiss()
