"""Modal screen for displaying a full sample text."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Label, Button
from textual.containers import ScrollableContainer, Vertical


class SampleModal(ModalScreen):
    DEFAULT_CSS = """
    SampleModal {
        align: center middle;
    }
    SampleModal > Vertical {
        width: 80%;
        height: 80%;
        background: #111111;
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        padding: 1 2;
    }
    SampleModal ScrollableContainer {
        height: 1fr;
        border: none;
    }
    SampleModal Label.body {
        color: #cccccc;
    }
    SampleModal Button {
        margin-top: 1;
        background: #1a1a1a;
        border: solid #333333;
        color: #888888;
        width: 10;
    }
    """
    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, index: int, text: str) -> None:
        super().__init__()
        self._index = index
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical() as v:
            v.border_title = f"sample {self._index + 1}"
            with ScrollableContainer():
                yield Label(self._text, classes="body")
            yield Button("close [esc]", id="close")

    def on_button_pressed(self) -> None:
        self.dismiss()
