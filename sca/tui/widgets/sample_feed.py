"""Sample feed widget — shows LLM samples as they arrive."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Label
from textual.containers import ScrollableContainer


class SampleFeed(ScrollableContainer):
    """Live-streaming feed of LLM samples, newest at the bottom."""

    BORDER_TITLE = "samples"

    DEFAULT_CSS = """
    SampleFeed {
        border: solid #333333;
        border-title-color: #00d7d7;
        border-title-style: bold;
        height: 100%;
        overflow-y: scroll;
        padding: 0 1;
    }
    SampleFeed Label.sample-item {
        color: #cccccc;
        border-bottom: solid #222222;
        padding: 0 0 1 0;
        margin-bottom: 1;
    }
    SampleFeed Label.sample-index {
        color: #00d7d7;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sample_count = 0

    def compose(self) -> ComposeResult:
        return iter([])

    def add_sample(self, index: int, text: str) -> None:
        self._sample_count += 1
        self.border_subtitle = str(self._sample_count)
        display = text[:280] + "…" if len(text) > 280 else text
        display = display.replace("\n", " ").strip()
        label = Label(f"[#555555]{index + 1:>3}[/]  {display}", classes="sample-item")
        self.mount(label)
        self.scroll_end(animate=False)
