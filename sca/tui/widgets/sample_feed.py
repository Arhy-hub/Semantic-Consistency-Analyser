"""Sample feed widget — shows LLM samples as they arrive."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Label, Log
from textual.containers import ScrollableContainer


class SampleFeed(ScrollableContainer):
    """
    Live-streaming feed of LLM samples.

    Each sample is displayed as a truncated block, newest at the bottom.
    """

    DEFAULT_CSS = """
    SampleFeed {
        border: solid $primary;
        height: 100%;
        overflow-y: scroll;
        padding: 0 1;
    }
    SampleFeed Label.feed-header {
        color: $accent;
        text-style: bold;
        padding: 0 0 1 0;
    }
    SampleFeed .sample-item {
        border-bottom: dashed $surface;
        padding: 0 0 1 0;
        margin-bottom: 1;
        color: $text;
    }
    SampleFeed .sample-index {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sample_count = 0

    def compose(self) -> ComposeResult:
        yield Label("Sample Feed", classes="feed-header")

    def add_sample(self, index: int, text: str) -> None:
        """Add a new sample to the feed."""
        self._sample_count += 1
        # Truncate long samples
        display_text = text[:300] + "..." if len(text) > 300 else text
        display_text = display_text.replace("\n", " ").strip()

        label_text = f"[{index + 1}] {display_text}"
        label = Label(label_text, classes="sample-item")
        self.mount(label)
        self.scroll_end(animate=False)

    def clear_samples(self) -> None:
        """Remove all sample items."""
        for child in self.query(".sample-item"):
            child.remove()
        self._sample_count = 0
