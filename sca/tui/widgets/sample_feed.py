"""Sample feed widget — shows LLM samples as they arrive."""

from __future__ import annotations

from textual.app import ComposeResult, RenderResult
from textual.widget import Widget
from textual.containers import ScrollableContainer
from rich.text import Text


class SampleItem(Widget):
    """A single clickable sample entry."""

    DEFAULT_CSS = """
    SampleItem {
        height: auto;
        padding: 0 1;
        border-bottom: solid #1a1a1a;
        color: #cccccc;
    }
    SampleItem:hover {
        background: #1a1a1a;
        color: white;
    }
    """

    def __init__(self, index: int, full_text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._index = index
        self._full_text = full_text

    def render(self) -> RenderResult:
        display = self._full_text[:240] + "…" if len(self._full_text) > 240 else self._full_text
        display = display.replace("\n", " ").strip()
        t = Text(no_wrap=False)
        t.append(f"{self._index + 1:>3}  ", style="#555555")
        t.append(display)
        return t

    def on_click(self) -> None:
        from sca.tui.widgets.sample_modal import SampleModal
        self.app.push_screen(SampleModal(self._index, self._full_text))


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
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sample_count = 0

    def add_sample(self, index: int, text: str) -> None:
        self._sample_count += 1
        self.border_subtitle = str(self._sample_count)
        item = SampleItem(index, text)
        self.mount(item)
        self.scroll_end(animate=False)
