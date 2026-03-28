"""Sequential and batch sampling with optional streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Callable

from sca.core.backends.protocol import BatchBackend


async def sample(
    backend: Callable,
    prompt: str,
    n: int,
    **kwargs: Any,
) -> list[str]:
    """
    Sample the backend N times sequentially, one request at a time.

    If the backend implements BatchBackend, uses batch_complete instead.
    """
    if isinstance(backend, BatchBackend):
        results = await asyncio.to_thread(
            backend.batch_complete, [prompt] * n, **kwargs
        )
        return list(results)

    results = []
    for _ in range(n):
        text = await asyncio.to_thread(backend, prompt, **kwargs)
        results.append(text)
    return results


async def sample_stream(
    backend: Callable,
    prompt: str,
    n: int,
    **kwargs: Any,
) -> AsyncIterator[tuple[int, str]]:
    """
    Yield (index, text) one at a time as each sample completes.

    Requests are made sequentially so each result is yielded immediately,
    allowing the TUI to update after every sample.
    """
    if isinstance(backend, BatchBackend):
        results = await asyncio.to_thread(
            backend.batch_complete, [prompt] * n, **kwargs
        )
        for i, text in enumerate(results):
            yield i, text
        return

    for i in range(n):
        text = await asyncio.to_thread(backend, prompt, **kwargs)
        yield i, text
