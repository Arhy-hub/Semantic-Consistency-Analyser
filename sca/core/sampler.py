"""Async parallel sampling with BatchBackend dispatch."""

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
    Sample the backend N times in parallel.

    If the backend implements BatchBackend, uses batch_complete for efficiency.
    Otherwise, spawns n concurrent tasks via asyncio.to_thread.
    """
    if isinstance(backend, BatchBackend):
        results = await asyncio.to_thread(
            backend.batch_complete, [prompt] * n, **kwargs
        )
        return list(results)
    else:
        tasks = [
            asyncio.create_task(asyncio.to_thread(backend, prompt, **kwargs))
            for _ in range(n)
        ]
        return list(await asyncio.gather(*tasks))


async def sample_stream(
    backend: Callable,
    prompt: str,
    n: int,
    **kwargs: Any,
) -> AsyncIterator[tuple[int, str]]:
    """
    Stream samples as they complete.

    Yields (index, text) tuples as each sample finishes.
    Useful for TUI live updates.
    """
    if isinstance(backend, BatchBackend):
        # Batch backends don't stream; yield all at once
        results = await asyncio.to_thread(
            backend.batch_complete, [prompt] * n, **kwargs
        )
        for i, text in enumerate(results):
            yield i, text
        return

    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

    async def _run_one(index: int) -> None:
        text = await asyncio.to_thread(backend, prompt, **kwargs)
        await queue.put((index, text))

    tasks = [asyncio.create_task(_run_one(i)) for i in range(n)]

    for _ in range(n):
        idx, text = await queue.get()
        yield idx, text

    # Ensure all tasks are cleaned up
    await asyncio.gather(*tasks, return_exceptions=True)
