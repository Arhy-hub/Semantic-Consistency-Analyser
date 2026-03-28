"""Concurrent and batch sampling with optional streaming."""

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
    Sample the backend N times concurrently.

    If the backend implements BatchBackend, uses batch_complete instead.
    """
    if isinstance(backend, BatchBackend):
        results = await asyncio.to_thread(
            backend.batch_complete, [prompt] * n, **kwargs
        )
        return list(results)

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
    Yield (index, text) as each concurrent request completes.

    Results are yielded in completion order, not dispatch order,
    so the TUI updates as soon as each sample arrives.
    """
    if isinstance(backend, BatchBackend):
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

    await asyncio.gather(*tasks, return_exceptions=True)
