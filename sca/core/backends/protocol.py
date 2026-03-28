"""Backend protocols for sca — duck-typed, no hard dependencies."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Minimal protocol: any callable (prompt, **kwargs) -> str."""

    def __call__(self, prompt: str, **kwargs) -> str: ...


@runtime_checkable
class BatchBackend(Protocol):
    """Extended protocol with batch completion support."""

    def __call__(self, prompt: str, **kwargs) -> str: ...

    def batch_complete(self, prompts: list[str], **kwargs) -> list[str]: ...
