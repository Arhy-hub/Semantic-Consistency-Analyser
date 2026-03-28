"""LiteLLM backend — supports OpenAI, Anthropic, Gemini, Ollama, vLLM."""

from __future__ import annotations


class LiteLLMBackend:
    """Wraps litellm.completion as a Backend-compatible callable."""

    def __init__(self, model: str, **default_kwargs) -> None:
        self.model = model
        self.default_kwargs = default_kwargs

    def __call__(self, prompt: str, **kwargs) -> str:
        import litellm  # noqa: PLC0415

        merged = {**self.default_kwargs, **kwargs}
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(model=self.model, messages=messages, **merged)
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"LiteLLMBackend(model={self.model!r})"
