"""HuggingFace transformers convenience wrapper — optional dependency."""

from __future__ import annotations


class HFBackend:
    """
    Wraps a HuggingFace text-generation pipeline.

    Implements BatchBackend protocol via batch_complete.
    Transformers is an optional dependency — import is deferred to first call.
    """

    def __init__(self, model_name: str, **pipeline_kwargs) -> None:
        self.model_name = model_name
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    "transformers is required for HFBackend. "
                    "Install with: pip install sca[hf]"
                ) from e
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                **self.pipeline_kwargs,
            )
        return self._pipeline

    def __call__(self, prompt: str, **kwargs) -> str:
        pipe = self._get_pipeline()
        results = pipe(prompt, max_new_tokens=kwargs.get("max_new_tokens", 512), **{
            k: v for k, v in kwargs.items() if k != "max_new_tokens"
        })
        generated = results[0]["generated_text"]
        # Strip the prompt from the output if it's repeated
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated

    def batch_complete(self, prompts: list[str], **kwargs) -> list[str]:
        pipe = self._get_pipeline()
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        extra = {k: v for k, v in kwargs.items() if k != "max_new_tokens"}
        results = pipe(prompts, max_new_tokens=max_new_tokens, **extra)
        outputs = []
        for prompt, result in zip(prompts, results):
            generated = result[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            outputs.append(generated)
        return outputs

    def __repr__(self) -> str:
        return f"HFBackend(model_name={self.model_name!r})"
