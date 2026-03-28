"""Sentence-transformers embedding wrapper."""

from __future__ import annotations

import numpy as np


class Embedder:
    """
    Wraps sentence_transformers.SentenceTransformer for embedding text.

    Default model: all-MiniLM-L6-v2 (fast, local, no API key needed).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns shape (n, embedding_dim).
        """
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Returns shape (embedding_dim,).
        """
        return self.embed([text])[0]

    def __repr__(self) -> str:
        return f"Embedder(model_name={self.model_name!r})"
