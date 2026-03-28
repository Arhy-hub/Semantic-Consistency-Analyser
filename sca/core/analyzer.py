"""SemanticConsistencyAnalyzer — the main public class."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from sca.core.backends.protocol import BatchBackend
from sca.core.clustering import Cluster, cluster_embeddings
from sca.core.embedder import Embedder
from sca.core.metrics import (
    Metrics,
    centroid_distance_variance,
    compute_similarity_matrix,
    mean_pairwise_similarity,
    semantic_entropy,
    silhouette,
    entailment_rate as compute_entailment_rate,
)
from sca.core.sampler import sample as async_sample


@dataclass
class Results:
    """Full results from a semantic consistency analysis run."""

    samples: list[str]
    embeddings: np.ndarray          # shape (n, embedding_dim)
    similarity_matrix: np.ndarray   # shape (n, n), pairwise cosine
    metrics: Metrics
    clusters: list[Cluster]
    config: dict                    # serializable config snapshot
    timestamp: str                  # ISO format

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict. Numpy arrays become lists."""
        return {
            "samples": self.samples,
            "embeddings": self.embeddings.tolist(),
            "similarity_matrix": self.similarity_matrix.tolist(),
            "metrics": {
                "mean_pairwise_similarity": self.metrics.mean_pairwise_similarity,
                "semantic_entropy": self.metrics.semantic_entropy,
                "cluster_count": self.metrics.cluster_count,
                "silhouette_score": self.metrics.silhouette_score,
                "centroid_distance_variance": self.metrics.centroid_distance_variance,
                "entailment_rate": self.metrics.entailment_rate,
            },
            "clusters": [
                {
                    "id": c.id,
                    "members": c.members,
                    "centroid": c.centroid.tolist(),
                    "summary": c.summary,
                }
                for c in self.clusters
            ],
            "config": self.config,
            "timestamp": self.timestamp,
        }


class SemanticConsistencyAnalyzer:
    """
    Analyze the semantic consistency of LLM outputs.

    Usage:
        results = SemanticConsistencyAnalyzer(
            model="claude-sonnet-4-6",
            prompt="What causes inflation?",
            n=20,
            temperature=0.9,
        ).run()
    """

    def __init__(
        self,
        prompt: str,
        model: str | None = None,
        backend: Callable | None = None,
        n: int = 20,
        temperature: float = 0.9,
        embedding_model: str = "all-MiniLM-L6-v2",
        nli: bool = False,
        convergence_threshold: float | None = None,
        min_cluster_size: int = 3,
    ) -> None:
        if model is None and backend is None:
            from sca.config import settings  # noqa: PLC0415
            model = settings.default_model

        self.prompt = prompt
        self.model = model
        self._backend = backend
        self.n = n
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.nli = nli
        self.convergence_threshold = convergence_threshold
        self.min_cluster_size = min_cluster_size

    def _get_backend(self) -> Callable:
        if self._backend is not None:
            return self._backend
        from sca.core.backends.litellm import LiteLLMBackend  # noqa: PLC0415
        return LiteLLMBackend(self.model, temperature=self.temperature)

    def _build_config(self) -> dict:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "n": self.n,
            "temperature": self.temperature,
            "embedding_model": self.embedding_model,
            "nli": self.nli,
            "convergence_threshold": self.convergence_threshold,
            "min_cluster_size": self.min_cluster_size,
        }

    def run(self) -> Results:
        """Run synchronously (wraps run_async via asyncio.run)."""
        return asyncio.run(self.run_async())

    async def run_async(self, on_sample: Callable | None = None) -> Results:
        """
        Run asynchronously.

        on_sample: optional callback(index, sample_text, partial_results_or_none)
        """
        backend = self._get_backend()
        embedder = Embedder(self.embedding_model)

        # ── Sampling ─────────────────────────────────────────────────────
        samples: list[str] = []
        all_embeddings: list[np.ndarray] = []

        # Convergence tracking
        entropy_window: list[float] = []
        converged = False

        if on_sample is not None:
            # Stream samples as they arrive for live TUI updates
            from sca.core.sampler import sample_stream  # noqa: PLC0415

            async for idx, text in sample_stream(backend, self.prompt, self.n,
                                                  temperature=self.temperature):
                samples.append(text)
                emb = await asyncio.to_thread(embedder.embed_one, text)
                all_embeddings.append(emb)

                await asyncio.to_thread(on_sample, idx, text, None)

                # Convergence check
                if self.convergence_threshold is not None and len(samples) >= self.min_cluster_size:
                    partial_embs = np.stack(all_embeddings)
                    partial_clusters = await asyncio.to_thread(
                        cluster_embeddings, partial_embs, samples, self.min_cluster_size
                    )
                    from sca.core.metrics import semantic_entropy as se  # noqa: PLC0415
                    current_entropy = se(partial_clusters)
                    entropy_window.append(current_entropy)
                    if len(entropy_window) > 3:
                        entropy_window.pop(0)
                    if len(entropy_window) == 3:
                        deltas = [abs(entropy_window[i+1] - entropy_window[i])
                                  for i in range(2)]
                        if all(d < self.convergence_threshold for d in deltas):
                            converged = True
                            break
        else:
            # Sample all at once
            raw_samples = await async_sample(
                backend, self.prompt, self.n, temperature=self.temperature
            )
            samples = list(raw_samples)

            # Check for convergence during embedding phase if threshold set
            if self.convergence_threshold is not None:
                # Embed incrementally and check convergence
                for i, text in enumerate(samples):
                    emb = await asyncio.to_thread(embedder.embed_one, text)
                    all_embeddings.append(emb)
                    if i >= self.min_cluster_size:
                        partial_embs = np.stack(all_embeddings)
                        partial_clusters = await asyncio.to_thread(
                            cluster_embeddings, partial_embs, samples[:i+1],
                            self.min_cluster_size
                        )
                        current_entropy = semantic_entropy(partial_clusters)
                        entropy_window.append(current_entropy)
                        if len(entropy_window) > 3:
                            entropy_window.pop(0)
                        if len(entropy_window) == 3:
                            deltas = [abs(entropy_window[j+1] - entropy_window[j])
                                      for j in range(2)]
                            if all(d < self.convergence_threshold for d in deltas):
                                samples = samples[:i+1]
                                converged = True
                                break
            else:
                all_embeddings_arr = await asyncio.to_thread(embedder.embed, samples)
                all_embeddings = list(all_embeddings_arr)

        if not all_embeddings:
            raise RuntimeError("No samples were collected.")

        embeddings = np.stack(all_embeddings)

        # ── Similarity matrix ─────────────────────────────────────────────
        sim_matrix = compute_similarity_matrix(embeddings)

        # ── Clustering ────────────────────────────────────────────────────
        clusters = await asyncio.to_thread(
            cluster_embeddings, embeddings, samples, self.min_cluster_size
        )

        # ── Cluster labels for silhouette ─────────────────────────────────
        labels = np.zeros(len(samples), dtype=int)
        for cluster in clusters:
            for member in cluster.members:
                for i, s in enumerate(samples):
                    if s == member and labels[i] == 0:
                        labels[i] = cluster.id
                        break

        # ── Metrics ───────────────────────────────────────────────────────
        mps = mean_pairwise_similarity(sim_matrix)
        s_entropy = semantic_entropy(clusters)
        cdv = centroid_distance_variance(embeddings)
        sil = silhouette(embeddings, labels)

        ent_rate: float | None = None
        if self.nli:
            ent_rate = await asyncio.to_thread(compute_entailment_rate, samples)

        metrics = Metrics(
            mean_pairwise_similarity=mps,
            semantic_entropy=s_entropy,
            cluster_count=len(clusters),
            silhouette_score=sil,
            centroid_distance_variance=cdv,
            entailment_rate=ent_rate,
        )

        # ── Cluster auto-summaries ─────────────────────────────────────────
        await self._summarize_clusters(clusters, backend)

        timestamp = datetime.now(timezone.utc).isoformat()
        config = self._build_config()
        config["converged_early"] = converged

        return Results(
            samples=samples,
            embeddings=embeddings,
            similarity_matrix=sim_matrix,
            metrics=metrics,
            clusters=clusters,
            config=config,
            timestamp=timestamp,
        )

    async def _summarize_clusters(
        self, clusters: list[Cluster], backend: Callable
    ) -> None:
        """Generate one-sentence summaries for each cluster via LLM call."""
        if not clusters:
            return

        async def summarize_one(cluster: Cluster) -> None:
            if len(cluster.members) == 0:
                return
            responses_text = "\n---\n".join(cluster.members[:5])  # cap at 5 for brevity
            summary_prompt = (
                f"Summarize these LLM responses in one sentence, "
                f"describing their common theme:\n\n{responses_text}"
            )
            try:
                summary = await asyncio.to_thread(backend, summary_prompt, temperature=0.3)
                cluster.summary = summary.strip()
            except Exception:
                cluster.summary = f"Cluster {cluster.id} ({len(cluster.members)} samples)"

        await asyncio.gather(*[summarize_one(c) for c in clusters])
