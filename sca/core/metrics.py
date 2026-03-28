"""Metrics for measuring semantic consistency of LLM outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sca.core.clustering import Cluster


@dataclass
class Metrics:
    """All consistency metrics for a single sampling run."""

    mean_pairwise_similarity: float
    semantic_entropy: float
    cluster_count: int
    silhouette_score: float
    centroid_distance_variance: float
    entailment_rate: float | None = None  # None if NLI not enabled

    def __repr__(self) -> str:
        parts = [
            f"mean_sim={self.mean_pairwise_similarity:.3f}",
            f"entropy={self.semantic_entropy:.3f}",
            f"clusters={self.cluster_count}",
            f"silhouette={self.silhouette_score:.3f}",
            f"cdv={self.centroid_distance_variance:.4f}",
        ]
        if self.entailment_rate is not None:
            parts.append(f"entailment={self.entailment_rate:.3f}")
        return f"Metrics({', '.join(parts)})"


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Uses normalized dot product for efficiency.
    Returns shape (n, n) with values in [-1, 1].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


def mean_pairwise_similarity(sim_matrix: np.ndarray) -> float:
    """
    Average of off-diagonal elements of the similarity matrix.

    The diagonal (self-similarity = 1.0) is excluded.
    """
    n = sim_matrix.shape[0]
    if n < 2:
        return 1.0
    # Sum all elements, subtract diagonal sum (n * 1.0)
    total = sim_matrix.sum() - n  # subtract diagonal
    count = n * n - n  # off-diagonal count
    return float(total / count)


def semantic_entropy(clusters: list["Cluster"]) -> float:
    """
    Shannon entropy over the cluster size distribution.

    p_i = cluster_size / total_samples
    H = -sum(p_i * log(p_i))

    Low entropy = samples cluster into few groups = consistent.
    High entropy = many equally-sized clusters = inconsistent.
    """
    if not clusters:
        return 0.0

    total = sum(len(c.members) for c in clusters)
    if total == 0:
        return 0.0

    entropy = 0.0
    for cluster in clusters:
        if not cluster.members:
            continue
        p = len(cluster.members) / total
        if p > 0:
            entropy -= p * math.log(p)

    return float(entropy)


def centroid_distance_variance(embeddings: np.ndarray) -> float:
    """
    Variance of L2 distances from the semantic centroid.

    Measures spread/dispersion of the output distribution.
    """
    if len(embeddings) < 2:
        return 0.0
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return float(distances.var())


def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute sklearn silhouette score.

    Returns 0.0 if there is only one unique cluster label.
    """
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    if len(embeddings) < 2:
        return 0.0

    try:
        return float(silhouette_score(embeddings, labels))
    except Exception:
        return 0.0


def entailment_rate(
    samples: list[str],
    model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> float:
    """
    Compute fraction of sample pairs that entail each other (bidirectional).

    Uses a cross-encoder NLI model. Optional — lazy import.
    This can be slow for large N.
    """
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "sentence-transformers with cross-encoder support is required for NLI. "
            "Install with: pip install sentence-transformers"
        ) from e

    if len(samples) < 2:
        return 1.0

    model = CrossEncoder(model_name)
    pairs = []
    pair_indices = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            # Check both directions
            pairs.append([samples[i], samples[j]])
            pairs.append([samples[j], samples[i]])
            pair_indices.append((i, j))

    if not pairs:
        return 1.0

    scores = model.predict(pairs)
    # NLI labels: 0=contradiction, 1=neutral, 2=entailment
    labels = scores.argmax(axis=1) if hasattr(scores, "argmax") else [
        int(s.argmax()) for s in scores
    ]

    entailment_label = 2
    entailing_pairs = 0
    total_pairs = len(pair_indices)

    for k, _ in enumerate(pair_indices):
        fwd = labels[k * 2]
        bwd = labels[k * 2 + 1]
        if fwd == entailment_label and bwd == entailment_label:
            entailing_pairs += 1

    return float(entailing_pairs / total_pairs) if total_pairs > 0 else 1.0
