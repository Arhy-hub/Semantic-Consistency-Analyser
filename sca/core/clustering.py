"""HDBSCAN clustering + optional UMAP dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    """A semantic cluster of LLM output samples."""

    id: int
    members: list[str]
    centroid: np.ndarray
    summary: str = ""  # populated later by LLM call

    def __repr__(self) -> str:
        return f"Cluster(id={self.id}, n={len(self.members)}, summary={self.summary!r})"


def cluster_embeddings(
    embeddings: np.ndarray,
    samples: list[str],
    min_cluster_size: int = 3,
) -> list[Cluster]:
    """
    Cluster embeddings using HDBSCAN.

    Noise points (label=-1) are each assigned as singleton clusters.
    Returns a list of Cluster objects sorted by size (largest first).
    """
    try:
        import hdbscan  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "hdbscan is required for clustering. Install with: pip install hdbscan"
        ) from e

    n = len(embeddings)
    if n < 2:
        # Can't cluster a single point
        centroid = embeddings[0] if n == 1 else np.zeros(1)
        return [Cluster(id=0, members=list(samples), centroid=centroid)]

    effective_min = min(min_cluster_size, max(2, n // 3))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)

    # If every point is noise, the embeddings may all be identical or very tightly packed.
    # Fall back to treating them as one cluster.
    if (labels == -1).all():
        centroid = embeddings.mean(axis=0)
        return [Cluster(id=0, members=list(samples), centroid=centroid)]

    # Gather named clusters
    cluster_map: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(idx)

    clusters: list[Cluster] = []
    next_id = 0

    # Process non-noise clusters first
    for label, indices in sorted(cluster_map.items()):
        if label == -1:
            continue  # handle noise below
        member_texts = [samples[i] for i in indices]
        centroid = embeddings[indices].mean(axis=0)
        clusters.append(Cluster(id=next_id, members=member_texts, centroid=centroid))
        next_id += 1

    # Each noise point becomes its own singleton cluster
    noise_indices = cluster_map.get(-1, [])
    for idx in noise_indices:
        clusters.append(
            Cluster(
                id=next_id,
                members=[samples[idx]],
                centroid=embeddings[idx].copy(),
            )
        )
        next_id += 1

    # Sort largest first
    clusters.sort(key=lambda c: len(c.members), reverse=True)

    # Re-assign sequential IDs after sort
    for new_id, cluster in enumerate(clusters):
        cluster.id = new_id

    return clusters


def umap_reduce(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce embeddings to n_components dimensions using UMAP.

    Optional dependency — requires umap-learn.
    Returns 2D projection as shape (n, n_components).
    """
    try:
        import umap  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP reduction. "
            "Install with: pip install sca[umap]"
        ) from e

    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)
