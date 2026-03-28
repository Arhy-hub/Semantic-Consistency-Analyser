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
        min_samples=1,          # every point belongs to a cluster; no noise
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


def cluster_by_entailment(
    samples: list[str],
    embeddings: np.ndarray | None = None,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> list[Cluster]:
    """
    Build meaning classes using bidirectional NLI entailment (Kuhn et al. 2023).

    Two samples belong to the same meaning class if and only if they mutually
    entail each other. Transitivity is handled via union-find so a whole chain
    of mutually-entailing samples collapses into one class.

    This is the clustering method required by the *semantic entropy* paper —
    embedding-based clusters (HDBSCAN) are only an approximation.

    Requires sentence-transformers with cross-encoder support.
    """
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for NLI entailment clustering. "
            "Install with: pip install sentence-transformers"
        ) from e

    n = len(samples)
    if n < 2:
        centroid = embeddings[0] if embeddings is not None and len(embeddings) else np.zeros(1)
        return [Cluster(id=0, members=list(samples), centroid=centroid)]

    model = CrossEncoder(model_name)

    # Build all ordered pairs for a single batched inference call
    pairs: list[list[str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append([samples[i], samples[j]])  # forward
            pairs.append([samples[j], samples[i]])  # backward

    scores = model.predict(pairs)
    # scores shape: (2 * n*(n-1)/2, 3); NLI labels: 0=contradiction, 1=neutral, 2=entailment
    label_array = scores.argmax(axis=1)

    # Union-find ──────────────────────────────────────────────────────────────
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            fwd = int(label_array[k])
            bwd = int(label_array[k + 1])
            if fwd == 2 and bwd == 2:  # bidirectional entailment
                union(i, j)
            k += 2

    # Build Cluster objects from union-find roots ─────────────────────────────
    class_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        class_map.setdefault(root, []).append(i)

    clusters: list[Cluster] = []
    for cid, indices in enumerate(sorted(class_map.values(), key=len, reverse=True)):
        member_texts = [samples[i] for i in indices]
        if embeddings is not None and len(embeddings) == n:
            centroid = embeddings[indices].mean(axis=0)
        else:
            centroid = np.zeros(1)
        clusters.append(Cluster(id=cid, members=member_texts, centroid=centroid))

    return clusters


def umap_reduce(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce embeddings to n_components dimensions using UMAP.

    Optional dependency — requires umap-learn.
    Returns 2D projection as shape (n, n_components).
    """
    if len(embeddings) < 3:
        raise ValueError(
            f"umap_reduce requires at least 3 embeddings, got {len(embeddings)}."
        )

    try:
        import umap  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP reduction. "
            "Install with: pip install sca[umap]"
        ) from e

    n_neighbors = min(len(embeddings) - 1, 15)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    return reducer.fit_transform(embeddings)
