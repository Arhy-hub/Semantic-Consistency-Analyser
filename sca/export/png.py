"""PNG scatter plot export via UMAP + matplotlib."""

from __future__ import annotations

from sca.core.analyzer import Results


def export_png(results: Results, path: str) -> None:
    """
    Create a UMAP scatter plot of the embeddings, colored by cluster.

    Requires umap-learn and matplotlib.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib.cm as cm  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for PNG export. "
            "Install with: pip install sca[export]"
        ) from e

    from sca.core.clustering import umap_reduce  # noqa: PLC0415

    embeddings = results.embeddings
    clusters = results.clusters

    # Build color map: sample index → cluster id
    sample_to_cluster: dict[str, int] = {}
    for cluster in clusters:
        for member in cluster.members:
            if member not in sample_to_cluster:
                sample_to_cluster[member] = cluster.id

    cluster_ids = [sample_to_cluster.get(s, -1) for s in results.samples]

    # UMAP reduction
    if embeddings.shape[0] < 4:
        # Not enough points for UMAP, fall back to PCA-like approach
        from sklearn.decomposition import PCA  # noqa: PLC0415
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(embeddings)
    else:
        coords_2d = umap_reduce(embeddings, n_components=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    n_clusters = len(set(cluster_ids))
    colors = cm.tab20(range(n_clusters))
    color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(sorted(set(cluster_ids)))}

    for i, (x, y) in enumerate(coords_2d):
        cid = cluster_ids[i]
        color = color_map.get(cid, "gray")
        ax.scatter(x, y, c=[color], s=60, alpha=0.8, edgecolors="white", linewidths=0.3)

    # Add cluster labels at centroids
    for cluster in clusters:
        member_indices = [
            i for i, s in enumerate(results.samples) if s in cluster.members
        ]
        if member_indices and len(coords_2d) > max(member_indices):
            cx = coords_2d[member_indices, 0].mean()
            cy = coords_2d[member_indices, 1].mean()
            ax.annotate(
                f"C{cluster.id}",
                (cx, cy),
                color="white",
                fontsize=9,
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.set_title(
        f"Semantic Clusters (UMAP)\n"
        f"n={len(results.samples)}, clusters={len(clusters)}, "
        f"entropy={results.metrics.semantic_entropy:.3f}",
        color="white",
        fontsize=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
