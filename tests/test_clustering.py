"""Tests for sca.core.clustering."""

from __future__ import annotations

import numpy as np
import pytest

from sca.core.clustering import Cluster, cluster_embeddings


# ── Helpers ───────────────────────────────────────────────────────────────


def make_two_cluster_data(n_per_cluster: int = 20, dims: int = 8, sep: float = 5.0):
    """Generate synthetic data with two well-separated clusters."""
    rng = np.random.default_rng(42)
    center_a = np.zeros(dims)
    center_b = np.zeros(dims)
    center_b[0] = sep

    cluster_a = rng.standard_normal((n_per_cluster, dims)) * 0.5 + center_a
    cluster_b = rng.standard_normal((n_per_cluster, dims)) * 0.5 + center_b

    embeddings = np.vstack([cluster_a, cluster_b])
    samples = [f"sample_a_{i}" for i in range(n_per_cluster)] + \
              [f"sample_b_{i}" for i in range(n_per_cluster)]
    return embeddings, samples


# ── Tests ─────────────────────────────────────────────────────────────────


class TestClusterEmbeddings:
    def test_returns_list_of_clusters(self):
        embeddings, samples = make_two_cluster_data()
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=5)
        assert isinstance(clusters, list)
        assert all(isinstance(c, Cluster) for c in clusters)

    def test_detects_two_clusters(self):
        embeddings, samples = make_two_cluster_data(n_per_cluster=20, sep=8.0)
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=5)
        # Should find approximately 2 clusters (some noise points may be singletons)
        # Count clusters with > 1 member
        large_clusters = [c for c in clusters if len(c.members) > 1]
        assert len(large_clusters) >= 2

    def test_all_samples_accounted_for(self):
        """All input samples should appear in exactly one cluster."""
        embeddings, samples = make_two_cluster_data(n_per_cluster=15)
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=3)
        all_members = []
        for cluster in clusters:
            all_members.extend(cluster.members)
        assert sorted(all_members) == sorted(samples)

    def test_single_sample(self):
        embeddings = np.array([[1.0, 0.0, 0.0]])
        samples = ["only_sample"]
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=2)
        assert len(clusters) == 1
        assert clusters[0].members == ["only_sample"]

    def test_cluster_ids_sequential(self):
        embeddings, samples = make_two_cluster_data(n_per_cluster=10)
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=3)
        ids = [c.id for c in clusters]
        assert ids == list(range(len(clusters)))

    def test_clusters_sorted_by_size(self):
        embeddings, samples = make_two_cluster_data(n_per_cluster=20, sep=10.0)
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=3)
        sizes = [len(c.members) for c in clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_centroid_shape(self):
        embeddings, samples = make_two_cluster_data(n_per_cluster=10)
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=3)
        for cluster in clusters:
            assert cluster.centroid.shape == (embeddings.shape[1],)

    def test_noise_points_become_singletons(self):
        """Noise points (if any) should be included as singleton clusters."""
        # Use very small min_cluster_size to force some noise
        rng = np.random.default_rng(7)
        # Very spread out data — many will be noise
        embeddings = rng.standard_normal((30, 16)) * 3.0
        samples = [f"s{i}" for i in range(30)]
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=2)

        # Verify all samples are in exactly one cluster
        all_members = []
        for c in clusters:
            all_members.extend(c.members)
        assert sorted(all_members) == sorted(samples)

    def test_cluster_dataclass_fields(self):
        embeddings, samples = make_two_cluster_data(n_per_cluster=10)
        clusters = cluster_embeddings(embeddings, samples)
        for c in clusters:
            assert hasattr(c, "id")
            assert hasattr(c, "members")
            assert hasattr(c, "centroid")
            assert hasattr(c, "summary")
            assert c.summary == ""  # summary starts empty

    def test_small_dataset_no_crash(self):
        """Should not crash on tiny datasets."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        samples = ["a", "b"]
        clusters = cluster_embeddings(embeddings, samples, min_cluster_size=2)
        assert isinstance(clusters, list)
        all_members = [m for c in clusters for m in c.members]
        assert sorted(all_members) == ["a", "b"]
