"""Tests for sca.core.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sca.core.metrics import (
    Metrics,
    centroid_distance_variance,
    compute_similarity_matrix,
    mean_pairwise_similarity,
    semantic_entropy,
    silhouette,
)
from sca.core.clustering import Cluster


# ── Helpers ───────────────────────────────────────────────────────────────


def make_cluster(id: int, size: int) -> Cluster:
    return Cluster(
        id=id,
        members=[f"sample_{id}_{i}" for i in range(size)],
        centroid=np.zeros(4),
    )


# ── compute_similarity_matrix ─────────────────────────────────────────────


class TestComputeSimilarityMatrix:
    def test_identical_vectors(self):
        emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sim = compute_similarity_matrix(emb)
        assert sim.shape == (2, 2)
        np.testing.assert_allclose(sim, np.ones((2, 2)), atol=1e-6)

    def test_orthogonal_vectors(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_similarity_matrix(emb)
        assert sim[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert sim[1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
        sim = compute_similarity_matrix(emb)
        assert sim[0, 1] == pytest.approx(-1.0, abs=1e-6)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((5, 8))
        sim = compute_similarity_matrix(emb)
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((4, 6))
        sim = compute_similarity_matrix(emb)
        np.testing.assert_allclose(np.diag(sim), np.ones(4), atol=1e-6)

    def test_shape(self):
        emb = np.random.randn(7, 16)
        sim = compute_similarity_matrix(emb)
        assert sim.shape == (7, 7)

    def test_zero_vector_handled(self):
        emb = np.array([[0.0, 0.0], [1.0, 0.0]])
        # Should not raise
        sim = compute_similarity_matrix(emb)
        assert sim.shape == (2, 2)


# ── mean_pairwise_similarity ──────────────────────────────────────────────


class TestMeanPairwiseSimilarity:
    def test_identical_embeddings(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        sim = compute_similarity_matrix(emb)
        mps = mean_pairwise_similarity(sim)
        assert mps == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_pair(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_similarity_matrix(emb)
        mps = mean_pairwise_similarity(sim)
        assert mps == pytest.approx(0.0, abs=1e-6)

    def test_single_sample(self):
        sim = np.array([[1.0]])
        mps = mean_pairwise_similarity(sim)
        assert mps == pytest.approx(1.0)

    def test_range(self):
        rng = np.random.default_rng(123)
        emb = rng.standard_normal((10, 32))
        sim = compute_similarity_matrix(emb)
        mps = mean_pairwise_similarity(sim)
        assert -1.0 <= mps <= 1.0


# ── semantic_entropy ──────────────────────────────────────────────────────


class TestSemanticEntropy:
    def test_single_cluster(self):
        # All in one cluster → entropy should be 0 (certainty)
        clusters = [make_cluster(0, 10)]
        entropy = semantic_entropy(clusters)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_uniform_two_clusters(self):
        # Two equal clusters → max entropy for 2 clusters = log(2)
        clusters = [make_cluster(0, 5), make_cluster(1, 5)]
        entropy = semantic_entropy(clusters)
        assert entropy == pytest.approx(math.log(2), abs=1e-6)

    def test_skewed_clusters(self):
        # One cluster much larger → lower entropy
        clusters_skewed = [make_cluster(0, 9), make_cluster(1, 1)]
        clusters_uniform = [make_cluster(0, 5), make_cluster(1, 5)]
        assert semantic_entropy(clusters_skewed) < semantic_entropy(clusters_uniform)

    def test_empty_clusters(self):
        assert semantic_entropy([]) == pytest.approx(0.0)

    def test_many_equal_clusters(self):
        # n equal clusters → entropy = log(n)
        n = 8
        clusters = [make_cluster(i, 1) for i in range(n)]
        entropy = semantic_entropy(clusters)
        assert entropy == pytest.approx(math.log(n), abs=1e-6)

    def test_non_negative(self):
        clusters = [make_cluster(i, i + 1) for i in range(5)]
        assert semantic_entropy(clusters) >= 0.0


# ── centroid_distance_variance ────────────────────────────────────────────


class TestCentroidDistanceVariance:
    def test_all_identical(self):
        emb = np.ones((5, 4))
        cdv = centroid_distance_variance(emb)
        assert cdv == pytest.approx(0.0, abs=1e-6)

    def test_spread(self):
        # Points at ±1 on first axis
        emb = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        cdv = centroid_distance_variance(emb)
        assert cdv >= 0.0

    def test_single_point(self):
        emb = np.array([[1.0, 2.0, 3.0]])
        cdv = centroid_distance_variance(emb)
        assert cdv == pytest.approx(0.0, abs=1e-6)

    def test_more_spread_means_higher_variance(self):
        # Use asymmetric data so distances from centroid actually vary
        tight = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.3], [0.15, 0.25]])
        spread = np.array([[10.0, 20.0], [30.0, 10.0], [20.0, 30.0], [15.0, 25.0]])
        assert centroid_distance_variance(spread) > centroid_distance_variance(tight)


# ── silhouette ────────────────────────────────────────────────────────────


class TestSilhouette:
    def test_single_cluster_returns_zero(self):
        emb = np.random.randn(5, 4)
        labels = np.zeros(5, dtype=int)
        score = silhouette(emb, labels)
        assert score == pytest.approx(0.0)

    def test_well_separated_clusters(self):
        # Create two clearly separated clusters
        cluster_a = np.random.randn(10, 4) + np.array([10.0, 0.0, 0.0, 0.0])
        cluster_b = np.random.randn(10, 4) - np.array([10.0, 0.0, 0.0, 0.0])
        emb = np.vstack([cluster_a, cluster_b])
        labels = np.array([0] * 10 + [1] * 10)
        score = silhouette(emb, labels)
        assert score > 0.8  # should be high for well-separated clusters

    def test_returns_float(self):
        emb = np.random.randn(6, 4)
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = silhouette(emb, labels)
        assert isinstance(score, float)

    def test_range(self):
        emb = np.random.randn(8, 4)
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        score = silhouette(emb, labels)
        assert -1.0 <= score <= 1.0
