"""Tests for sca.core.analyzer — uses a mock backend."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from sca.core.analyzer import Results, SemanticConsistencyAnalyzer


# ── Mock backends ─────────────────────────────────────────────────────────


class FixedBackend:
    """Always returns the same response regardless of prompt."""

    def __init__(self, response: str = "The answer is 42.") -> None:
        self.response = response
        self.call_count = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return self.response


class AlternatingBackend:
    """Alternates between two responses to create two semantic clusters."""

    def __init__(self, response_a: str, response_b: str) -> None:
        self.responses = [response_a, response_b]
        self.call_count = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        response = self.responses[self.call_count % 2]
        self.call_count += 1
        return response


class VariedBackend:
    """Returns different responses from a predefined list."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


# ── Test Results dataclass ────────────────────────────────────────────────


class TestResults:
    def _make_results(self, n: int = 5) -> Results:
        backend = FixedBackend("Test response about economics.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test prompt",
            backend=backend,
            n=n,
        )
        return analyzer.run()

    def test_results_has_expected_fields(self):
        results = self._make_results(5)
        assert hasattr(results, "samples")
        assert hasattr(results, "embeddings")
        assert hasattr(results, "similarity_matrix")
        assert hasattr(results, "metrics")
        assert hasattr(results, "clusters")
        assert hasattr(results, "config")
        assert hasattr(results, "timestamp")

    def test_samples_count(self):
        results = self._make_results(5)
        assert len(results.samples) == 5

    def test_embeddings_shape(self):
        n = 5
        results = self._make_results(n)
        assert results.embeddings.ndim == 2
        assert results.embeddings.shape[0] == n

    def test_similarity_matrix_shape(self):
        n = 5
        results = self._make_results(n)
        assert results.similarity_matrix.shape == (n, n)

    def test_similarity_matrix_symmetric(self):
        results = self._make_results(5)
        np.testing.assert_allclose(
            results.similarity_matrix,
            results.similarity_matrix.T,
            atol=1e-5,
        )

    def test_metrics_populated(self):
        results = self._make_results(5)
        m = results.metrics
        assert isinstance(m.mean_pairwise_similarity, float)
        assert isinstance(m.semantic_entropy, float)
        assert isinstance(m.cluster_count, int)
        assert isinstance(m.silhouette_score, float)
        assert isinstance(m.centroid_distance_variance, float)
        assert m.entailment_rate is None  # NLI disabled

    def test_timestamp_is_iso_format(self):
        results = self._make_results(5)
        from datetime import datetime  # noqa: PLC0415
        # Should not raise
        dt = datetime.fromisoformat(results.timestamp.replace("Z", "+00:00"))
        assert dt is not None

    def test_config_has_prompt(self):
        results = self._make_results(5)
        assert results.config.get("prompt") == "Test prompt"
        assert results.config.get("n") == 5


# ── Test JSON serialization ───────────────────────────────────────────────


class TestResultsSerialization:
    def test_to_dict_returns_dict(self):
        backend = FixedBackend("Simple response.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        d = results.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_is_json_serializable(self):
        backend = FixedBackend("Hello world.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        d = results.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_to_dict_embeddings_are_lists(self):
        backend = FixedBackend("Consistent answer.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        d = results.to_dict()
        assert isinstance(d["embeddings"], list)
        assert isinstance(d["embeddings"][0], list)

    def test_to_dict_similarity_matrix_is_list(self):
        backend = FixedBackend("Test.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        d = results.to_dict()
        assert isinstance(d["similarity_matrix"], list)
        assert isinstance(d["similarity_matrix"][0], list)

    def test_to_dict_cluster_centroids_are_lists(self):
        backend = FixedBackend("Response.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        d = results.to_dict()
        for cluster in d["clusters"]:
            assert isinstance(cluster["centroid"], list)

    def test_roundtrip_samples(self):
        responses = ["Alpha", "Beta", "Gamma", "Delta"]
        backend = VariedBackend(responses)
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        d = results.to_dict()
        parsed = json.loads(json.dumps(d))
        assert len(parsed["samples"]) == 4
        for s in parsed["samples"]:
            assert s in responses


# ── Test SemanticConsistencyAnalyzer behavior ─────────────────────────────


class TestAnalyzerBehavior:
    def test_fixed_backend_high_similarity(self):
        """Identical responses should have near-perfect similarity."""
        backend = FixedBackend("The quick brown fox jumps over the lazy dog.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=6,
        )
        results = analyzer.run()
        # All identical → should be very high similarity
        assert results.metrics.mean_pairwise_similarity > 0.95

    def test_fixed_backend_low_entropy(self):
        """Identical responses should cluster into 1 cluster with low entropy."""
        backend = FixedBackend("Same response every time.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=6,
        )
        results = analyzer.run()
        # Single cluster → entropy = 0
        assert results.metrics.semantic_entropy == pytest.approx(0.0, abs=1e-6)

    def test_nli_disabled_by_default(self):
        backend = FixedBackend("Test response.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=4,
        )
        results = analyzer.run()
        assert results.metrics.entailment_rate is None

    def test_custom_n(self):
        backend = FixedBackend("Response.")
        for n in [3, 7, 10]:
            analyzer = SemanticConsistencyAnalyzer(
                prompt="Test",
                backend=backend,
                n=n,
            )
            results = analyzer.run()
            assert len(results.samples) == n

    def test_on_sample_callback(self):
        """on_sample callback should be called for each sample."""
        received = []

        def callback(index: int, text: str, _) -> None:
            received.append((index, text))

        backend = FixedBackend("Test.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=5,
        )
        results = asyncio.run(analyzer.run_async(on_sample=callback))
        assert len(received) == 5
        assert len(results.samples) == 5

    def test_clusters_all_samples_accounted(self):
        """All samples should be accounted for in clusters."""
        backend = VariedBackend(["A", "B", "C", "D", "E", "F"])
        analyzer = SemanticConsistencyAnalyzer(
            prompt="Test",
            backend=backend,
            n=6,
        )
        results = analyzer.run()
        all_cluster_members = []
        for c in results.clusters:
            all_cluster_members.extend(c.members)
        assert sorted(all_cluster_members) == sorted(results.samples)

    def test_config_in_results(self):
        backend = FixedBackend("Answer.")
        analyzer = SemanticConsistencyAnalyzer(
            prompt="What is the meaning?",
            backend=backend,
            n=4,
            temperature=0.7,
        )
        results = analyzer.run()
        assert results.config["prompt"] == "What is the meaning?"
        assert results.config["n"] == 4
        assert results.config["temperature"] == 0.7
