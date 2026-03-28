"""Textual TUI application root for sca."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header
from textual import work

from sca.core.analyzer import Results, SemanticConsistencyAnalyzer
from sca.core.clustering import Cluster, cluster_embeddings
from sca.core.embedder import Embedder
from sca.core.metrics import (
    Metrics,
    centroid_distance_variance,
    compute_similarity_matrix,
    mean_pairwise_similarity,
    semantic_entropy,
    silhouette,
)
from sca.tui.widgets.cluster_panel import ClusterPanel
from sca.tui.widgets.metrics_panel import MetricsPanel
from sca.tui.widgets.sample_feed import SampleFeed
from sca.tui.widgets.similarity_heatmap import SimilarityHeatmap


class SampleReceived(Message):
    """Posted when a new sample arrives from the backend."""

    def __init__(self, index: int, text: str) -> None:
        super().__init__()
        self.index = index
        self.text = text


class MetricsUpdated(Message):
    """Posted when metrics have been recomputed."""

    def __init__(self, metrics: Metrics, sample_count: int) -> None:
        super().__init__()
        self.metrics = metrics
        self.sample_count = sample_count


class ClustersUpdated(Message):
    """Posted when clusters have been recomputed."""

    def __init__(self, clusters: list[Cluster]) -> None:
        super().__init__()
        self.clusters = clusters


class HeatmapUpdated(Message):
    """Posted when the similarity matrix has been updated."""

    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__()
        self.matrix = matrix


class AnalysisComplete(Message):
    """Posted when the full analysis is done."""

    def __init__(self, results: Results) -> None:
        super().__init__()
        self.results = results


class SCAApp(App):
    """
    Semantic Consistency Analyzer TUI.

    4-panel layout:
    ┌─────────────────────┬──────────────────────────────┐
    │  Sample Feed        │  Similarity Heatmap           │
    │  (live streaming)   │  (updates per sample)         │
    ├─────────────────────┼──────────────────────────────┤
    │  Metrics Panel      │  Cluster Panel                │
    │  entropy / sim /    │  N clusters, auto-summaries,  │
    │  silhouette         │  member count per cluster     │
    └─────────────────────┴──────────────────────────────┘
    """

    CSS = """
    Screen {
        layout: vertical;
        background: #0a0a0a;
        color: white;
    }
    Header {
        background: #0a0a0a;
        color: #00d7d7;
        text-style: bold;
    }
    Footer {
        background: #111111;
        color: #555555;
    }
    #top-row {
        height: 60%;
        layout: horizontal;
    }
    #bottom-row {
        height: 40%;
        layout: horizontal;
    }
    SampleFeed, SimilarityHeatmap, MetricsPanel, ClusterPanel {
        width: 50%;
        background: #0a0a0a;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, analyzer: SemanticConsistencyAnalyzer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analyzer = analyzer
        self._results: Results | None = None
        # Internal state for incremental updates
        self._samples: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._embedder = Embedder(analyzer.embedding_model)
        self._min_cluster_size = analyzer.min_cluster_size

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            yield SampleFeed(id="sample-feed")
            yield SimilarityHeatmap(id="heatmap")
        with Horizontal(id="bottom-row"):
            yield MetricsPanel(id="metrics")
            yield ClusterPanel(id="clusters")
        yield Footer()

    def on_mount(self) -> None:
        """Start the sampling worker when the app mounts."""
        self._run_analysis()

    @work(exclusive=True)
    async def _run_analysis(self) -> None:
        """Background worker that runs the analyzer and posts UI messages."""
        backend = self.analyzer._get_backend()

        from sca.core.sampler import sample_stream  # noqa: PLC0415
        from sca.core.backends.protocol import BatchBackend  # noqa: PLC0415

        if isinstance(backend, BatchBackend):
            # For batch backends, get all samples then stream display
            raw = await asyncio.to_thread(
                backend.batch_complete,
                [self.analyzer.prompt] * self.analyzer.n,
                temperature=self.analyzer.temperature,
            )
            for i, text in enumerate(raw):
                await self._process_new_sample(i, text)
        else:
            async for idx, text in sample_stream(
                backend,
                self.analyzer.prompt,
                self.analyzer.n,
                temperature=self.analyzer.temperature,
            ):
                await self._process_new_sample(idx, text)

        # Final full analysis
        if self._samples and self._embeddings:
            embeddings = np.stack(self._embeddings)
            sim_matrix = compute_similarity_matrix(embeddings)
            clusters = await asyncio.to_thread(
                cluster_embeddings, embeddings, self._samples, self._min_cluster_size
            )
            # Generate summaries
            await self.analyzer._summarize_clusters(clusters, backend)
            self.post_message(ClustersUpdated(clusters))

            labels = self._build_labels(self._samples, clusters)
            metrics = self._compute_metrics(embeddings, sim_matrix, clusters, labels)
            self.post_message(MetricsUpdated(metrics, len(self._samples)))

    async def _process_new_sample(self, index: int, text: str) -> None:
        """Process a single new sample: embed, update matrix, recompute metrics."""
        self._samples.append(text)
        emb = await asyncio.to_thread(self._embedder.embed_one, text)
        self._embeddings.append(emb)

        self.post_message(SampleReceived(index, text))

        embeddings = np.stack(self._embeddings)
        sim_matrix = compute_similarity_matrix(embeddings)
        self.post_message(HeatmapUpdated(sim_matrix))

        # Recompute clusters and metrics once we have enough samples
        if len(self._samples) >= self._min_cluster_size:
            clusters = await asyncio.to_thread(
                cluster_embeddings, embeddings, self._samples, self._min_cluster_size
            )
            labels = self._build_labels(self._samples, clusters)
            metrics = self._compute_metrics(embeddings, sim_matrix, clusters, labels)
            self.post_message(MetricsUpdated(metrics, len(self._samples)))
            self.post_message(ClustersUpdated(clusters))

    def _build_labels(self, samples: list[str], clusters: list[Cluster]) -> np.ndarray:
        labels = np.zeros(len(samples), dtype=int)
        for cluster in clusters:
            for member in cluster.members:
                for i, s in enumerate(samples):
                    if s == member and labels[i] == 0:
                        labels[i] = cluster.id
                        break
        return labels

    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        sim_matrix: np.ndarray,
        clusters: list[Cluster],
        labels: np.ndarray,
    ) -> Metrics:
        mps = mean_pairwise_similarity(sim_matrix)
        s_entropy = semantic_entropy(clusters)
        cdv = centroid_distance_variance(embeddings)
        sil = silhouette(embeddings, labels)
        return Metrics(
            mean_pairwise_similarity=mps,
            semantic_entropy=s_entropy,
            cluster_count=len(clusters),
            silhouette_score=sil,
            centroid_distance_variance=cdv,
            entailment_rate=None,
        )

    # ── Message handlers ─────────────────────────────────────────────────

    def on_sample_received(self, message: SampleReceived) -> None:
        feed = self.query_one("#sample-feed", SampleFeed)
        feed.add_sample(message.index, message.text)

    def on_metrics_updated(self, message: MetricsUpdated) -> None:
        panel = self.query_one("#metrics", MetricsPanel)
        panel.update_metrics(message.metrics, message.sample_count)

    def on_clusters_updated(self, message: ClustersUpdated) -> None:
        panel = self.query_one("#clusters", ClusterPanel)
        panel.update_clusters(message.clusters)

    def on_heatmap_updated(self, message: HeatmapUpdated) -> None:
        heatmap = self.query_one("#heatmap", SimilarityHeatmap)
        heatmap.update_matrix(message.matrix)

    def on_analysis_complete(self, message: AnalysisComplete) -> None:
        self._results = message.results

    def get_results(self) -> Results | None:
        """Return results after analysis is complete."""
        return self._results
