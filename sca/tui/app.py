"""Textual TUI application root for sca."""

from __future__ import annotations

import asyncio

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Footer, Header
from textual import work

from sca.core.analyzer import Results, SemanticConsistencyAnalyzer
from sca.core.clustering import Cluster, cluster_embeddings
from sca.core.embedder import Embedder
from sca.core.metrics import (
    Metrics,
    centroid_distance_variance,
    centroid_distance_matrix,
    compute_euclidean_matrix,
    compute_similarity_matrix,
    mean_pairwise_similarity,
    semantic_entropy,
    silhouette,
    silhouette_matrix,
)
from sca.tui.widgets.cluster_panel import ClusterPanel
from sca.tui.widgets.metrics_panel import MetricsPanel
from sca.tui.widgets.sample_feed import SampleFeed
from sca.tui.widgets.similarity_heatmap import SimilarityHeatmap
from sca.tui.widgets.scatter_plot import ScatterPlot


# ── Messages ──────────────────────────────────────────────────────────────

class SampleReceived(Message):
    def __init__(self, index: int, text: str) -> None:
        super().__init__()
        self.index = index
        self.text = text


class MetricsUpdated(Message):
    def __init__(self, metrics: Metrics, sample_count: int, entropy_history: list[float] | None = None) -> None:
        super().__init__()
        self.metrics = metrics
        self.sample_count = sample_count
        self.entropy_history = entropy_history or []


class ClustersUpdated(Message):
    def __init__(self, clusters: list[Cluster]) -> None:
        super().__init__()
        self.clusters = clusters


class HeatmapUpdated(Message):
    def __init__(self, matrix: np.ndarray, title: str = "cosine sim") -> None:
        super().__init__()
        self.matrix = matrix
        self.title = title


class ScatterUpdated(Message):
    def __init__(self, coords: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
        self.coords = coords
        self.labels = labels


class AnalysisComplete(Message):
    def __init__(self, results: Results) -> None:
        super().__init__()
        self.results = results


# ── Measure → display title ────────────────────────────────────────────────

_MEASURE_TITLES: dict[str, str] = {
    "cosine":     "cosine similarity",
    "euclidean":  "euclidean distance",
    "agreement":  "cluster agreement",
    "silhouette": "silhouette scores",
    "centroid":   "centroid distance",
}


def _agreement_matrix(labels: np.ndarray) -> np.ndarray:
    """Binary matrix: +1 if same cluster, -1 if different."""
    same = (labels[:, None] == labels[None, :]).astype(float)
    return same * 2.0 - 1.0


# ── App ────────────────────────────────────────────────────────────────────

class SCA(App):
    """
    Semantic Consistency Analyzer TUI.

    ┌───────────────────┬──────────────────────────────┐
    │  Sample Feed      │  Heatmap                      │  60%
    ├─────────┬─────────┴──────────────┬────────────────┤
    │ Metrics │ Clusters               │ UMAP Scatter    │  40%
    └─────────┴────────────────────────┴────────────────┘
    """

    CSS = """
    Screen { layout: vertical; background: #0a0a0a; color: white; }
    Header { background: #0a0a0a; color: #00d7d7; text-style: bold; }
    Footer { background: #111111; color: #555555; }
    #top-row { height: 60%; layout: horizontal; }
    #bottom-row { height: 40%; layout: horizontal; }
    SampleFeed    { width: 40%; background: #0a0a0a; }
    SimilarityHeatmap { width: 60%; background: #0a0a0a; }
    MetricsPanel  { width: 30%; background: #0a0a0a; }
    ClusterPanel  { width: 40%; background: #0a0a0a; }
    ScatterPlot   { width: 30%; background: #0a0a0a; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        analyzer: SemanticConsistencyAnalyzer,
        measure: str = "cosine",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.analyzer = analyzer
        self._measure = measure
        self._results: Results | None = None
        self._samples: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._embedder = Embedder(analyzer.embedding_model)
        self._min_cluster_size = analyzer.min_cluster_size
        # Per-measure matrices, populated as data arrives
        self._cosine_mat:     np.ndarray | None = None
        self._euclidean_mat:  np.ndarray | None = None
        self._centroid_mat:   np.ndarray | None = None
        self._agreement_mat:  np.ndarray | None = None
        self._silhouette_mat: np.ndarray | None = None
        self._labels:         np.ndarray | None = None
        self._entropy_history: list[float] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            yield SampleFeed(id="sample-feed")
            yield SimilarityHeatmap(id="heatmap")
        with Horizontal(id="bottom-row"):
            yield MetricsPanel(id="metrics")
            yield ClusterPanel(id="clusters")
            yield ScatterPlot(id="scatter")
        yield Footer()

    def on_mount(self) -> None:
        self._run_analysis()

    # ── Heatmap dispatch ───────────────────────────────────────────────────

    def _post_heatmap(self) -> None:
        """Post whichever matrix corresponds to the configured measure."""
        matrix_for: dict[str, np.ndarray | None] = {
            "cosine":     self._cosine_mat,
            "euclidean":  self._euclidean_mat,
            "centroid":   self._centroid_mat,
            "agreement":  self._agreement_mat,
            "silhouette": self._silhouette_mat,
        }
        mat = matrix_for.get(self._measure)
        # Fall back to cosine if the requested measure isn't ready yet
        if mat is None:
            mat = self._cosine_mat
            if mat is None:
                return
            title = "cosine similarity"
        else:
            title = _MEASURE_TITLES.get(self._measure, self._measure)

        self.post_message(HeatmapUpdated(mat, title))


    # ── Scatter update ─────────────────────────────────────────────────────

    async def _update_scatter(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        if len(embeddings) < 4:
            return
        try:
            from sca.core.clustering import umap_reduce  # noqa: PLC0415
            coords = await asyncio.to_thread(umap_reduce, embeddings)
            self.post_message(ScatterUpdated(coords, labels))
        except Exception:
            pass

    # ── Background analysis worker ─────────────────────────────────────────

    @work(exclusive=True)
    async def _run_analysis(self) -> None:
        backend = self.analyzer._get_backend()

        from sca.core.sampler import sample_stream          # noqa: PLC0415
        from sca.core.backends.protocol import BatchBackend  # noqa: PLC0415

        if isinstance(backend, BatchBackend):
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

        # Final pass: generate cluster summaries then refresh everything
        if self._samples and self._embeddings:
            embeddings = np.stack(self._embeddings)
            clusters = await asyncio.to_thread(
                cluster_embeddings, embeddings, self._samples, self._min_cluster_size
            )
            await self.analyzer._summarize_clusters(clusters, backend)
            labels = self._build_labels(self._samples, clusters)
            self._labels         = labels
            self._agreement_mat  = _agreement_matrix(labels)
            self._silhouette_mat = await asyncio.to_thread(
                silhouette_matrix, embeddings, labels
            )
            self.post_message(ClustersUpdated(clusters))
            self._post_heatmap()
            metrics = self._compute_metrics(embeddings, self._cosine_mat, clusters, labels)
            self._entropy_history.append(metrics.semantic_entropy)
            self.post_message(MetricsUpdated(metrics, len(self._samples), list(self._entropy_history)))
            await self._update_scatter(embeddings, labels)

    async def _process_new_sample(self, index: int, text: str) -> None:
        self._samples.append(text)
        emb = await asyncio.to_thread(self._embedder.embed_one, text)
        self._embeddings.append(emb)

        self.post_message(SampleReceived(index, text))

        embeddings = np.stack(self._embeddings)

        # Embedding-based matrices — available immediately
        self._cosine_mat    = compute_similarity_matrix(embeddings)
        self._euclidean_mat = await asyncio.to_thread(compute_euclidean_matrix, embeddings)
        self._centroid_mat  = await asyncio.to_thread(centroid_distance_matrix, embeddings)

        if len(self._samples) >= self._min_cluster_size:
            clusters = await asyncio.to_thread(
                cluster_embeddings, embeddings, self._samples, self._min_cluster_size
            )
            labels = self._build_labels(self._samples, clusters)
            self._labels         = labels
            self._agreement_mat  = _agreement_matrix(labels)
            self._silhouette_mat = await asyncio.to_thread(
                silhouette_matrix, embeddings, labels
            )
            metrics = self._compute_metrics(embeddings, self._cosine_mat, clusters, labels)
            self._entropy_history.append(metrics.semantic_entropy)
            self.post_message(MetricsUpdated(metrics, len(self._samples), list(self._entropy_history)))
            self.post_message(ClustersUpdated(clusters))
            await self._update_scatter(embeddings, labels)

        self._post_heatmap()

    # ── Helpers ────────────────────────────────────────────────────────────

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
        return Metrics(
            mean_pairwise_similarity=mean_pairwise_similarity(sim_matrix),
            semantic_entropy=semantic_entropy(clusters),
            cluster_count=len(clusters),
            silhouette_score=silhouette(embeddings, labels),
            centroid_distance_variance=centroid_distance_variance(embeddings),
            entailment_rate=None,
        )

    # ── Message handlers ───────────────────────────────────────────────────

    def on_sample_received(self, message: SampleReceived) -> None:
        self.query_one("#sample-feed", SampleFeed).add_sample(message.index, message.text)

    def on_metrics_updated(self, message: MetricsUpdated) -> None:
        self.query_one("#metrics", MetricsPanel).update_metrics(
            message.metrics, message.sample_count, message.entropy_history
        )

    def on_clusters_updated(self, message: ClustersUpdated) -> None:
        self.query_one("#clusters", ClusterPanel).update_clusters(message.clusters)

    def on_heatmap_updated(self, message: HeatmapUpdated) -> None:
        self.query_one("#heatmap", SimilarityHeatmap).update_matrix(message.matrix, message.title)

    def on_scatter_updated(self, message: ScatterUpdated) -> None:
        self.query_one("#scatter", ScatterPlot).update(message.coords, message.labels)

    def on_analysis_complete(self, message: AnalysisComplete) -> None:
        self._results = message.results

    def get_results(self) -> Results | None:
        return self._results
