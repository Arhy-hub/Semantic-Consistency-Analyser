# CLAUDE.md — Semantic Consistency Analyzer (sca)

## Project Overview

A terminal tool and importable Python package that samples an LLM (or any text-generation backend) N times on a fixed prompt, measures the semantic consistency of the outputs using embedding-based metrics, and visualises the distribution live in a TUI.

**Core insight:** semantic consistency is measured by treating outputs as points in embedding space and analysing the shape of that distribution — not just a single similarity score.

---

## Repository Structure

```
sca/
├── core/
│   ├── backends/
│   │   ├── protocol.py       # Backend + BatchBackend protocols (duck-typed)
│   │   ├── litellm.py        # API models + Ollama/vLLM via LiteLLM
│   │   └── hf.py             # HuggingFace transformers convenience wrapper
│   ├── sampler.py            # Async parallel sampling, batch dispatch
│   ├── embedder.py           # sentence-transformers wrapper
│   ├── metrics.py            # Entropy, cosine similarity, silhouette, entailment
│   ├── clustering.py         # HDBSCAN + optional UMAP dim reduction
│   └── analyzer.py           # SemanticConsistencyAnalyzer — main public class
├── tui/
│   ├── app.py                # Textual application root
│   └── widgets/
│       ├── similarity_heatmap.py
│       ├── cluster_panel.py
│       ├── metrics_panel.py
│       └── sample_feed.py
├── cli/
│   └── main.py               # Typer entrypoint
├── export/
│   ├── json.py
│   ├── html.py               # Static Textual snapshot
│   └── png.py                # UMAP scatter via matplotlib
├── config.py                 # Defaults, model aliases, pydantic-settings
├── pyproject.toml
└── tests/
```

---

## Architecture

### Backend Protocol

The core abstraction. Anything callable with signature `(str, **kwargs) -> str` is a valid backend. Never add hard dependencies on specific inference frameworks into `core/` — only the convenience wrappers in `backends/` may import them.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Backend(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str: ...

@runtime_checkable
class BatchBackend(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str: ...
    def batch_complete(self, prompts: list[str], **kwargs) -> list[str]: ...
```

The sampler checks `isinstance(backend, BatchBackend)` and dispatches to `batch_complete` when available — important for local GPU models where batching is a significant performance win.

### SemanticConsistencyAnalyzer

The main public class. Accepts either a `model` string (routed through LiteLLM) or a raw `backend` callable. Returns a `Results` dataclass.

```python
@dataclass
class Results:
    samples: list[str]
    embeddings: np.ndarray          # shape (n, embedding_dim)
    similarity_matrix: np.ndarray   # shape (n, n), pairwise cosine
    metrics: Metrics
    clusters: list[Cluster]

@dataclass
class Metrics:
    mean_pairwise_similarity: float
    semantic_entropy: float
    cluster_count: int
    silhouette_score: float
    centroid_distance_variance: float
    entailment_rate: float | None   # None if NLI not enabled
```

### Metrics

Implement in order of priority:

1. **Mean pairwise cosine similarity** — fast baseline, always computed
2. **Semantic entropy** (Kuhn et al. 2023) — cluster outputs by entailment, compute entropy over cluster probability distribution. This is the primary metric.
3. **Cluster count + silhouette score** — from HDBSCAN, captures multimodality
4. **Centroid distance variance** — spread from semantic centroid
5. **NLI entailment rate** — optional, gated behind `--nli` flag, uses `cross-encoder/nli-deberta-v3-small`

Note: cosine similarity and semantic entropy can *disagree*. A bimodal distribution (two equal opposite clusters) has low cosine similarity but also low entropy. Surface both — the disagreement is informative.

---

## Tech Stack

| Layer | Library | Notes |
|---|---|---|
| CLI | `typer` | Typed, autodocs, shell completion |
| TUI | `textual` | Async, composable, live updates |
| LLM APIs | `litellm` | OpenAI / Anthropic / Gemini / Ollama / vLLM |
| Local models | `transformers`, `llama-cpp-python` | Convenience wrappers only, optional deps |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast, local, no API key needed |
| Clustering | `hdbscan` | Density-based, no fixed k |
| Dim reduction | `umap-learn` | 2D projection for scatter export |
| Similarity | `numpy` | Cosine via normalised dot product |
| NLI (optional) | `sentence-transformers` cross-encoder | `cross-encoder/nli-deberta-v3-small` |
| Async | `asyncio` + `anyio` | Parallel sampling |
| Export | `matplotlib` | PNG scatter plot |
| Config | `pydantic-settings` | Env vars + config file |
| Packaging | `uv` + `pyproject.toml` | Modern, fast |
| Dev tooling | `ruff`, `mypy`, `pytest` | Lint, types, tests |

---

## CLI Interface

```bash
# Basic single run
sca run "What causes inflation?" --model gpt-4o --n 20 --temperature 0.9

# Local HuggingFace model
sca run "..." --model hf:mistralai/Mistral-7B-Instruct-v0.3 --n 20

# Raw callable backend (path::function syntax)
sca run "..." --backend path/to/generate.py::my_function --n 20

# Temperature sweep — the headline feature
sca sweep "..." --model claude-sonnet-4-6 --temps 0.0,0.5,1.0,1.5 --n 20

# Model comparison
sca compare "..." --models gpt-4o,claude-sonnet-4-6,llama3.2 --n 20

# With NLI entailment rate
sca run "..." --model gpt-4o --n 20 --nli

# Export
sca run "..." --model gpt-4o --n 20 --export json --out results.json
sca run "..." --model gpt-4o --n 20 --export png --out scatter.png
sca run "..." --model gpt-4o --n 20 --export html --out report.html
```

## Package Interface

```python
from sca import SemanticConsistencyAnalyzer
from sca.backends import HFBackend

# API model via LiteLLM
results = SemanticConsistencyAnalyzer(
    model="claude-sonnet-4-6",
    prompt="What causes inflation?",
    n=20,
    temperature=0.9
).run()

# Raw callable — anything works
results = SemanticConsistencyAnalyzer(
    backend=my_model.generate,
    prompt="...",
    n=20
).run()

# HuggingFace convenience wrapper
results = SemanticConsistencyAnalyzer(
    backend=HFBackend("mistralai/Mistral-7B-Instruct-v0.3"),
    prompt="...",
    n=20
).run()

# Access results
print(results.metrics.semantic_entropy)
print(results.metrics.cluster_count)
print(results.similarity_matrix)          # np.ndarray (n x n)
for cluster in results.clusters:
    print(cluster.summary)                # auto-generated one-sentence label
    print(cluster.members)               # list of sample strings
```

---

## TUI Layout

```
┌─────────────────────┬──────────────────────────────┐
│  Sample Feed        │  Similarity Heatmap           │
│  (live streaming)   │  (updates per sample)         │
├─────────────────────┼──────────────────────────────┤
│  Metrics Panel      │  Cluster Panel                │
│  entropy / sim /    │  N clusters, auto-summaries,  │
│  silhouette         │  member count per cluster     │
└─────────────────────┴──────────────────────────────┘
```

- Heatmap and metrics update **incrementally** as each sample arrives — do not wait for all N
- Cluster panel populates once enough samples exist to cluster (configurable minimum, default 5)
- Temperature sweep mode shows a side-by-side bar chart of entropy/cluster count across temperatures
- Textual's built-in HTML export captures the full TUI state as a static shareable file

---

## Modes

**Single run** — N samples, one temperature, one model. Default mode.

**Temperature sweep** — Same prompt, same model, temperatures ∈ {0.0, 0.3, 0.7, 1.0, 1.5} (configurable). Shows how entropy and cluster count change with temperature. This is the most visually compelling output.

**Model comparison** — Same prompt, same temperature, different models side by side. Panels laid out horizontally.

**Prompt perturbation** — LLM-assisted rephrasing of the prompt N ways, then measure whether meaning-equivalent prompts produce meaning-equivalent distributions. Tests input robustness.

---

## Key Implementation Notes

### Incremental similarity matrix
Do not recompute the full (n×n) matrix per sample. Keep a running matrix and only compute the new row/column when a sample arrives.

### Convergence detection
Track a rolling window of semantic entropy. Stop sampling early if entropy delta falls below `--convergence-threshold` (default 0.01) for 3 consecutive windows. This is optional and off by default.

### Cluster auto-summarisation
After clustering, make one LLM call per cluster: pass the cluster's member responses and ask for a one-sentence characterisation. Use the same backend as the sampling run. This is cheap and high-value for long-form outputs.

### Reproducibility
`Results` must be fully serialisable to JSON — all samples, embeddings (as lists), config, model name, timestamp. A saved JSON result can be replayed and re-visualised without re-querying the model.

### BatchBackend dispatch
```python
async def sample(backend, prompt, n, **kwargs):
    if isinstance(backend, BatchBackend):
        return await backend.batch_complete([prompt] * n, **kwargs)
    else:
        tasks = [asyncio.create_task(
            asyncio.to_thread(backend, prompt, **kwargs)
        ) for _ in range(n)]
        return await asyncio.gather(*tasks)
```

---

## Optional Dependencies

Use optional dependency groups in `pyproject.toml` to avoid forcing heavy installs:

```toml
[project.optional-dependencies]
hf = ["transformers", "torch"]
gguf = ["llama-cpp-python"]
nli = ["sentence-transformers"]   # cross-encoder, heavier model
umap = ["umap-learn"]
all = ["sca[hf,gguf,nli,umap]"]
```

Core install (`pip install sca`) should only require: `typer`, `textual`, `litellm`, `sentence-transformers` (for embeddings only, not NLI), `hdbscan`, `numpy`, `pydantic-settings`.

---

## Non-Goals

- GUI / web interface
- Database persistence between runs
- Multi-turn / conversational consistency (single prompt only, for now)
- Fine-tuning or training integration
- Serving as an LLM evaluation benchmark framework — that is ConsistencyChecker's territory
