# sca — Semantic Consistency Analyzer
(main is the most up to date version)

A terminal tool that samples an LLM N times on the same prompt and measures how semantically consistent the outputs are. Results stream into a live TUI as samples arrive.

**Core idea:** if you ask a model the same question 20 times at temperature 0.9, do you get 20 versions of the same answer, or 20 different ones? High consistency → the model has a confident, stable response. High entropy → the model is uncertain or the question is genuinely ambiguous.

---

## TUI Layout

```
┌─────────────────────┬──────────────────────────────┐
│  Sample Feed        │  Similarity Heatmap           │
│  (live streaming)   │  (n×n cosine similarity)      │
├─────────────────────┼───────────────┬───────────────┤
│  Metrics Panel      │  Cluster Panel│  Similarity   │
│  entropy / sim /    │  (click to    │  Histogram    │
│  silhouette / etc   │   expand)     │               │
└─────────────────────┴───────────────┴───────────────┘
```

- **Sample Feed** — each LLM response as it arrives; click any sample to read the full text
- **Similarity Heatmap** — n×n matrix where bright = semantically similar, dark = different; switchable to euclidean / silhouette / centroid / cluster-agreement views
- **Metrics Panel** — semantic entropy, mean pairwise similarity, cluster count, silhouette score, centroid distance variance; includes an entropy sparkline over time
- **Cluster Panel** — HDBSCAN clusters with auto-generated one-sentence summaries; click any cluster to see all its members
- **Similarity Histogram** — distribution of all pairwise cosine similarity values; a spike near 1.0 = consistent, spread = diverse

---

## Install

Requires Python ≥ 3.11.

```bash
git clone <repo>
cd ai-safety-hack

# create venv and install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Or with uv:

```bash
uv sync
```

---

## Usage

### Basic run

```bash
sca run "What causes inflation?" --model gpt-4o --n 20 --temperature 0.9
```

Opens the TUI. Press `q` to quit.

### With Ollama (local model)

```bash
# Start Ollama first: ollama serve
sca run "What causes inflation?" --model ollama/llama3.2 --n 20
```

### Choose what the heatmap displays

```bash
sca run "..." --model gpt-4o --measure cosine      # default
sca run "..." --model gpt-4o --measure euclidean
sca run "..." --model gpt-4o --measure silhouette
sca run "..." --model gpt-4o --measure centroid
sca run "..." --model gpt-4o --measure agreement   # cluster co-membership
```

### No TUI — print results to terminal

```bash
sca run "What causes inflation?" --model gpt-4o --n 20 --no-tui
```

### Temperature sweep

Runs the same prompt at multiple temperatures and prints a comparison table.

```bash
sca sweep "What causes inflation?" --model gpt-4o --temps 0.0,0.3,0.5,0.7,1.0 --n 10
```

### Model comparison

```bash
sca compare "What causes inflation?" gpt-4o,claude-sonnet-4-6,ollama/llama3.2 --n 15
```

### Export results

```bash
sca run "..." --model gpt-4o --n 20 --export json --out results.json
sca run "..." --model gpt-4o --n 20 --export html --out report.html
sca run "..." --model gpt-4o --n 20 --export png  --out scatter.png   # requires sca[export]
```

### NLI-accurate semantic entropy

Uses a DeBERTa NLI model to build meaning classes — closer to the Kuhn et al. 2023 definition of semantic entropy (slower).

```bash
sca run "..." --model gpt-4o --n 20 --nli
```

### Custom backend

Any Python function `(prompt: str, **kwargs) -> str` works as a backend:

```bash
sca run "..." --backend path/to/generate.py::my_function --n 20
```

---

## Metrics explained

| Metric | What it means |
|---|---|
| **Semantic entropy** | Shannon entropy over cluster size distribution. 0 = all outputs in one meaning cluster (very consistent). Higher = more distinct semantic groups. |
| **Mean pairwise similarity** | Average cosine similarity between all output pairs. 1.0 = identical embeddings, 0.0 = orthogonal. |
| **Cluster count** | Number of distinct semantic groups found by HDBSCAN. 1 = consistent, N = every output is different. |
| **Silhouette score** | How well-separated the clusters are. Close to 1.0 = clear groups, close to 0 = overlapping. |
| **Centroid distance variance** | Spread of outputs around the semantic mean. Low = tight distribution, high = scattered. |

---

## API keys

Set the relevant environment variable before running:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

Ollama and other local models need no API key — just set `--model ollama/<model-name>`.



[Paper for Semantic Entropy](https://arxiv.org/pdf/2302.09664)
