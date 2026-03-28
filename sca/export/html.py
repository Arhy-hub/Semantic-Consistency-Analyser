"""HTML export — static snapshot of the Textual TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sca.tui.app import SCA


def export_html(app: "SCA", path: str) -> None:
    """
    Export the current TUI state as a static HTML file.

    Uses Textual's built-in export_screenshot capability.
    """
    try:
        # Textual's save_screenshot saves SVG; export_screenshot returns the SVG string
        svg_content = app.export_screenshot()
        # Wrap in minimal HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SCA Results</title>
<style>
body {{
    background: #1a1a2e;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 20px;
    font-family: monospace;
}}
img, svg {{
    max-width: 100%;
}}
</style>
</head>
<body>
{svg_content}
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        raise RuntimeError(f"Failed to export HTML: {e}") from e


def export_results_html(results, path: str) -> None:
    """
    Export Results as a static HTML report (no TUI required).

    Generates a formatted HTML table of metrics and samples.
    """
    import json  # noqa: PLC0415

    data = results.to_dict()
    metrics = data["metrics"]

    samples_html = ""
    for i, sample in enumerate(data["samples"]):
        escaped = sample.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        samples_html += f'<div class="sample"><span class="idx">[{i+1}]</span> {escaped}</div>\n'

    clusters_html = ""
    for c in data["clusters"]:
        summary = c.get("summary", "") or f"Cluster {c['id']}"
        clusters_html += (
            f'<div class="cluster">'
            f'<strong>Cluster {c["id"]}</strong> ({len(c["members"])} members): '
            f'{summary}</div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SCA Report — {data["config"].get("model", "unknown")}</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1, h2 {{ color: #7fbfff; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #444; padding: 6px 12px; text-align: left; }}
th {{ background: #2a2a4e; }}
.sample {{ padding: 5px 0; border-bottom: 1px solid #333; }}
.idx {{ color: #7fbfff; font-weight: bold; }}
.cluster {{ padding: 5px 0; }}
</style>
</head>
<body>
<h1>Semantic Consistency Analyzer Report</h1>
<p>Model: {data["config"].get("model", "unknown")} |
   Prompt: {data["config"].get("prompt", "")[:100]} |
   Timestamp: {data["timestamp"]}</p>

<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Mean Pairwise Similarity</td><td>{metrics["mean_pairwise_similarity"]:.4f}</td></tr>
<tr><td>Semantic Entropy</td><td>{metrics["semantic_entropy"]:.4f}</td></tr>
<tr><td>Cluster Count</td><td>{metrics["cluster_count"]}</td></tr>
<tr><td>Silhouette Score</td><td>{metrics["silhouette_score"]:.4f}</td></tr>
<tr><td>Centroid Distance Variance</td><td>{metrics["centroid_distance_variance"]:.6f}</td></tr>
{"<tr><td>Entailment Rate</td><td>" + f'{metrics["entailment_rate"]:.4f}' + "</td></tr>" if metrics.get("entailment_rate") is not None else ""}
</table>

<h2>Clusters ({len(data["clusters"])})</h2>
{clusters_html}

<h2>Samples ({len(data["samples"])})</h2>
{samples_html}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
