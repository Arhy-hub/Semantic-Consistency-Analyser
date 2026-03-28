"""CLI entrypoint for sca — Typer-based."""

from __future__ import annotations

import importlib.util
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="sca",
    help="Semantic Consistency Analyzer — measure LLM output consistency.",
    add_completion=True,
)
console = Console()


def _load_backend_from_path(backend_path: str):
    """Load a backend callable from 'path/to/file.py::function_name' syntax."""
    if "::" not in backend_path:
        raise typer.BadParameter(
            f"Backend path must use 'file.py::function_name' syntax, got: {backend_path}"
        )
    file_path, func_name = backend_path.rsplit("::", 1)
    spec = importlib.util.spec_from_file_location("_sca_backend", file_path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Cannot load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, func_name):
        raise typer.BadParameter(f"Function '{func_name}' not found in {file_path}")
    return getattr(module, func_name)


def _print_results(results) -> None:
    """Print a formatted text summary of results to stdout."""
    from sca.core.metrics import Metrics  # noqa: PLC0415

    m = results.metrics

    console.print(Panel(
        f"[bold white]Model:[/bold white] {results.config.get('model', 'custom backend')}\n"
        f"[bold white]Prompt:[/bold white] {results.config.get('prompt', '')[:100]}\n"
        f"[bold white]Samples:[/bold white] {len(results.samples)}\n"
        f"[bold white]Timestamp:[/bold white] {results.timestamp}",
        title="[bold]Semantic Consistency Analysis[/bold]",
        border_style="white",
    ))

    table = Table(title="Metrics", show_header=True)
    table.add_column("Metric", style="white")
    table.add_column("Value", style="white")

    table.add_row("Mean Pairwise Similarity", f"{m.mean_pairwise_similarity:.4f}")
    table.add_row("Semantic Entropy", f"{m.semantic_entropy:.4f}")
    table.add_row("Cluster Count", str(m.cluster_count))
    table.add_row("Silhouette Score", f"{m.silhouette_score:.4f}")
    table.add_row("Centroid Distance Variance", f"{m.centroid_distance_variance:.6f}")
    if m.entailment_rate is not None:
        table.add_row("Entailment Rate", f"{m.entailment_rate:.4f}")

    console.print(table)

    if results.clusters:
        console.print("\n[bold]Clusters:[/bold]")
        for cluster in results.clusters:
            summary = cluster.summary or f"Cluster {cluster.id}"
            console.print(f"  [white]#{cluster.id}[/white] ({len(cluster.members)} samples): {summary}")

    console.print(f"\n[dim]Samples:[/dim]")
    for i, sample in enumerate(results.samples):
        truncated = sample[:200] + "..." if len(sample) > 200 else sample
        console.print(f"  [dim]{i+1}.[/dim] {truncated}")


def _do_export(results, export: str | None, out: str | None) -> None:
    """Handle export based on --export and --out flags."""
    if not export:
        return
    if not out:
        ext_map = {"json": "results.json", "png": "scatter.png", "html": "report.html"}
        out = ext_map.get(export, f"results.{export}")
        console.print(f"[yellow]No --out specified, using: {out}[/yellow]")

    if export == "json":
        from sca.export.json import export_json  # noqa: PLC0415
        export_json(results, out)
        console.print(f"[green]Exported JSON to: {out}[/green]")
    elif export == "png":
        from sca.export.png import export_png  # noqa: PLC0415
        export_png(results, out)
        console.print(f"[green]Exported PNG to: {out}[/green]")
    elif export == "html":
        from sca.export.html import export_results_html  # noqa: PLC0415
        export_results_html(results, out)
        console.print(f"[green]Exported HTML to: {out}[/green]")
    else:
        console.print(f"[red]Unknown export format: {export}[/red]")


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to sample from the model."),
    model: Optional[str] = typer.Option(None, help="Model name (via LiteLLM)."),
    backend_path: Optional[str] = typer.Option(
        None, "--backend", help="Custom backend: 'path/to/file.py::function_name'."
    ),
    n: int = typer.Option(20, help="Number of samples."),
    temperature: float = typer.Option(0.9, help="Sampling temperature."),
    nli: bool = typer.Option(False, help="Enable NLI entailment rate computation."),
    export: Optional[str] = typer.Option(None, help="Export format: json, png, html."),
    out: Optional[str] = typer.Option(None, help="Output file path for export."),
    no_tui: bool = typer.Option(False, "--no-tui", help="Print results to stdout instead of TUI."),
    convergence_threshold: Optional[float] = typer.Option(
        None, help="Stop early if entropy delta < threshold for 3 windows."
    ),
    measure: str = typer.Option(
        "cosine", help="Heatmap measure: cosine, agreement."
    ),
) -> None:
    """Run N samples and analyze semantic consistency."""
    backend = None
    if backend_path:
        backend = _load_backend_from_path(backend_path)

    from sca.core.analyzer import SemanticConsistencyAnalyzer  # noqa: PLC0415

    analyzer = SemanticConsistencyAnalyzer(
        prompt=prompt,
        model=model,
        backend=backend,
        n=n,
        temperature=temperature,
        nli=nli,
        convergence_threshold=convergence_threshold,
    )

    if no_tui:
        with console.status("[bold green]Sampling..."):
            results = analyzer.run()
        _print_results(results)
        _do_export(results, export, out)
    else:
        from sca.tui.app import SCAApp  # noqa: PLC0415

        tui_app = SCAApp(analyzer, measure=measure)
        tui_app.run()
        results = tui_app.get_results()

        # After TUI exits, handle export if requested
        if results:
            _do_export(results, export, out)
        else:
            # Re-run without TUI if TUI exited early
            console.print("[yellow]TUI closed. Re-running for export...[/yellow]")
            results = analyzer.run()
            _do_export(results, export, out)


@app.command()
def sweep(
    prompt: str = typer.Argument(..., help="The prompt to analyze."),
    model: Optional[str] = typer.Option(None, help="Model name (via LiteLLM)."),
    temps: str = typer.Option("0.0,0.3,0.7,1.0,1.5", help="Comma-separated temperatures."),
    n: int = typer.Option(20, help="Number of samples per temperature."),
    export: Optional[str] = typer.Option(None, help="Export format: json, png, html."),
    out: Optional[str] = typer.Option(None, help="Output file path for export."),
) -> None:
    """Temperature sweep — analyze consistency across multiple temperatures."""
    temperature_list = [float(t.strip()) for t in temps.split(",")]

    from sca.core.analyzer import SemanticConsistencyAnalyzer  # noqa: PLC0415

    all_results: list[tuple[float, object]] = []

    for temp in temperature_list:
        console.print(f"[cyan]Sampling at temperature={temp}...[/cyan]")
        analyzer = SemanticConsistencyAnalyzer(
            prompt=prompt,
            model=model,
            n=n,
            temperature=temp,
        )
        with console.status(f"[bold green]Sampling (temp={temp})..."):
            results = analyzer.run()
        all_results.append((temp, results))
        console.print(
            f"  entropy={results.metrics.semantic_entropy:.3f}, "
            f"clusters={results.metrics.cluster_count}, "
            f"mean_sim={results.metrics.mean_pairwise_similarity:.3f}"
        )

    # Summary table
    table = Table(title="Temperature Sweep Results", show_header=True)
    table.add_column("Temperature", style="cyan")
    table.add_column("Semantic Entropy", style="white")
    table.add_column("Cluster Count", style="yellow")
    table.add_column("Mean Pairwise Sim", style="green")
    table.add_column("Silhouette", style="blue")

    for temp, results in all_results:
        m = results.metrics
        table.add_row(
            str(temp),
            f"{m.semantic_entropy:.4f}",
            str(m.cluster_count),
            f"{m.mean_pairwise_similarity:.4f}",
            f"{m.silhouette_score:.4f}",
        )

    console.print(table)

    # Export the last result or all results as a combined JSON
    if export and out:
        if export == "json":
            import json  # noqa: PLC0415
            combined = [
                {"temperature": temp, "results": r.to_dict()}
                for temp, r in all_results
            ]
            with open(out, "w") as f:
                json.dump(combined, f, indent=2)
            console.print(f"[green]Exported sweep JSON to: {out}[/green]")
        else:
            # Export the last result
            _do_export(all_results[-1][1], export, out)


@app.command()
def compare(
    prompt: str = typer.Argument(..., help="The prompt to analyze."),
    models: str = typer.Argument(..., help="Comma-separated model names."),
    n: int = typer.Option(20, help="Number of samples per model."),
    temperature: float = typer.Option(0.9, help="Sampling temperature."),
    export: Optional[str] = typer.Option(None, help="Export format: json, png, html."),
    out: Optional[str] = typer.Option(None, help="Output file path for export."),
) -> None:
    """Model comparison — analyze consistency across multiple models."""
    model_list = [m.strip() for m in models.split(",")]

    from sca.core.analyzer import SemanticConsistencyAnalyzer  # noqa: PLC0415

    all_results: list[tuple[str, object]] = []

    for model_name in model_list:
        console.print(f"[cyan]Sampling with model={model_name}...[/cyan]")
        analyzer = SemanticConsistencyAnalyzer(
            prompt=prompt,
            model=model_name,
            n=n,
            temperature=temperature,
        )
        with console.status(f"[bold green]Sampling ({model_name})..."):
            results = analyzer.run()
        all_results.append((model_name, results))
        console.print(
            f"  entropy={results.metrics.semantic_entropy:.3f}, "
            f"clusters={results.metrics.cluster_count}, "
            f"mean_sim={results.metrics.mean_pairwise_similarity:.3f}"
        )

    # Summary table
    table = Table(title="Model Comparison Results", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Semantic Entropy", style="white")
    table.add_column("Cluster Count", style="yellow")
    table.add_column("Mean Pairwise Sim", style="green")
    table.add_column("Silhouette", style="blue")

    for model_name, results in all_results:
        m = results.metrics
        table.add_row(
            model_name,
            f"{m.semantic_entropy:.4f}",
            str(m.cluster_count),
            f"{m.mean_pairwise_similarity:.4f}",
            f"{m.silhouette_score:.4f}",
        )

    console.print(table)

    if export and out:
        if export == "json":
            import json  # noqa: PLC0415
            combined = [
                {"model": model_name, "results": r.to_dict()}
                for model_name, r in all_results
            ]
            with open(out, "w") as f:
                json.dump(combined, f, indent=2)
            console.print(f"[green]Exported comparison JSON to: {out}[/green]")
        else:
            _do_export(all_results[-1][1], export, out)


if __name__ == "__main__":
    app()
