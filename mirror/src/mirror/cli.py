"""CLI interface for MIRROR.

Provides command-line access to the MIRROR probing tools
via Click, with Rich-formatted output.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import mirror
from mirror.llm import LLMClient
from mirror.models import CoTSample
from mirror.prober import CycleConsistencyProber

console = Console()


@click.group()
def cli() -> None:
    """MIRROR — Measuring Inference Reliability via Round-trip Observation Reconciliation."""
    pass


@cli.command()
def version() -> None:
    """Print the MIRROR version."""
    console.print(f"MIRROR v{mirror.__version__}")


@cli.command()
@click.option("--model", default="gpt-4", help="Model to use for probing.")
@click.option("--input", "input_prompt", required=True, help="The original input/question.")
@click.option("--cot", required=True, help="The chain-of-thought explanation.")
@click.option("--answer", required=True, help="The final answer.")
@click.option(
    "--cross-model",
    "cross_model",
    default=None,
    help="Different model for cross-model consistency test.",
)
def probe(
    model: str,
    input_prompt: str,
    cot: str,
    answer: str,
    cross_model: str | None,
) -> None:
    """Probe a chain-of-thought explanation for faithfulness."""
    sample = CoTSample(
        input_prompt=input_prompt,
        cot_explanation=cot,
        answer=answer,
        model_id=model,
    )

    llm = LLMClient(default_model=model)
    cross_llm = LLMClient(default_model=cross_model) if cross_model else None
    prober = CycleConsistencyProber(llm=llm, cross_model_llm=cross_llm)

    console.print(
        Panel(
            f"[bold]Input:[/bold] {input_prompt}\n"
            f"[bold]CoT:[/bold] {cot}\n"
            f"[bold]Answer:[/bold] {answer}\n"
            f"[bold]Model:[/bold] {model}",
            title="🔍 MIRROR Probe",
            border_style="cyan",
        )
    )

    try:
        result = prober.probe(sample)
    except Exception as e:
        console.print(f"[red]Error during probing:[/red] {e}")
        raise SystemExit(1) from e

    _display_result(result)


def _display_result(result: "mirror.models.ProbeResult") -> None:
    """Display a ProbeResult using Rich formatting."""
    from mirror.models import ProbeResult  # noqa: F811 — avoid circular at module level

    table = Table(title="Probe Results", show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", min_width=15)
    table.add_column("Consistent", justify="center", min_width=12)
    table.add_column("Score", justify="right", min_width=8)

    def _icon(b: bool) -> str:
        return "✅" if b else "❌"

    table.add_row(
        "Forward",
        _icon(result.forward_consistent),
        f"{result.scores.get('forward', 0.0):.2f}",
    )
    table.add_row(
        "Reverse",
        _icon(result.reverse_consistent),
        f"{result.scores.get('reverse', 0.0):.2f}",
    )
    table.add_row(
        "Cross-Model",
        _icon(result.cross_model_consistent),
        f"{result.scores.get('cross_model', 0.0):.2f}",
    )

    console.print(table)

    overall = result.overall_score
    color = "green" if overall >= 0.7 else "yellow" if overall >= 0.4 else "red"
    verdict = "FAITHFUL" if result.is_faithful else "UNFAITHFUL"

    console.print(
        Panel(
            f"[bold {color}]{verdict}[/bold {color}]  —  Overall score: {overall:.2f}",
            border_style=color,
        )
    )
