"""Main CLI entry point for PAPoG.

Usage::

    python -m src.cli run --goal "Build a chatbot" --workers 4
    python -m src.cli visualize --goal "Build a chatbot" --output graph.dot
    python -m src.cli scenario --name deep_research
    python -m src.cli list-scenarios

Can also be invoked as ``papog`` if installed as a console_scripts entry.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.benchmark import BenchmarkRunner, format_result
from src.visualizer import export_dot, export_png, export_svg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _handle_run(args: argparse.Namespace) -> int:
    """Run a goal through the full PAPoG pipeline."""
    runner = BenchmarkRunner(
        max_workers=args.workers,
        max_retries=args.max_retries,
    )
    result = runner.run_scenario(goal=args.goal, scenario_name="cli-run")

    print(format_result(result))

    # Optionally save graph visualization
    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        dot_path = str(outdir / "graph.dot")
        from src.architect import Architect

        architect = Architect()
        graph = architect.decompose(args.goal)
        export_dot(graph, dot_path)
        print(f"\nGraph exported to: {dot_path}")

    return 0 if result.nodes_failed == 0 else 1


def _handle_visualize(args: argparse.Namespace) -> int:
    """Decompose a goal and visualize the resulting task graph."""
    from src.architect import Architect

    architect = Architect()
    graph = architect.decompose(args.goal)

    output = args.output or "graph"
    fmt = args.format

    if fmt == "dot":
        if not output.endswith(".dot"):
            output += ".dot"
        export_dot(graph, output)
        print(f"DOT graph written to: {output}")
    elif fmt == "png":
        if not output.endswith(".png"):
            output += ".png"
        try:
            result_path = export_png(graph, output)
            print(f"PNG graph written to: {result_path}")
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            # Fall back to DOT
            fallback = output.replace(".png", ".dot")
            export_dot(graph, fallback)
            print(f"Fell back to DOT export: {fallback}")
            return 1
    elif fmt == "svg":
        if not output.endswith(".svg"):
            output += ".svg"
        try:
            result_path = export_svg(graph, output)
            print(f"SVG graph written to: {result_path}")
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            fallback = output.replace(".svg", ".dot")
            export_dot(graph, fallback)
            print(f"Fell back to DOT export: {fallback}")
            return 1

    return 0


def _handle_scenario(args: argparse.Namespace) -> int:
    """Run a predefined benchmark scenario."""
    runner = BenchmarkRunner(
        max_workers=args.workers,
    )
    try:
        result = runner.run_named_scenario(args.name)
    except (ModuleNotFoundError, ImportError) as exc:
        print(f"Error loading scenario '{args.name}': {exc}", file=sys.stderr)
        return 1

    print(format_result(result))
    return 0 if result.nodes_failed == 0 else 1


def _handle_list_scenarios(args: argparse.Namespace) -> int:
    """List available benchmark scenarios."""
    try:
        scenarios = BenchmarkRunner.list_scenarios()
    except Exception as exc:
        print(f"Error discovering scenarios: {exc}", file=sys.stderr)
        return 1

    if not scenarios:
        print("No scenarios found.")
        return 0

    print("Available scenarios:")
    for name in scenarios:
        print(f"  • {name}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="papog",
        description="PAPoG — Parallel Agentic Plan-over-Graph CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run a goal through the full PAPoG pipeline",
    )
    run_parser.add_argument(
        "--goal", required=True, type=str, help="Goal to execute"
    )
    run_parser.add_argument(
        "--workers", type=int, default=4, help="Worker thread count (default: 4)"
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max re-plan retries per node (default: 2)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save graph visualization",
    )

    # --- visualize ---------------------------------------------------------
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Decompose a goal and visualize the task graph",
    )
    viz_parser.add_argument(
        "--goal", required=True, type=str, help="Goal to decompose"
    )
    viz_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: graph.<format>)",
    )
    viz_parser.add_argument(
        "--format",
        type=str,
        choices=["dot", "png", "svg"],
        default="dot",
        help="Output format (default: dot)",
    )

    # --- scenario ----------------------------------------------------------
    scen_parser = subparsers.add_parser(
        "scenario",
        help="Run a predefined benchmark scenario",
    )
    scen_parser.add_argument(
        "--name", required=True, type=str, help="Scenario name"
    )
    scen_parser.add_argument(
        "--workers", type=int, default=4, help="Worker thread count (default: 4)"
    )

    # --- list-scenarios ----------------------------------------------------
    subparsers.add_parser(
        "list-scenarios",
        help="List available benchmark scenarios",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate handler."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "run": _handle_run,
        "visualize": _handle_visualize,
        "scenario": _handle_scenario,
        "list-scenarios": _handle_list_scenarios,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
