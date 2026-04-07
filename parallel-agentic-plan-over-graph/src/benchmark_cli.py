"""CLI entry point for PAPoG benchmarks.

Usage::

    python -m src.benchmark --scenario deep_research
    python -m src.benchmark --scenario failure_recovery
    python -m src.benchmark --goal "Build a chatbot" --workers 8
    python -m src.benchmark --list
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

from src.benchmark import BenchmarkRunner, format_result


def _list_scenarios() -> list[str]:
    """Discover scenario modules under ``data.scenarios``."""
    import data.scenarios as pkg

    names: list[str] = []
    for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if not modname.startswith("_"):
            names.append(modname)
    return sorted(names)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run a benchmark scenario."""
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Run PAPoG benchmark scenarios",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Name of a built-in scenario (e.g. deep_research, failure_recovery)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Custom goal string (architect will decompose it)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread-pool size (default: 4)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max re-plan retries per node (default: 2)",
    )
    parser.add_argument(
        "--no-replan",
        action="store_true",
        help="Disable dynamic re-planning",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List available built-in scenarios and exit",
    )
    args = parser.parse_args(argv)

    if args.list_scenarios:
        scenarios = _list_scenarios()
        print("Available scenarios:")
        for s in scenarios:
            print(f"  - {s}")
        return

    if args.scenario is None and args.goal is None:
        parser.error("Provide --scenario <name> or --goal '<text>'")

    runner = BenchmarkRunner(
        max_workers=args.workers,
        max_retries=args.retries,
        enable_replan=not args.no_replan,
    )

    if args.scenario:
        result = runner.run_named_scenario(args.scenario)
    else:
        result = runner.run_scenario(goal=args.goal, scenario_name="custom")

    print(format_result(result))


if __name__ == "__main__":
    main()
