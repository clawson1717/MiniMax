"""Benchmark Runner for PAPoG.

Orchestrates the full PAPoG pipeline end-to-end:
Architect → PolicyController → ExecutionEngine → Observer → RePlanner.

Collects metrics (wall-clock time, per-node times, parallelism ratio,
failure/recovery stats) and returns a ``BenchmarkResult`` dataclass.

Can also be invoked via ``python -m src.benchmark --scenario <name>``.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

from src.architect import Architect, DecompositionStrategy, MockLLMStrategy
from src.engine import ExecutionEngine
from src.models import TaskGraph, TaskNode, TaskStatus
from src.observer import EventType, GraphObserver, ObservedPolicyController, StateTransition
from src.policy import PolicyController
from src.replan import DynamicRePlanner
from src.tools import MemoryTool, PythonREPLTool, SearchTool, ToolRegistry
from src.tools.registry import ToolRegistryProvider
from src.worker import MockExecutionStrategy, ReasoningWorker


# ---------------------------------------------------------------------------
# Timeline entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineEntry:
    """One event in the execution timeline."""

    timestamp: float
    node_id: str
    from_status: str
    to_status: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated result of a benchmark run.

    Attributes
    ----------
    total_time:
        Wall-clock time for the entire run (seconds).
    nodes_completed:
        Number of nodes that reached COMPLETED status.
    nodes_failed:
        Number of nodes that reached FAILED status (excluding replan
        replacements).
    nodes_skipped:
        Number of nodes that were SKIPPED.
    parallel_speedup_estimate:
        ``sum(per_node_times) / total_time`` — a rough measure of
        how much parallelism helped.
    execution_timeline:
        Chronological list of ``TimelineEntry`` objects.
    graph_snapshot:
        Final ``{node_id: status}`` mapping.
    per_node_times:
        ``{node_id: seconds}`` for each node's execution time.
    replan_count:
        How many re-plan attempts were triggered.
    total_nodes:
        Total number of nodes in the (possibly expanded) graph at the
        end of execution.
    scenario_name:
        Human-readable name of the scenario that was run.
    goal:
        The goal string that was benchmarked.
    """

    total_time: float = 0.0
    nodes_completed: int = 0
    nodes_failed: int = 0
    nodes_skipped: int = 0
    parallel_speedup_estimate: float = 1.0
    execution_timeline: list[TimelineEntry] = field(default_factory=list)
    graph_snapshot: dict[str, str] = field(default_factory=dict)
    per_node_times: dict[str, float] = field(default_factory=dict)
    replan_count: int = 0
    total_nodes: int = 0
    scenario_name: str = ""
    goal: str = ""


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates the full PAPoG pipeline for benchmarking.

    Parameters
    ----------
    max_workers:
        Thread-pool size for ``ExecutionEngine``.
    max_retries:
        Max re-plan attempts per failed node.
    enable_replan:
        If ``True`` (default) a ``DynamicRePlanner`` is wired up.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_retries: int = 2,
        enable_replan: bool = True,
    ) -> None:
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._enable_replan = enable_replan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        goal: str,
        graph: TaskGraph | None = None,
        scenario_name: str = "",
    ) -> BenchmarkResult:
        """Run the full PAPoG pipeline and return metrics.

        Parameters
        ----------
        goal:
            The high-level objective.  If *graph* is ``None``, the
            ``Architect`` will decompose this goal into a ``TaskGraph``.
        graph:
            Optional pre-built ``TaskGraph``.  When supplied the
            architect step is skipped.
        scenario_name:
            Label attached to the result for identification.

        Returns
        -------
        BenchmarkResult
        """
        t0 = time.monotonic()

        # --- tools ---------------------------------------------------------
        registry = self._build_tool_registry()
        provider = ToolRegistryProvider(registry)

        # --- architect (decompose goal → graph) ----------------------------
        architect = Architect()  # uses MockLLMStrategy by default
        if graph is None:
            graph = architect.decompose(goal)

        # --- replanner ----------------------------------------------------
        replanner: DynamicRePlanner | None = None
        replan_callback = None
        if self._enable_replan:
            replanner = DynamicRePlanner(
                architect=architect, max_retries=self._max_retries
            )
            replan_callback = replanner.as_replan_callback()

        # --- observer -----------------------------------------------------
        observer = GraphObserver(
            graph,
            critical_threshold=1,
            replan_callback=replan_callback,
        )

        # Collect timeline entries via listener
        timeline: list[TimelineEntry] = []
        node_start_times: dict[str, float] = {}
        per_node_times: dict[str, float] = {}

        def _on_event(event):
            if event.transition is not None:
                t = event.transition
                timeline.append(TimelineEntry(
                    timestamp=t.timestamp,
                    node_id=t.node_id,
                    from_status=t.from_status.value,
                    to_status=t.to_status.value,
                    metadata=dict(t.metadata),
                ))
                # Track per-node times
                if t.to_status == TaskStatus.RUNNING:
                    node_start_times[t.node_id] = t.timestamp
                elif t.to_status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    start = node_start_times.get(t.node_id)
                    if start is not None:
                        per_node_times[t.node_id] = t.timestamp - start

        observer.add_listener(_on_event)

        # --- policy (observed) --------------------------------------------
        policy = PolicyController(graph)
        observed_policy = observer.observe_policy(policy)

        # --- worker -------------------------------------------------------
        worker = ReasoningWorker(
            agent_id="benchmark-worker",
            strategy=MockExecutionStrategy(),
            tool_provider=provider,
        )

        # Build an engine-compatible callable that uses the worker
        def _worker_fn(node: TaskNode) -> Any:
            # Reset status back to PENDING so ReasoningWorker accepts it
            # (the engine calls assign_next which sets RUNNING via policy;
            # we need the underlying worker to also do its bookkeeping).
            # However, ReasoningWorker.execute expects PENDING. We'll
            # bypass and call the strategy directly to avoid the status
            # conflict, since the policy already handles transitions.
            tools = provider.get_tools() if provider else None
            return worker.strategy.execute(node, tools=tools)

        # --- engine -------------------------------------------------------
        engine = ExecutionEngine(
            worker=_worker_fn,
            max_workers=self._max_workers,
            poll_interval=0.02,
        )

        exec_result = engine.execute_with_policy(graph, observed_policy)

        total_time = time.monotonic() - t0

        # --- metrics ------------------------------------------------------
        snapshot = observer.get_snapshot()
        n_completed = sum(1 for s in snapshot.values() if s == TaskStatus.COMPLETED.value)
        n_failed = sum(1 for s in snapshot.values() if s == TaskStatus.FAILED.value)
        n_skipped = sum(1 for s in snapshot.values() if s == TaskStatus.SKIPPED.value)

        sum_node_times = sum(per_node_times.values()) if per_node_times else 0.0
        speedup = (sum_node_times / total_time) if total_time > 0 else 1.0

        replan_count = len(replanner.history) if replanner else 0

        return BenchmarkResult(
            total_time=total_time,
            nodes_completed=n_completed,
            nodes_failed=n_failed,
            nodes_skipped=n_skipped,
            parallel_speedup_estimate=round(speedup, 3),
            execution_timeline=timeline,
            graph_snapshot=snapshot,
            per_node_times=per_node_times,
            replan_count=replan_count,
            total_nodes=len(graph.nodes),
            scenario_name=scenario_name,
            goal=goal,
        )

    def run_named_scenario(self, name: str) -> BenchmarkResult:
        """Load a scenario module from ``data.scenarios.<name>`` and run it.

        The module must expose ``SCENARIO_GOAL`` (str) and
        ``build_graph() -> TaskGraph``.

        Parameters
        ----------
        name:
            Module name under ``data.scenarios``.

        Returns
        -------
        BenchmarkResult
        """
        mod = importlib.import_module(f"data.scenarios.{name}")
        goal: str = mod.SCENARIO_GOAL  # type: ignore[attr-defined]
        graph: TaskGraph = mod.build_graph()  # type: ignore[attr-defined]
        return self.run_scenario(goal=goal, graph=graph, scenario_name=name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_registry() -> ToolRegistry:
        """Create a ``ToolRegistry`` populated with the standard tools."""
        registry = ToolRegistry()
        # Clear global memory state between runs
        MemoryTool.clear_global_store()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())
        registry.register(MemoryTool())
        return registry

    @staticmethod
    def list_scenarios() -> list[str]:
        """Discover scenario modules under ``data.scenarios``."""
        import pkgutil

        import data.scenarios as pkg

        names: list[str] = []
        for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
            if not modname.startswith("_"):
                names.append(modname)
        return sorted(names)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def format_result(result: BenchmarkResult) -> str:
    """Return a human-readable summary of a BenchmarkResult."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"  PAPoG Benchmark Results — {result.scenario_name or 'custom'}")
    lines.append("=" * 64)
    lines.append(f"  Goal: {result.goal}")
    lines.append("")
    lines.append("  Summary")
    lines.append("  " + "-" * 40)
    lines.append(f"  Total time       : {result.total_time:.4f} s")
    lines.append(f"  Total nodes      : {result.total_nodes}")
    lines.append(f"  Completed        : {result.nodes_completed}")
    lines.append(f"  Failed           : {result.nodes_failed}")
    lines.append(f"  Skipped          : {result.nodes_skipped}")
    lines.append(f"  Re-plan attempts : {result.replan_count}")
    lines.append(f"  Parallelism est. : {result.parallel_speedup_estimate:.3f}x")
    lines.append("")

    if result.per_node_times:
        lines.append("  Per-node times (seconds)")
        lines.append("  " + "-" * 40)
        for nid, t in sorted(result.per_node_times.items()):
            lines.append(f"    {nid:40s} {t:.6f}")
        lines.append("")

    lines.append("  Graph snapshot")
    lines.append("  " + "-" * 40)
    for nid, status in sorted(result.graph_snapshot.items()):
        marker = {
            "completed": "✓",
            "failed": "✗",
            "skipped": "⊘",
            "running": "▸",
            "pending": "○",
        }.get(status, "?")
        lines.append(f"    {marker} {nid:40s} {status}")
    lines.append("")

    if result.execution_timeline:
        lines.append("  Execution timeline (first 20 events)")
        lines.append("  " + "-" * 40)
        for entry in result.execution_timeline[:20]:
            lines.append(
                f"    [{entry.timestamp:.4f}] {entry.node_id}: "
                f"{entry.from_status} → {entry.to_status}"
            )
        if len(result.execution_timeline) > 20:
            lines.append(f"    ... and {len(result.execution_timeline) - 20} more")
        lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)


if __name__ == "__main__":
    from src.benchmark_cli import main as _cli_main

    _cli_main()
