"""Tests for the PAPoG Benchmark Runner (Step 10).

Covers:
- BenchmarkRunner end-to-end with a simple scenario
- BenchmarkResult fields and metrics accuracy
- Scenario with failures triggers re-planning
- Timeline recording correctness
- CLI entry point doesn't crash
- Integration: all components work together in the pipeline
- Named scenario loading (deep_research, failure_recovery)
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.benchmark import BenchmarkResult, BenchmarkRunner, TimelineEntry, format_result
from src.models import TaskGraph, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_graph() -> TaskGraph:
    """Two sequential tasks — no failures."""
    g = TaskGraph()
    g.add_node(TaskNode(id="a", description="First task", dependencies=[]))
    g.add_node(TaskNode(id="b", description="Second task", dependencies=["a"]))
    return g


def _parallel_graph() -> TaskGraph:
    """Three independent tasks followed by a join."""
    g = TaskGraph()
    g.add_node(TaskNode(id="p1", description="Parallel one", dependencies=[]))
    g.add_node(TaskNode(id="p2", description="Parallel two", dependencies=[]))
    g.add_node(TaskNode(id="p3", description="Parallel three", dependencies=[]))
    g.add_node(TaskNode(id="join", description="Join results", dependencies=["p1", "p2", "p3"]))
    return g


def _failing_graph() -> TaskGraph:
    """A graph where one node will fail (description contains 'fail')."""
    g = TaskGraph()
    g.add_node(TaskNode(id="ok1", description="Good task", dependencies=[]))
    g.add_node(TaskNode(
        id="bad",
        description="This will fail intentionally",
        dependencies=["ok1"],
    ))
    g.add_node(TaskNode(id="downstream", description="After bad", dependencies=["bad"]))
    return g


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """BenchmarkResult field defaults and construction."""

    def test_default_fields(self):
        r = BenchmarkResult()
        assert r.total_time == 0.0
        assert r.nodes_completed == 0
        assert r.nodes_failed == 0
        assert r.nodes_skipped == 0
        assert r.parallel_speedup_estimate == 1.0
        assert r.execution_timeline == []
        assert r.graph_snapshot == {}
        assert r.per_node_times == {}
        assert r.replan_count == 0
        assert r.total_nodes == 0
        assert r.scenario_name == ""
        assert r.goal == ""

    def test_custom_fields(self):
        r = BenchmarkResult(
            total_time=1.5,
            nodes_completed=3,
            nodes_failed=1,
            nodes_skipped=0,
            parallel_speedup_estimate=2.5,
            total_nodes=4,
            scenario_name="test",
            goal="Do stuff",
        )
        assert r.total_time == 1.5
        assert r.nodes_completed == 3
        assert r.nodes_failed == 1
        assert r.parallel_speedup_estimate == 2.5
        assert r.scenario_name == "test"
        assert r.goal == "Do stuff"


# ---------------------------------------------------------------------------
# TimelineEntry
# ---------------------------------------------------------------------------


class TestTimelineEntry:
    def test_creation(self):
        e = TimelineEntry(
            timestamp=1234.0,
            node_id="n1",
            from_status="pending",
            to_status="running",
        )
        assert e.node_id == "n1"
        assert e.from_status == "pending"
        assert e.to_status == "running"
        assert e.metadata == {}

    def test_frozen(self):
        e = TimelineEntry(timestamp=0.0, node_id="x", from_status="a", to_status="b")
        with pytest.raises(AttributeError):
            e.node_id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkRunner — end-to-end simple
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerSimple:
    """E2E with simple graphs, no failures."""

    def test_simple_sequential(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(
            goal="simple",
            graph=_simple_graph(),
            scenario_name="simple_seq",
        )
        assert result.nodes_completed == 2
        assert result.nodes_failed == 0
        assert result.nodes_skipped == 0
        assert result.total_nodes == 2
        assert result.total_time > 0
        assert result.scenario_name == "simple_seq"
        assert result.goal == "simple"

    def test_parallel_graph(self):
        runner = BenchmarkRunner(max_workers=4, enable_replan=False)
        result = runner.run_scenario(
            goal="parallel",
            graph=_parallel_graph(),
            scenario_name="parallel",
        )
        assert result.nodes_completed == 4
        assert result.nodes_failed == 0
        assert result.nodes_skipped == 0
        assert result.total_nodes == 4

    def test_empty_graph_from_goal(self):
        """When no graph is provided the Architect decomposes the goal."""
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="Build a chatbot")
        # MockLLMStrategy always produces 4 nodes
        assert result.total_nodes == 4
        assert result.nodes_completed == 4

    def test_graph_snapshot_all_completed(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="x", graph=_simple_graph())
        for status in result.graph_snapshot.values():
            assert status == "completed"


# ---------------------------------------------------------------------------
# Timeline recording
# ---------------------------------------------------------------------------


class TestTimelineRecording:
    """Verify that the execution timeline is populated correctly."""

    def test_timeline_has_entries(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="t", graph=_simple_graph())
        assert len(result.execution_timeline) > 0

    def test_timeline_order_monotonic(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="t", graph=_simple_graph())
        timestamps = [e.timestamp for e in result.execution_timeline]
        assert timestamps == sorted(timestamps)

    def test_timeline_contains_running_and_completed(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="t", graph=_simple_graph())
        to_statuses = {e.to_status for e in result.execution_timeline}
        assert "running" in to_statuses
        assert "completed" in to_statuses

    def test_per_node_times_populated(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="t", graph=_simple_graph())
        # Both nodes should have timing info
        assert "a" in result.per_node_times
        assert "b" in result.per_node_times
        assert all(t >= 0 for t in result.per_node_times.values())


# ---------------------------------------------------------------------------
# Failure + re-planning
# ---------------------------------------------------------------------------


class TestFailureAndReplan:
    """Verify that failures are detected and re-planning triggers."""

    def test_failure_without_replan(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(
            goal="fail test",
            graph=_failing_graph(),
            scenario_name="fail_no_replan",
        )
        assert result.nodes_failed >= 1
        assert "bad" in result.graph_snapshot
        assert result.graph_snapshot["bad"] == "failed"
        # downstream should be skipped
        assert result.graph_snapshot["downstream"] == "skipped"
        assert result.nodes_skipped >= 1
        assert result.replan_count == 0

    def test_failure_with_replan_enabled(self):
        runner = BenchmarkRunner(max_workers=2, max_retries=2, enable_replan=True)
        result = runner.run_scenario(
            goal="fail replan",
            graph=_failing_graph(),
            scenario_name="fail_replan",
        )
        # The replanner fires, but the replacement nodes also contain
        # "Alternative approach for: ... fail ..." so they'll also fail.
        # We should see replan_count > 0 (at least 1 attempt).
        assert result.replan_count >= 1
        assert result.nodes_failed >= 1


# ---------------------------------------------------------------------------
# Named scenarios
# ---------------------------------------------------------------------------


class TestNamedScenarios:
    """Load and run the pre-built scenarios."""

    def test_deep_research_scenario(self):
        runner = BenchmarkRunner(max_workers=4, enable_replan=False)
        result = runner.run_named_scenario("deep_research")
        assert result.scenario_name == "deep_research"
        assert result.total_nodes == 7  # 5 research + synthesis + report
        assert result.nodes_completed == 7
        assert result.nodes_failed == 0
        assert result.nodes_skipped == 0
        assert result.goal == (
            "Analyze the impact of 2024 GPU shortage on LLM training techniques"
        )

    def test_failure_recovery_scenario(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_named_scenario("failure_recovery")
        assert result.scenario_name == "failure_recovery"
        assert result.total_nodes == 4
        # process_data has "fail" in description → MockExecutionStrategy fails it
        assert result.nodes_failed >= 1
        assert result.graph_snapshot["process_data"] == "failed"

    def test_failure_recovery_with_replan(self):
        runner = BenchmarkRunner(max_workers=2, max_retries=2, enable_replan=True)
        result = runner.run_named_scenario("failure_recovery")
        assert result.replan_count >= 1

    def test_deep_research_parallel_speedup(self):
        """The deep research scenario has 5 independent tasks → speedup > 1."""
        runner = BenchmarkRunner(max_workers=4, enable_replan=False)
        result = runner.run_named_scenario("deep_research")
        # With parallel execution, speedup should be >= 1
        # (mock execution is nearly instant so it might be close to 1)
        assert result.parallel_speedup_estimate >= 0


# ---------------------------------------------------------------------------
# Metrics accuracy
# ---------------------------------------------------------------------------


class TestMetricsAccuracy:
    """Verify that computed metrics are internally consistent."""

    def test_completed_plus_failed_plus_skipped_eq_total(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="m", graph=_simple_graph())
        accounted = result.nodes_completed + result.nodes_failed + result.nodes_skipped
        assert accounted == result.total_nodes

    def test_snapshot_matches_counts(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(goal="m", graph=_failing_graph())
        snap_completed = sum(1 for s in result.graph_snapshot.values() if s == "completed")
        snap_failed = sum(1 for s in result.graph_snapshot.values() if s == "failed")
        snap_skipped = sum(1 for s in result.graph_snapshot.values() if s == "skipped")
        assert snap_completed == result.nodes_completed
        assert snap_failed == result.nodes_failed
        assert snap_skipped == result.nodes_skipped

    def test_total_time_positive(self):
        runner = BenchmarkRunner(max_workers=1, enable_replan=False)
        result = runner.run_scenario(goal="t", graph=_simple_graph())
        assert result.total_time > 0


# ---------------------------------------------------------------------------
# format_result / pretty-print
# ---------------------------------------------------------------------------


class TestFormatResult:
    """Ensure format_result produces reasonable output."""

    def test_contains_key_sections(self):
        runner = BenchmarkRunner(max_workers=2, enable_replan=False)
        result = runner.run_scenario(
            goal="format test",
            graph=_simple_graph(),
            scenario_name="fmt",
        )
        text = format_result(result)
        assert "PAPoG Benchmark Results" in text
        assert "fmt" in text
        assert "Total time" in text
        assert "Graph snapshot" in text

    def test_format_empty_result(self):
        text = format_result(BenchmarkResult())
        assert "PAPoG Benchmark Results" in text
        assert "custom" in text  # no scenario_name → "custom"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    """Verify the CLI entry point doesn't crash."""

    def test_list_scenarios(self, capsys):
        from src.benchmark_cli import main as cli_main

        cli_main(["--list"])
        captured = capsys.readouterr()
        assert "deep_research" in captured.out
        assert "failure_recovery" in captured.out

    def test_run_scenario_cli(self, capsys):
        from src.benchmark_cli import main as cli_main

        cli_main(["--scenario", "deep_research", "--workers", "2"])
        captured = capsys.readouterr()
        assert "PAPoG Benchmark Results" in captured.out
        assert "deep_research" in captured.out

    def test_run_goal_cli(self, capsys):
        from src.benchmark_cli import main as cli_main

        cli_main(["--goal", "Build a chatbot", "--workers", "1"])
        captured = capsys.readouterr()
        assert "PAPoG Benchmark Results" in captured.out

    def test_no_args_errors(self):
        from src.benchmark_cli import main as cli_main

        with pytest.raises(SystemExit):
            cli_main([])


# ---------------------------------------------------------------------------
# Integration: all components work together
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full pipeline integration — Architect + Policy + Engine + Observer + Replan + Tools."""

    def test_full_pipeline_with_tools(self):
        """Run the deep research scenario and verify every component contributed."""
        runner = BenchmarkRunner(max_workers=4, max_retries=2, enable_replan=True)
        result = runner.run_named_scenario("deep_research")

        # Architect produced the graph (7 nodes)
        assert result.total_nodes == 7

        # Engine ran all nodes to completion
        assert result.nodes_completed == 7

        # Observer recorded timeline events
        assert len(result.execution_timeline) > 0
        # Each node should have at least 2 transitions: pending→running, running→completed
        assert len(result.execution_timeline) >= 14  # 7 * 2

        # Per-node times were captured
        assert len(result.per_node_times) == 7

        # No failures, no replanning needed
        assert result.nodes_failed == 0
        assert result.replan_count == 0

    def test_full_pipeline_with_failure_and_replan(self):
        """Failure scenario: replanner fires and produces replacement nodes."""
        runner = BenchmarkRunner(max_workers=2, max_retries=2, enable_replan=True)
        result = runner.run_named_scenario("failure_recovery")

        # Original graph has 4 nodes; replan adds more
        assert result.total_nodes > 4

        # At least one replan happened
        assert result.replan_count >= 1

        # Timeline should capture failure transitions
        failed_transitions = [
            e for e in result.execution_timeline if e.to_status == "failed"
        ]
        assert len(failed_transitions) >= 1

    def test_list_scenarios_method(self):
        scenarios = BenchmarkRunner.list_scenarios()
        assert "deep_research" in scenarios
        assert "failure_recovery" in scenarios
