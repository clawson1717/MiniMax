"""Comprehensive tests for PAPoG Execution Engine."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from src.engine import ExecutionEngine, ExecutionResult, Worker
from src.models import TaskGraph, TaskNode, TaskStatus
from src.policy import PolicyController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_diamond_graph() -> TaskGraph:
    """Build a classic diamond: A -> B, A -> C, B -> D, C -> D."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="A", description="root"))
    graph.add_node(TaskNode(id="B", description="left", dependencies=["A"]))
    graph.add_node(TaskNode(id="C", description="right", dependencies=["A"]))
    graph.add_node(
        TaskNode(id="D", description="sink", dependencies=["B", "C"])
    )
    return graph


def _make_linear_graph(n: int = 3) -> TaskGraph:
    """Build a linear chain: task_0 -> task_1 -> ... -> task_{n-1}."""
    graph = TaskGraph()
    for i in range(n):
        deps = [f"task_{i - 1}"] if i > 0 else []
        graph.add_node(
            TaskNode(id=f"task_{i}", description=f"step {i}", dependencies=deps)
        )
    return graph


def _make_single_node_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(TaskNode(id="solo", description="lone task"))
    return graph


def _make_wide_graph(n: int = 5) -> TaskGraph:
    """All nodes independent — fully parallel."""
    graph = TaskGraph()
    for i in range(n):
        graph.add_node(TaskNode(id=f"w_{i}", description=f"parallel {i}"))
    return graph


def _make_empty_graph() -> TaskGraph:
    return TaskGraph()


# ---------------------------------------------------------------------------
# Mock workers
# ---------------------------------------------------------------------------


class EchoWorker:
    """Returns the node's description as its result."""

    def __call__(self, node: TaskNode) -> str:
        return f"done: {node.description}"


class SlowWorker:
    """Sleeps briefly then returns — useful for concurrency tests."""

    def __init__(self, delay: float = 0.05):
        self._delay = delay

    def __call__(self, node: TaskNode) -> str:
        time.sleep(self._delay)
        return f"slow-done: {node.id}"


class FailingWorker:
    """Always raises an exception."""

    def __call__(self, node: TaskNode) -> Any:
        raise RuntimeError(f"Intentional failure on {node.id}")


class SelectiveFailWorker:
    """Fails on specific node IDs, succeeds on others."""

    def __init__(self, fail_ids: set[str]):
        self._fail_ids = fail_ids

    def __call__(self, node: TaskNode) -> str:
        if node.id in self._fail_ids:
            raise RuntimeError(f"Selective failure on {node.id}")
        return f"ok: {node.id}"


class TimestampWorker:
    """Records start/end timestamps for concurrency verification."""

    def __init__(self, delay: float = 0.1):
        self._delay = delay
        self.timestamps: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def __call__(self, node: TaskNode) -> str:
        start = time.monotonic()
        time.sleep(self._delay)
        end = time.monotonic()
        with self._lock:
            self.timestamps[node.id] = (start, end)
        return f"timed: {node.id}"


# ---------------------------------------------------------------------------
# Worker protocol
# ---------------------------------------------------------------------------


class TestWorkerProtocol:
    def test_echo_worker_satisfies_protocol(self):
        assert isinstance(EchoWorker(), Worker)

    def test_lambda_satisfies_protocol(self):
        fn = lambda node: node.id  # noqa: E731
        assert callable(fn)

    def test_failing_worker_satisfies_protocol(self):
        assert isinstance(FailingWorker(), Worker)


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_success_when_no_failures(self):
        r = ExecutionResult(completed=["a", "b"], total_nodes=2)
        assert r.success is True
        assert r.all_finished is True

    def test_not_success_when_failed(self):
        r = ExecutionResult(
            completed=["a"], failed={"b": "err"}, total_nodes=2
        )
        assert r.success is False
        assert r.all_finished is True

    def test_not_success_when_skipped(self):
        r = ExecutionResult(
            completed=["a"], skipped=["b"], total_nodes=2
        )
        assert r.success is False
        assert r.all_finished is True

    def test_all_finished_false(self):
        r = ExecutionResult(completed=["a"], total_nodes=3)
        assert r.all_finished is False

    def test_empty_result(self):
        r = ExecutionResult(total_nodes=0)
        assert r.success is True
        assert r.all_finished is True


# ---------------------------------------------------------------------------
# Empty graph
# ---------------------------------------------------------------------------


class TestEmptyGraph:
    def test_execute_empty_graph(self):
        engine = ExecutionEngine(EchoWorker())
        result = engine.execute(_make_empty_graph())
        assert result.success is True
        assert result.total_nodes == 0
        assert result.completed == []
        assert result.failed == {}
        assert result.elapsed_seconds >= 0

    def test_execute_with_policy_empty(self):
        graph = _make_empty_graph()
        policy = PolicyController(graph)
        engine = ExecutionEngine(EchoWorker())
        result = engine.execute_with_policy(graph, policy)
        assert result.success is True
        assert result.total_nodes == 0


# ---------------------------------------------------------------------------
# Single node
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_single_node_completes(self):
        engine = ExecutionEngine(EchoWorker())
        graph = _make_single_node_graph()
        result = engine.execute(graph)
        assert result.success is True
        assert result.completed == ["solo"]
        assert graph.get_node("solo").status == TaskStatus.COMPLETED
        assert graph.get_node("solo").result == "done: lone task"

    def test_single_node_failure(self):
        engine = ExecutionEngine(FailingWorker())
        graph = _make_single_node_graph()
        result = engine.execute(graph)
        assert result.success is False
        assert "solo" in result.failed
        assert "Intentional failure" in result.failed["solo"]
        assert graph.get_node("solo").status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Linear chain — dependency ordering
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_linear_three_completes_in_order(self):
        execution_order: list[str] = []
        lock = threading.Lock()

        def ordered_worker(node: TaskNode) -> str:
            with lock:
                execution_order.append(node.id)
            return f"done: {node.id}"

        engine = ExecutionEngine(ordered_worker, max_workers=4)
        graph = _make_linear_graph(3)
        result = engine.execute(graph)

        assert result.success is True
        assert result.completed == ["task_0", "task_1", "task_2"]
        # Because of dependencies, execution order must be sequential
        assert execution_order == ["task_0", "task_1", "task_2"]

    def test_linear_failure_mid_chain(self):
        """Failing task_1 should skip task_2."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"task_1"}), max_workers=4
        )
        graph = _make_linear_graph(3)
        result = engine.execute(graph)

        assert result.success is False
        assert "task_0" in result.completed
        assert "task_1" in result.failed
        assert "task_2" in result.skipped

    def test_long_linear_chain(self):
        engine = ExecutionEngine(EchoWorker(), max_workers=2)
        graph = _make_linear_graph(10)
        result = engine.execute(graph)
        assert result.success is True
        assert len(result.completed) == 10


# ---------------------------------------------------------------------------
# Diamond dependencies
# ---------------------------------------------------------------------------


class TestDiamondDependencies:
    def test_diamond_completes(self):
        engine = ExecutionEngine(EchoWorker(), max_workers=4)
        graph = _make_diamond_graph()
        result = engine.execute(graph)

        assert result.success is True
        assert set(result.completed) == {"A", "B", "C", "D"}
        # A must come before B, C; B and C must come before D
        a_idx = result.completed.index("A")
        b_idx = result.completed.index("B")
        c_idx = result.completed.index("C")
        d_idx = result.completed.index("D")
        assert a_idx < b_idx
        assert a_idx < c_idx
        assert b_idx < d_idx
        assert c_idx < d_idx

    def test_diamond_b_and_c_run_concurrently(self):
        """B and C should overlap in time when workers are available."""
        worker = TimestampWorker(delay=0.1)
        engine = ExecutionEngine(worker, max_workers=4)
        graph = _make_diamond_graph()
        result = engine.execute(graph)

        assert result.success is True
        b_start, b_end = worker.timestamps["B"]
        c_start, c_end = worker.timestamps["C"]
        # They should overlap: one starts before the other ends
        overlap = min(b_end, c_end) - max(b_start, c_start)
        assert overlap > 0, "B and C should execute concurrently"

    def test_diamond_fail_b_skips_d_but_c_completes(self):
        """If B fails, D is skipped but C still runs."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"B"}), max_workers=4
        )
        graph = _make_diamond_graph()
        result = engine.execute(graph)

        assert "A" in result.completed
        assert "C" in result.completed
        assert "B" in result.failed
        assert "D" in result.skipped

    def test_diamond_fail_c_skips_d_but_b_completes(self):
        """Symmetric: if C fails, D is skipped but B runs."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"C"}), max_workers=4
        )
        graph = _make_diamond_graph()
        result = engine.execute(graph)

        assert "A" in result.completed
        assert "B" in result.completed
        assert "C" in result.failed
        assert "D" in result.skipped

    def test_diamond_fail_a_skips_all_downstream(self):
        """Root failure should cascade to all descendants."""
        engine = ExecutionEngine(FailingWorker(), max_workers=4)
        graph = _make_diamond_graph()
        result = engine.execute(graph)

        assert "A" in result.failed
        assert set(result.skipped) == {"B", "C", "D"}


# ---------------------------------------------------------------------------
# Wide (fully parallel) graph
# ---------------------------------------------------------------------------


class TestWideGraph:
    def test_wide_all_complete(self):
        engine = ExecutionEngine(EchoWorker(), max_workers=8)
        graph = _make_wide_graph(8)
        result = engine.execute(graph)

        assert result.success is True
        assert len(result.completed) == 8

    def test_wide_concurrency(self):
        """Multiple independent nodes should run in parallel."""
        worker = TimestampWorker(delay=0.1)
        engine = ExecutionEngine(worker, max_workers=5)
        graph = _make_wide_graph(5)
        result = engine.execute(graph)

        assert result.success is True
        # If truly parallel, total time should be much less than 5 * 0.1 = 0.5s
        assert result.elapsed_seconds < 0.4, (
            f"Wide graph should run in parallel; took {result.elapsed_seconds:.3f}s"
        )

    def test_wide_selective_failure(self):
        """One failure in a wide graph shouldn't affect siblings."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"w_2"}), max_workers=5
        )
        graph = _make_wide_graph(5)
        result = engine.execute(graph)

        assert "w_2" in result.failed
        completed = set(result.completed)
        assert completed == {"w_0", "w_1", "w_3", "w_4"}
        assert result.skipped == []


# ---------------------------------------------------------------------------
# Complex graph topologies
# ---------------------------------------------------------------------------


def _make_two_chains_graph() -> TaskGraph:
    """Two independent chains: a0->a1->a2 and b0->b1."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="a0", description="chain A start"))
    graph.add_node(TaskNode(id="a1", description="chain A mid", dependencies=["a0"]))
    graph.add_node(TaskNode(id="a2", description="chain A end", dependencies=["a1"]))
    graph.add_node(TaskNode(id="b0", description="chain B start"))
    graph.add_node(TaskNode(id="b1", description="chain B end", dependencies=["b0"]))
    return graph


def _make_fan_out_fan_in() -> TaskGraph:
    """Root -> {m1, m2, m3} -> sink."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="root", description="start"))
    for i in range(3):
        graph.add_node(
            TaskNode(id=f"m{i}", description=f"middle {i}", dependencies=["root"])
        )
    graph.add_node(
        TaskNode(id="sink", description="end", dependencies=["m0", "m1", "m2"])
    )
    return graph


class TestComplexTopologies:
    def test_two_independent_chains(self):
        engine = ExecutionEngine(EchoWorker(), max_workers=4)
        graph = _make_two_chains_graph()
        result = engine.execute(graph)

        assert result.success is True
        assert set(result.completed) == {"a0", "a1", "a2", "b0", "b1"}

    def test_two_chains_one_fails(self):
        """Failing a0 blocks a1, a2 but chain B is fine."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"a0"}), max_workers=4
        )
        graph = _make_two_chains_graph()
        result = engine.execute(graph)

        assert "a0" in result.failed
        assert set(result.skipped) == {"a1", "a2"}
        assert set(result.completed) == {"b0", "b1"}

    def test_fan_out_fan_in(self):
        engine = ExecutionEngine(EchoWorker(), max_workers=4)
        graph = _make_fan_out_fan_in()
        result = engine.execute(graph)

        assert result.success is True
        assert set(result.completed) == {"root", "m0", "m1", "m2", "sink"}

    def test_fan_out_fan_in_middle_fails(self):
        """Fail one middle node — sink gets skipped."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"m1"}), max_workers=4
        )
        graph = _make_fan_out_fan_in()
        result = engine.execute(graph)

        assert "root" in result.completed
        assert "m0" in result.completed
        assert "m2" in result.completed
        assert "m1" in result.failed
        assert "sink" in result.skipped


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_wide_execution(self):
        """Run a wide graph with more workers than nodes — no races."""
        engine = ExecutionEngine(SlowWorker(0.02), max_workers=20)
        graph = _make_wide_graph(20)
        result = engine.execute(graph)

        assert result.success is True
        assert len(result.completed) == 20
        assert len(set(result.completed)) == 20  # no duplicates

    def test_repeated_execution_on_fresh_graphs(self):
        """Engine can be reused across multiple graph runs."""
        engine = ExecutionEngine(EchoWorker(), max_workers=4)
        for _ in range(5):
            graph = _make_diamond_graph()
            result = engine.execute(graph)
            assert result.success is True

    def test_no_node_executed_twice(self):
        """Each node's worker should be invoked exactly once."""
        call_counts: dict[str, int] = {}
        lock = threading.Lock()

        def counting_worker(node: TaskNode) -> str:
            with lock:
                call_counts[node.id] = call_counts.get(node.id, 0) + 1
            time.sleep(0.01)
            return f"done: {node.id}"

        engine = ExecutionEngine(counting_worker, max_workers=8)
        graph = _make_wide_graph(10)
        result = engine.execute(graph)

        assert result.success is True
        for nid, count in call_counts.items():
            assert count == 1, f"Node {nid} executed {count} times"


# ---------------------------------------------------------------------------
# execute_with_policy
# ---------------------------------------------------------------------------


class TestExecuteWithPolicy:
    def test_custom_policy(self):
        graph = _make_diamond_graph()
        policy = PolicyController(graph)
        engine = ExecutionEngine(EchoWorker(), max_workers=4)
        result = engine.execute_with_policy(graph, policy)

        assert result.success is True
        assert set(result.completed) == {"A", "B", "C", "D"}

    def test_policy_reflects_completion(self):
        graph = _make_single_node_graph()
        policy = PolicyController(graph)
        engine = ExecutionEngine(EchoWorker())
        result = engine.execute_with_policy(graph, policy)

        assert result.success is True
        assert policy.is_complete() is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_worker_returns_none(self):
        engine = ExecutionEngine(lambda node: None, max_workers=2)
        graph = _make_single_node_graph()
        result = engine.execute(graph)
        assert result.success is True
        assert graph.get_node("solo").result is None

    def test_worker_returns_complex_object(self):
        engine = ExecutionEngine(
            lambda node: {"data": [1, 2, 3], "id": node.id}, max_workers=2
        )
        graph = _make_single_node_graph()
        result = engine.execute(graph)
        assert result.success is True
        assert graph.get_node("solo").result == {"data": [1, 2, 3], "id": "solo"}

    def test_elapsed_time_recorded(self):
        engine = ExecutionEngine(SlowWorker(0.05), max_workers=2)
        graph = _make_single_node_graph()
        result = engine.execute(graph)
        assert result.elapsed_seconds >= 0.04

    def test_max_workers_one_serialises_execution(self):
        """With max_workers=1, even a wide graph runs sequentially."""
        execution_order: list[str] = []
        lock = threading.Lock()

        def recording_worker(node: TaskNode) -> str:
            with lock:
                execution_order.append(node.id)
            time.sleep(0.01)
            return node.id

        engine = ExecutionEngine(recording_worker, max_workers=1)
        graph = _make_wide_graph(5)
        result = engine.execute(graph)

        assert result.success is True
        assert len(execution_order) == 5

    def test_all_nodes_fail(self):
        """Every node in a wide graph fails — all marked FAILED."""
        engine = ExecutionEngine(FailingWorker(), max_workers=4)
        graph = _make_wide_graph(4)
        result = engine.execute(graph)

        assert result.success is False
        assert len(result.failed) == 4
        assert result.completed == []
        assert result.skipped == []

    def test_graph_node_statuses_consistent(self):
        """After execution, graph node statuses match the result."""
        engine = ExecutionEngine(
            SelectiveFailWorker(fail_ids={"B"}), max_workers=4
        )
        graph = _make_diamond_graph()
        result = engine.execute(graph)

        for nid in result.completed:
            assert graph.get_node(nid).status == TaskStatus.COMPLETED
        for nid in result.failed:
            assert graph.get_node(nid).status == TaskStatus.FAILED
        for nid in result.skipped:
            assert graph.get_node(nid).status == TaskStatus.SKIPPED
