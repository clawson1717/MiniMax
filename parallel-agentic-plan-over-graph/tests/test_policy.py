"""Comprehensive tests for PAPoG Policy Controller."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.models import TaskGraph, TaskNode, TaskStatus
from src.policy import DepthPriorityStrategy, PolicyController, PriorityStrategy


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
        graph.add_node(TaskNode(id=f"task_{i}", description=f"step {i}", dependencies=deps))
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
# Ready-node identification
# ---------------------------------------------------------------------------


class TestReadyNodeIdentification:
    def test_initial_ready_nodes_diamond(self):
        pc = PolicyController(_make_diamond_graph())
        ready = pc.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "A"

    def test_initial_ready_nodes_wide(self):
        pc = PolicyController(_make_wide_graph(4))
        ready = pc.get_ready_nodes()
        assert len(ready) == 4

    def test_no_ready_on_empty_graph(self):
        pc = PolicyController(_make_empty_graph())
        assert pc.get_ready_nodes() == []

    def test_ready_after_completion_diamond(self):
        pc = PolicyController(_make_diamond_graph())
        pc.assign_next("agent-1")
        pc.complete_task("A", "done")
        ready = pc.get_ready_nodes()
        ids = {n.id for n in ready}
        assert ids == {"B", "C"}

    def test_ready_after_all_deps_completed(self):
        pc = PolicyController(_make_diamond_graph())
        # Complete A
        pc.assign_next("a1")
        pc.complete_task("A")
        # Complete B and C
        pc.assign_next("a2")
        pc.assign_next("a3")
        pc.complete_task("B")
        pc.complete_task("C")
        ready = pc.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "D"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_depth_priority_wide_graph(self):
        """All nodes at depth 0 — sorted by ID."""
        pc = PolicyController(_make_wide_graph(3))
        ready = pc.get_ready_nodes()
        ids = [n.id for n in ready]
        assert ids == ["w_0", "w_1", "w_2"]

    def test_depth_priority_diamond_after_root(self):
        pc = PolicyController(_make_diamond_graph())
        pc.assign_next("a")
        pc.complete_task("A")
        ready = pc.get_ready_nodes()
        ids = [n.id for n in ready]
        # B and C at depth 1 — sorted lexicographically
        assert ids == ["B", "C"]

    def test_custom_priority_strategy(self):
        """Plug in a custom strategy that reverses the default order."""

        class ReversePriority:
            def score(self, node: TaskNode, graph: TaskGraph) -> tuple[float, str]:
                # Reverse sort: negate the trailing digit to flip order
                # e.g. w_0 -> score 0 becomes tiebreak "~2", w_2 -> "~0"
                return (0.0, "".join(chr(255 - ord(c)) for c in node.id))

        pc = PolicyController(_make_wide_graph(3), strategy=ReversePriority())
        ready = pc.get_ready_nodes()
        ids = [n.id for n in ready]
        # Reversed lexicographic (all same depth)
        assert ids == ["w_2", "w_1", "w_0"]

    def test_custom_strategy_satisfies_protocol(self):
        class MyStrategy:
            def score(self, node: TaskNode, graph: TaskGraph) -> tuple[float, str]:
                return (0.0, node.id)

        assert isinstance(MyStrategy(), PriorityStrategy)


# ---------------------------------------------------------------------------
# Assignment and status transitions
# ---------------------------------------------------------------------------


class TestAssignment:
    def test_assign_next_returns_node(self):
        pc = PolicyController(_make_single_node_graph())
        node = pc.assign_next("agent-x")
        assert node is not None
        assert node.id == "solo"
        assert node.status == TaskStatus.RUNNING
        assert node.assigned_agent == "agent-x"

    def test_assign_next_returns_none_when_empty(self):
        pc = PolicyController(_make_empty_graph())
        assert pc.assign_next("agent-x") is None

    def test_assign_next_removes_from_queue(self):
        pc = PolicyController(_make_single_node_graph())
        pc.assign_next("a")
        assert pc.get_ready_nodes() == []
        assert pc.assign_next("b") is None

    def test_assign_next_picks_highest_priority(self):
        pc = PolicyController(_make_wide_graph(3))
        node = pc.assign_next("a")
        assert node is not None
        assert node.id == "w_0"

    def test_sequential_assign_respects_order(self):
        pc = PolicyController(_make_wide_graph(3))
        ids = []
        for i in range(3):
            node = pc.assign_next(f"a{i}")
            assert node is not None
            ids.append(node.id)
        assert ids == ["w_0", "w_1", "w_2"]


# ---------------------------------------------------------------------------
# Batch assignment
# ---------------------------------------------------------------------------


class TestBatchAssignment:
    def test_batch_assigns_all(self):
        pc = PolicyController(_make_wide_graph(3))
        assignments = pc.assign_batch(["a1", "a2", "a3"])
        assert len(assignments) == 3
        agent_ids = [a for a, _ in assignments]
        node_ids = [n.id for _, n in assignments]
        assert agent_ids == ["a1", "a2", "a3"]
        assert node_ids == ["w_0", "w_1", "w_2"]

    def test_batch_partial_when_fewer_ready(self):
        pc = PolicyController(_make_single_node_graph())
        assignments = pc.assign_batch(["a1", "a2"])
        assert len(assignments) == 1
        assert assignments[0][0] == "a1"
        assert assignments[0][1].id == "solo"

    def test_batch_empty_agents(self):
        pc = PolicyController(_make_wide_graph(3))
        assert pc.assign_batch([]) == []

    def test_batch_marks_running(self):
        pc = PolicyController(_make_wide_graph(2))
        assignments = pc.assign_batch(["a1", "a2"])
        for _, node in assignments:
            assert node.status == TaskStatus.RUNNING

    def test_batch_drains_queue(self):
        pc = PolicyController(_make_wide_graph(2))
        pc.assign_batch(["a1", "a2"])
        assert pc.get_ready_nodes() == []


# ---------------------------------------------------------------------------
# Complete / fail task
# ---------------------------------------------------------------------------


class TestCompleteTask:
    def test_complete_stores_result(self):
        pc = PolicyController(_make_single_node_graph())
        pc.assign_next("a")
        pc.complete_task("solo", result="output-data")
        node = pc.graph.get_node("solo")
        assert node.status == TaskStatus.COMPLETED
        assert node.result == "output-data"

    def test_complete_refreshes_queue(self):
        pc = PolicyController(_make_linear_graph(3))
        pc.assign_next("a")
        pc.complete_task("task_0")
        ready = pc.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "task_1"

    def test_complete_non_running_raises(self):
        pc = PolicyController(_make_single_node_graph())
        with pytest.raises(ValueError, match="expected running"):
            pc.complete_task("solo")

    def test_complete_unknown_node_raises(self):
        pc = PolicyController(_make_empty_graph())
        with pytest.raises(KeyError):
            pc.complete_task("nope")

    def test_complete_unlocks_diamond_sink(self):
        pc = PolicyController(_make_diamond_graph())
        # A
        pc.assign_next("a")
        pc.complete_task("A")
        # B & C
        pc.assign_batch(["b", "c"])
        pc.complete_task("B")
        pc.complete_task("C")
        ready = pc.get_ready_nodes()
        assert [n.id for n in ready] == ["D"]


class TestFailTask:
    def test_fail_stores_error(self):
        pc = PolicyController(_make_single_node_graph())
        pc.assign_next("a")
        pc.fail_task("solo", error="boom")
        node = pc.graph.get_node("solo")
        assert node.status == TaskStatus.FAILED
        assert node.result == "boom"

    def test_fail_refreshes_queue(self):
        pc = PolicyController(_make_linear_graph(2))
        pc.assign_next("a")
        pc.fail_task("task_0", "err")
        # task_1 depends on task_0 which failed, so nothing is ready
        assert pc.get_ready_nodes() == []

    def test_fail_non_running_raises(self):
        pc = PolicyController(_make_single_node_graph())
        with pytest.raises(ValueError, match="expected running"):
            pc.fail_task("solo", "err")


# ---------------------------------------------------------------------------
# is_complete / is_stuck
# ---------------------------------------------------------------------------


class TestGraphStateQueries:
    def test_empty_graph_is_complete(self):
        pc = PolicyController(_make_empty_graph())
        assert pc.is_complete() is True

    def test_single_node_not_complete_initially(self):
        pc = PolicyController(_make_single_node_graph())
        assert pc.is_complete() is False

    def test_single_node_complete_after_done(self):
        pc = PolicyController(_make_single_node_graph())
        pc.assign_next("a")
        pc.complete_task("solo")
        assert pc.is_complete() is True

    def test_diamond_complete(self):
        pc = PolicyController(_make_diamond_graph())
        for node_id in ["A", "B", "C", "D"]:
            pc.assign_next("a")
            pc.complete_task(node_id)
        assert pc.is_complete() is True

    def test_not_stuck_when_running(self):
        pc = PolicyController(_make_single_node_graph())
        pc.assign_next("a")
        assert pc.is_stuck() is False

    def test_not_stuck_when_ready(self):
        pc = PolicyController(_make_single_node_graph())
        assert pc.is_stuck() is False

    def test_stuck_after_root_failure(self):
        pc = PolicyController(_make_linear_graph(2))
        pc.assign_next("a")
        pc.fail_task("task_0", "err")
        assert pc.is_stuck() is True

    def test_stuck_diamond_partial_fail(self):
        pc = PolicyController(_make_diamond_graph())
        pc.assign_next("a")
        pc.complete_task("A")
        # Fail B, complete C
        b = pc.assign_next("a")
        c = pc.assign_next("a")
        assert b is not None and c is not None
        pc.fail_task(b.id, "err")
        pc.complete_task(c.id)
        # D needs both B and C completed — B failed, so stuck
        assert pc.is_stuck() is True

    def test_not_stuck_empty_graph(self):
        pc = PolicyController(_make_empty_graph())
        assert pc.is_stuck() is False


# ---------------------------------------------------------------------------
# get_status_summary
# ---------------------------------------------------------------------------


class TestStatusSummary:
    def test_summary_keys(self):
        pc = PolicyController(_make_single_node_graph())
        s = pc.get_status_summary()
        expected_keys = {
            "total", "pending", "running", "completed",
            "failed", "skipped", "ready_queue_size",
            "is_complete", "is_stuck",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_initial_state(self):
        pc = PolicyController(_make_diamond_graph())
        s = pc.get_status_summary()
        assert s["total"] == 4
        assert s["pending"] == 4
        assert s["running"] == 0
        assert s["completed"] == 0
        assert s["ready_queue_size"] == 1
        assert s["is_complete"] is False
        assert s["is_stuck"] is False

    def test_summary_after_assignment(self):
        pc = PolicyController(_make_diamond_graph())
        pc.assign_next("a")
        s = pc.get_status_summary()
        assert s["running"] == 1
        assert s["pending"] == 3
        assert s["ready_queue_size"] == 0

    def test_summary_empty_graph(self):
        pc = PolicyController(_make_empty_graph())
        s = pc.get_status_summary()
        assert s["total"] == 0
        assert s["is_complete"] is True
        assert s["is_stuck"] is False


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_assign_no_double_assignment(self):
        """Spawning many threads to call assign_next — each node assigned once."""
        graph = _make_wide_graph(20)
        pc = PolicyController(graph)
        results: list[TaskNode | None] = []
        lock = threading.Lock()

        def grab():
            node = pc.assign_next("agent")
            with lock:
                results.append(node)

        threads = [threading.Thread(target=grab) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assigned = [r for r in results if r is not None]
        assigned_ids = [n.id for n in assigned]
        # Exactly 20 nodes, no duplicates
        assert len(assigned) == 20
        assert len(set(assigned_ids)) == 20

    def test_concurrent_assign_batch(self):
        graph = _make_wide_graph(10)
        pc = PolicyController(graph)
        all_assignments: list[tuple[str, TaskNode]] = []
        lock = threading.Lock()

        def batch_grab(batch_id: int):
            assigns = pc.assign_batch([f"a{batch_id}_0", f"a{batch_id}_1"])
            with lock:
                all_assignments.extend(assigns)

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(batch_grab, range(10)))

        assigned_ids = [n.id for _, n in all_assignments]
        assert len(assigned_ids) == 10
        assert len(set(assigned_ids)) == 10

    def test_concurrent_complete_and_assign(self):
        """Complete tasks in parallel while others are being assigned."""
        graph = _make_wide_graph(10)
        pc = PolicyController(graph)

        # Assign all 10
        assignments = pc.assign_batch([f"a{i}" for i in range(10)])
        assert len(assignments) == 10

        errors: list[Exception] = []
        lock = threading.Lock()

        def complete_one(node_id: str):
            try:
                pc.complete_task(node_id)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=complete_one, args=(n.id,))
            for _, n in assignments
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert pc.is_complete() is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_node_lifecycle(self):
        pc = PolicyController(_make_single_node_graph())
        assert not pc.is_complete()
        assert not pc.is_stuck()

        node = pc.assign_next("a")
        assert node is not None
        assert node.id == "solo"

        pc.complete_task("solo", "result")
        assert pc.is_complete()

    def test_diamond_full_lifecycle(self):
        pc = PolicyController(_make_diamond_graph())

        # Phase 1: root
        node = pc.assign_next("a1")
        assert node is not None and node.id == "A"
        pc.complete_task("A")

        # Phase 2: B and C ready
        assignments = pc.assign_batch(["a2", "a3"])
        ids = {n.id for _, n in assignments}
        assert ids == {"B", "C"}
        for _, n in assignments:
            pc.complete_task(n.id)

        # Phase 3: D ready
        node = pc.assign_next("a4")
        assert node is not None and node.id == "D"
        pc.complete_task("D")

        assert pc.is_complete()

    def test_empty_graph_lifecycle(self):
        pc = PolicyController(_make_empty_graph())
        assert pc.is_complete()
        assert not pc.is_stuck()
        assert pc.assign_next("a") is None
        assert pc.assign_batch(["a", "b"]) == []

    def test_get_ready_returns_copy(self):
        pc = PolicyController(_make_wide_graph(2))
        r1 = pc.get_ready_nodes()
        r2 = pc.get_ready_nodes()
        assert r1 is not r2
        assert [n.id for n in r1] == [n.id for n in r2]

    def test_linear_chain_step_by_step(self):
        pc = PolicyController(_make_linear_graph(4))
        for i in range(4):
            ready = pc.get_ready_nodes()
            assert len(ready) == 1
            assert ready[0].id == f"task_{i}"
            pc.assign_next("a")
            pc.complete_task(f"task_{i}")
        assert pc.is_complete()

    def test_fail_then_stuck_then_summary(self):
        pc = PolicyController(_make_linear_graph(3))
        pc.assign_next("a")
        pc.fail_task("task_0", "err")
        assert pc.is_stuck()
        s = pc.get_status_summary()
        assert s["failed"] == 1
        assert s["is_stuck"] is True
        assert s["is_complete"] is False
