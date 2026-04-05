"""Comprehensive tests for PAPoG task graph models."""

import pytest

from src.models import TaskGraph, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_enum_members(self):
        assert set(TaskStatus) == {
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        }

    def test_values_are_strings(self):
        for member in TaskStatus:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# TaskNode — creation
# ---------------------------------------------------------------------------


class TestTaskNodeCreation:
    def test_minimal_creation(self):
        node = TaskNode(id="a", description="do stuff")
        assert node.id == "a"
        assert node.description == "do stuff"
        assert node.dependencies == []
        assert node.status == TaskStatus.PENDING
        assert node.result is None
        assert node.assigned_agent is None
        assert node.metadata == {}

    def test_full_creation(self):
        node = TaskNode(
            id="b",
            description="process data",
            dependencies=["a"],
            status=TaskStatus.RUNNING,
            result="partial",
            assigned_agent="agent-1",
            metadata={"priority": "high"},
        )
        assert node.dependencies == ["a"]
        assert node.status == TaskStatus.RUNNING
        assert node.result == "partial"
        assert node.assigned_agent == "agent-1"
        assert node.metadata == {"priority": "high"}


# ---------------------------------------------------------------------------
# TaskNode — status transitions
# ---------------------------------------------------------------------------


class TestTaskNodeTransitions:
    def test_mark_running(self):
        node = TaskNode(id="t", description="test")
        node.mark_running("agent-42")
        assert node.status == TaskStatus.RUNNING
        assert node.assigned_agent == "agent-42"

    def test_mark_completed(self):
        node = TaskNode(id="t", description="test")
        node.mark_running("a1")
        node.mark_completed({"answer": 42})
        assert node.status == TaskStatus.COMPLETED
        assert node.result == {"answer": 42}

    def test_mark_failed(self):
        node = TaskNode(id="t", description="test")
        node.mark_running("a1")
        node.mark_failed("timeout after 30s")
        assert node.status == TaskStatus.FAILED
        assert node.result == "timeout after 30s"


# ---------------------------------------------------------------------------
# TaskNode — is_ready
# ---------------------------------------------------------------------------


class TestTaskNodeIsReady:
    def test_no_deps_always_ready(self):
        node = TaskNode(id="x", description="root")
        assert node.is_ready(set()) is True

    def test_deps_not_met(self):
        node = TaskNode(id="x", description="child", dependencies=["a", "b"])
        assert node.is_ready({"a"}) is False

    def test_deps_met(self):
        node = TaskNode(id="x", description="child", dependencies=["a", "b"])
        assert node.is_ready({"a", "b"}) is True

    def test_superset_completed_is_fine(self):
        node = TaskNode(id="x", description="child", dependencies=["a"])
        assert node.is_ready({"a", "b", "c"}) is True


# ---------------------------------------------------------------------------
# TaskGraph — construction
# ---------------------------------------------------------------------------


class TestTaskGraphConstruction:
    def test_empty_graph(self):
        g = TaskGraph()
        assert len(g.nodes) == 0

    def test_add_node(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="first"))
        assert "a" in g.nodes
        assert g.get_node("a").id == "a"

    def test_add_duplicate_raises(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="first"))
        with pytest.raises(ValueError, match="Duplicate"):
            g.add_node(TaskNode(id="a", description="second"))

    def test_get_node_missing_raises(self):
        g = TaskGraph()
        with pytest.raises(KeyError, match="no-such"):
            g.get_node("no-such")

    def test_edges_created_from_dependencies(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="first"))
        g.add_node(TaskNode(id="b", description="second", dependencies=["a"]))
        assert g._graph.has_edge("a", "b")


# ---------------------------------------------------------------------------
# TaskGraph — DAG validation & cycle detection
# ---------------------------------------------------------------------------


class TestTaskGraphValidation:
    def test_valid_dag(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="root"))
        g.add_node(TaskNode(id="b", description="mid", dependencies=["a"]))
        g.add_node(TaskNode(id="c", description="leaf", dependencies=["b"]))
        g.validate_dag()  # should not raise

    def test_cycle_detection(self):
        """Manually inject a cycle via the internal graph to test detection."""
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="A", dependencies=[]))
        g.add_node(TaskNode(id="b", description="B", dependencies=["a"]))
        # Force a back-edge to create a cycle
        g._graph.add_edge("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            g.validate_dag()


# ---------------------------------------------------------------------------
# TaskGraph — ready nodes
# ---------------------------------------------------------------------------


class TestTaskGraphReadyNodes:
    def _build_diamond(self) -> TaskGraph:
        """A → B, A → C, B+C → D."""
        g = TaskGraph()
        g.add_node(TaskNode(id="A", description="root"))
        g.add_node(TaskNode(id="B", description="left", dependencies=["A"]))
        g.add_node(TaskNode(id="C", description="right", dependencies=["A"]))
        g.add_node(TaskNode(id="D", description="join", dependencies=["B", "C"]))
        return g

    def test_initial_ready(self):
        g = self._build_diamond()
        ready = g.get_ready_nodes()
        assert [n.id for n in ready] == ["A"]

    def test_after_root_completed(self):
        g = self._build_diamond()
        g.get_node("A").mark_running("agent")
        g.get_node("A").mark_completed("done")
        ready_ids = {n.id for n in g.get_ready_nodes()}
        assert ready_ids == {"B", "C"}

    def test_join_not_ready_until_both(self):
        g = self._build_diamond()
        g.get_node("A").mark_completed("ok")
        g.get_node("B").mark_completed("ok")
        # C still pending — so D is NOT ready (via auto-detection)
        ready_ids = {n.id for n in g.get_ready_nodes()}
        assert "D" not in ready_ids
        assert "C" in ready_ids

    def test_join_ready_when_both_complete(self):
        g = self._build_diamond()
        for nid in ("A", "B", "C"):
            g.get_node(nid).mark_completed("ok")
        ready_ids = {n.id for n in g.get_ready_nodes()}
        assert ready_ids == {"D"}

    def test_explicit_completed_ids(self):
        g = self._build_diamond()
        # Pass explicit set instead of relying on node status
        ready = g.get_ready_nodes(completed_ids={"A", "B", "C"})
        # A, B, C are still PENDING status-wise, but A has no deps so it'd be ready
        # Actually get_ready_nodes filters on PENDING status AND deps met
        ready_ids = {n.id for n in ready}
        assert "D" in ready_ids

    def test_running_nodes_excluded(self):
        g = self._build_diamond()
        g.get_node("A").mark_running("agent")
        ready = g.get_ready_nodes(completed_ids=set())
        # A is RUNNING not PENDING, so not in ready list
        assert len(ready) == 0


# ---------------------------------------------------------------------------
# TaskGraph — topological order
# ---------------------------------------------------------------------------


class TestTaskGraphTopologicalOrder:
    def test_linear_chain(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="1", description="first"))
        g.add_node(TaskNode(id="2", description="second", dependencies=["1"]))
        g.add_node(TaskNode(id="3", description="third", dependencies=["2"]))
        order = g.topological_order()
        assert order == ["1", "2", "3"]

    def test_diamond_order_respects_deps(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="A", description="root"))
        g.add_node(TaskNode(id="B", description="left", dependencies=["A"]))
        g.add_node(TaskNode(id="C", description="right", dependencies=["A"]))
        g.add_node(TaskNode(id="D", description="join", dependencies=["B", "C"]))
        order = g.topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_topological_order_raises_on_cycle(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="A"))
        g.add_node(TaskNode(id="b", description="B", dependencies=["a"]))
        g._graph.add_edge("b", "a")  # force cycle
        with pytest.raises(ValueError, match="cycle"):
            g.topological_order()


# ---------------------------------------------------------------------------
# TaskGraph — serialisation round-trip
# ---------------------------------------------------------------------------


class TestTaskGraphSerialisation:
    def test_round_trip(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="a", description="root", metadata={"k": "v"}))
        g.add_node(TaskNode(id="b", description="child", dependencies=["a"]))
        g.get_node("a").mark_completed("result-a")

        data = g.to_dict()
        g2 = TaskGraph.from_dict(data)

        assert set(g2.nodes.keys()) == {"a", "b"}
        assert g2.get_node("a").status == TaskStatus.COMPLETED
        assert g2.get_node("a").result == "result-a"
        assert g2.get_node("a").metadata == {"k": "v"}
        assert g2.get_node("b").dependencies == ["a"]
        # Internal graph should also be rebuilt
        assert g2._graph.has_edge("a", "b")

    def test_round_trip_preserves_all_statuses(self):
        g = TaskGraph()
        g.add_node(TaskNode(id="p", description="pending"))
        g.add_node(TaskNode(id="r", description="running"))
        g.add_node(TaskNode(id="c", description="completed"))
        g.add_node(TaskNode(id="f", description="failed"))
        g.add_node(TaskNode(id="s", description="skipped", status=TaskStatus.SKIPPED))
        g.get_node("r").mark_running("agent-1")
        g.get_node("c").mark_completed(42)
        g.get_node("f").mark_failed("boom")

        g2 = TaskGraph.from_dict(g.to_dict())
        assert g2.get_node("p").status == TaskStatus.PENDING
        assert g2.get_node("r").status == TaskStatus.RUNNING
        assert g2.get_node("r").assigned_agent == "agent-1"
        assert g2.get_node("c").status == TaskStatus.COMPLETED
        assert g2.get_node("c").result == 42
        assert g2.get_node("f").status == TaskStatus.FAILED
        assert g2.get_node("f").result == "boom"
        assert g2.get_node("s").status == TaskStatus.SKIPPED

    def test_empty_graph_round_trip(self):
        g = TaskGraph()
        g2 = TaskGraph.from_dict(g.to_dict())
        assert len(g2.nodes) == 0
