"""Tests for the Dynamic Re-Plan Logic (src/replan.py).

Covers:
- Successful re-plan on critical node failure
- Subgraph injection with correct dependency rewiring
- Re-plan retry limit enforcement
- Observer integration (callback triggered on critical failure)
- Edge cases: non-critical failure, already-completed node, empty graph
- Dependency rewiring correctness
- Namespace collision avoidance
- Re-plan history tracking
"""

from __future__ import annotations

import pytest

from src.architect import Architect, KeywordDecompositionStrategy, MockLLMStrategy
from src.models import TaskGraph, TaskNode, TaskStatus
from src.observer import EventType, GraphObserver, ObserverEvent
from src.policy import PolicyController
from src.replan import DynamicRePlanner, RePlanRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_graph() -> TaskGraph:
    """Create A -> B -> C (linear chain)."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="A", description="Task A", dependencies=[]))
    graph.add_node(TaskNode(id="B", description="Task B", dependencies=["A"]))
    graph.add_node(TaskNode(id="C", description="Task C", dependencies=["B"]))
    return graph


def _make_diamond_graph() -> TaskGraph:
    """Create A -> B, A -> C, B -> D, C -> D (diamond)."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="A", description="Task A", dependencies=[]))
    graph.add_node(TaskNode(id="B", description="Task B", dependencies=["A"]))
    graph.add_node(TaskNode(id="C", description="Task C", dependencies=["A"]))
    graph.add_node(TaskNode(id="D", description="Task D", dependencies=["B", "C"]))
    return graph


class SingleNodeStrategy:
    """Strategy that produces a single replacement node."""

    def decompose(self, goal: str) -> TaskGraph:
        graph = TaskGraph()
        graph.add_node(TaskNode(id="replacement", description=goal, dependencies=[]))
        return graph


class TwoNodeChainStrategy:
    """Strategy that produces two chained replacement nodes."""

    def decompose(self, goal: str) -> TaskGraph:
        graph = TaskGraph()
        graph.add_node(TaskNode(id="step1", description=f"Step 1: {goal}", dependencies=[]))
        graph.add_node(TaskNode(id="step2", description=f"Step 2: {goal}", dependencies=["step1"]))
        return graph


class FailingStrategy:
    """Strategy that always raises an exception."""

    def decompose(self, goal: str) -> TaskGraph:
        raise RuntimeError("Architect decomposition failed")


# ---------------------------------------------------------------------------
# Test: can_replan
# ---------------------------------------------------------------------------


class TestCanReplan:
    """Tests for DynamicRePlanner.can_replan()."""

    def test_can_replan_failed_node(self):
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=3)
        node = TaskNode(id="x", description="test", status=TaskStatus.FAILED)
        assert replanner.can_replan(node) is True

    def test_cannot_replan_pending_node(self):
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        node = TaskNode(id="x", description="test", status=TaskStatus.PENDING)
        assert replanner.can_replan(node) is False

    def test_cannot_replan_running_node(self):
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        node = TaskNode(id="x", description="test", status=TaskStatus.RUNNING)
        assert replanner.can_replan(node) is False

    def test_cannot_replan_completed_node(self):
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        node = TaskNode(id="x", description="test", status=TaskStatus.COMPLETED)
        assert replanner.can_replan(node) is False

    def test_cannot_replan_when_budget_exhausted(self):
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=1)
        # Simulate one re-plan already done
        graph = _make_linear_graph()
        graph.get_node("B").status = TaskStatus.FAILED
        replanner.replan_for_failure(graph, "B")
        # Now the budget is exhausted — even though the original "B" is failed
        node = TaskNode(id="B", description="test", status=TaskStatus.FAILED)
        assert replanner.can_replan(node) is False


# ---------------------------------------------------------------------------
# Test: replan_for_failure
# ---------------------------------------------------------------------------


class TestReplanForFailure:
    """Tests for DynamicRePlanner.replan_for_failure()."""

    def test_successful_replan_linear_graph(self):
        """Fail B in A->B->C; replacement nodes should appear and C should depend on them."""
        graph = _make_linear_graph()
        # Mark A completed, B failed
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "B")

        assert result is not None
        # The replacement node should be in the main graph (namespaced)
        replacement_ids = [nid for nid in graph.nodes if nid.startswith("replan_B_")]
        assert len(replacement_ids) == 1
        rep_id = replacement_ids[0]

        # C should now depend on the replacement, not on B
        c_node = graph.get_node("C")
        assert rep_id in c_node.dependencies
        assert "B" not in c_node.dependencies

        # Replacement root should depend on A (B's upstream)
        rep_node = graph.get_node(rep_id)
        assert "A" in rep_node.dependencies

    def test_replan_with_chain_strategy(self):
        """Fail B in A->B->C using a two-node chain replacement."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(TwoNodeChainStrategy()))
        result = replanner.replan_for_failure(graph, "B")

        assert result is not None
        # Two new namespaced nodes
        replan_ids = [nid for nid in graph.nodes if nid.startswith("replan_B_")]
        assert len(replan_ids) == 2

        # Find root and sink of the replacement
        step1_id = [nid for nid in replan_ids if "step1" in nid][0]
        step2_id = [nid for nid in replan_ids if "step2" in nid][0]

        # step1 (root) depends on A
        assert "A" in graph.get_node(step1_id).dependencies
        # step2 depends on step1
        assert step1_id in graph.get_node(step2_id).dependencies
        # C depends on step2 (the sink), not B
        c_node = graph.get_node("C")
        assert step2_id in c_node.dependencies
        assert "B" not in c_node.dependencies

    def test_replan_diamond_graph(self):
        """Fail B in diamond A->{B,C}->D; D should gain replacement + keep C dep."""
        graph = _make_diamond_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "B")

        assert result is not None
        d_node = graph.get_node("D")
        # D should still depend on C
        assert "C" in d_node.dependencies
        # D should also depend on the replacement sink
        replan_ids = [nid for nid in graph.nodes if nid.startswith("replan_B_")]
        assert any(rid in d_node.dependencies for rid in replan_ids)
        # D should NOT depend on B anymore
        assert "B" not in d_node.dependencies

    def test_replan_nonexistent_node(self):
        """Re-plan for a node that doesn't exist returns None."""
        graph = _make_linear_graph()
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "nonexistent")
        assert result is None

    def test_replan_non_failed_node(self):
        """Re-plan for a PENDING node returns None."""
        graph = _make_linear_graph()
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "A")
        assert result is None

    def test_replan_completed_node(self):
        """Re-plan for a COMPLETED node returns None."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "A")
        assert result is None

    def test_replan_architect_failure(self):
        """If the Architect raises, re-plan records failure and returns None."""
        graph = _make_linear_graph()
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(FailingStrategy()))
        result = replanner.replan_for_failure(graph, "B")

        assert result is None
        # Should still increment counter
        assert replanner.get_replan_count("B") == 1
        # History should record the failure
        assert len(replanner.history) == 1
        assert replanner.history[0].success is False
        assert "Architect decomposition failed" in replanner.history[0].error

    def test_replan_empty_graph(self):
        """Re-plan on an empty graph returns None."""
        graph = TaskGraph()
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "A")
        assert result is None


# ---------------------------------------------------------------------------
# Test: retry limit enforcement
# ---------------------------------------------------------------------------


class TestRetryLimits:
    """Tests for re-plan retry budget enforcement."""

    def test_retry_limit_enforced(self):
        """After max_retries re-plans, further attempts return None."""
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=2)

        for i in range(2):
            graph = _make_linear_graph()
            graph.get_node("A").status = TaskStatus.COMPLETED
            graph.get_node("B").status = TaskStatus.FAILED
            result = replanner.replan_for_failure(graph, "B")
            assert result is not None, f"Re-plan {i+1} should succeed"

        # Third attempt should be blocked
        graph2 = _make_linear_graph()
        graph2.get_node("A").status = TaskStatus.COMPLETED
        graph2.get_node("B").status = TaskStatus.FAILED
        result = replanner.replan_for_failure(graph2, "B")
        assert result is None

    def test_retry_count_tracking(self):
        """get_replan_count increments correctly."""
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=5)
        assert replanner.get_replan_count("B") == 0

        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED
        replanner.replan_for_failure(graph, "B")
        assert replanner.get_replan_count("B") == 1

    def test_failed_architect_counts_toward_limit(self):
        """A failed architect attempt still consumes a retry slot."""
        replanner = DynamicRePlanner(Architect(FailingStrategy()), max_retries=1)

        graph = _make_linear_graph()
        graph.get_node("B").status = TaskStatus.FAILED
        replanner.replan_for_failure(graph, "B")  # fails, consumes 1 retry

        # Now budget is exhausted
        assert replanner.can_replan(graph.get_node("B")) is False

    def test_different_nodes_independent_budgets(self):
        """Retry budgets are tracked per-node, not globally."""
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=1)

        # Exhaust B's budget
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED
        replanner.replan_for_failure(graph, "B")

        # A's budget should be independent (even though A isn't failed, the count is 0)
        assert replanner.get_replan_count("A") == 0
        assert replanner.get_replan_count("B") == 1


# ---------------------------------------------------------------------------
# Test: inject_subgraph
# ---------------------------------------------------------------------------


class TestInjectSubgraph:
    """Tests for DynamicRePlanner.inject_subgraph()."""

    def test_inject_preserves_completed_nodes(self):
        """Injection should not touch completed nodes."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        subgraph = TaskGraph()
        subgraph.add_node(TaskNode(id="new_B", description="replacement", dependencies=[]))

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        replanner.inject_subgraph(graph, subgraph, replacement_for="B")

        assert graph.get_node("A").status == TaskStatus.COMPLETED

    def test_inject_wires_roots_to_upstream(self):
        """Subgraph roots should inherit the failed node's upstream deps."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        subgraph = TaskGraph()
        subgraph.add_node(TaskNode(id="alt1", description="alt step 1", dependencies=[]))
        subgraph.add_node(TaskNode(id="alt2", description="alt step 2", dependencies=["alt1"]))

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        replanner.inject_subgraph(graph, subgraph, replacement_for="B")

        # alt1 (root) should depend on A
        assert "A" in graph.get_node("alt1").dependencies
        # alt2 (non-root) should still depend on alt1
        assert "alt1" in graph.get_node("alt2").dependencies

    def test_inject_rewires_downstream(self):
        """Downstream nodes should depend on subgraph sinks, not the failed node."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        subgraph = TaskGraph()
        subgraph.add_node(TaskNode(id="alt1", description="alt", dependencies=[]))

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        replanner.inject_subgraph(graph, subgraph, replacement_for="B")

        c_deps = graph.get_node("C").dependencies
        assert "alt1" in c_deps
        assert "B" not in c_deps

    def test_inject_maintains_dag(self):
        """After injection the graph should still be a valid DAG."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        subgraph = TaskGraph()
        subgraph.add_node(TaskNode(id="r1", description="r1", dependencies=[]))
        subgraph.add_node(TaskNode(id="r2", description="r2", dependencies=["r1"]))

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        replanner.inject_subgraph(graph, subgraph, replacement_for="B")

        # Should not raise
        graph.validate_dag()

    def test_inject_multiple_downstream(self):
        """When a failed node has multiple successors, all get rewired."""
        graph = TaskGraph()
        graph.add_node(TaskNode(id="A", description="A", dependencies=[]))
        graph.add_node(TaskNode(id="B", description="B", dependencies=["A"]))
        graph.add_node(TaskNode(id="C", description="C", dependencies=["B"]))
        graph.add_node(TaskNode(id="D", description="D", dependencies=["B"]))
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        subgraph = TaskGraph()
        subgraph.add_node(TaskNode(id="new_B", description="replacement", dependencies=[]))

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        replanner.inject_subgraph(graph, subgraph, replacement_for="B")

        assert "new_B" in graph.get_node("C").dependencies
        assert "B" not in graph.get_node("C").dependencies
        assert "new_B" in graph.get_node("D").dependencies
        assert "B" not in graph.get_node("D").dependencies


# ---------------------------------------------------------------------------
# Test: Observer integration
# ---------------------------------------------------------------------------


class TestObserverIntegration:
    """Tests for DynamicRePlanner integration with GraphObserver."""

    def test_callback_triggers_on_critical_failure(self):
        """When a critical node fails, the observer fires the re-plan callback."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        observer = GraphObserver(
            graph,
            critical_threshold=1,
            replan_callback=replanner.as_replan_callback(),
        )

        # Fail B (which is critical because C depends on it)
        # The node's actual status must be FAILED before the callback fires,
        # mirroring what ObservedPolicyController does (mutate then record).
        graph.get_node("B").status = TaskStatus.FAILED
        observer.record_transition("B", TaskStatus.RUNNING, TaskStatus.FAILED)

        # Re-plan should have been triggered
        assert replanner.get_replan_count("B") == 1
        # Replacement nodes should exist in the graph
        replan_ids = [nid for nid in graph.nodes if nid.startswith("replan_B_")]
        assert len(replan_ids) > 0

    def test_replan_triggered_event_emitted(self):
        """The observer should emit a REPLAN_TRIGGERED event."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        observer = GraphObserver(
            graph,
            critical_threshold=1,
            replan_callback=replanner.as_replan_callback(),
        )

        events_captured: list[ObserverEvent] = []
        observer.add_listener(
            lambda e: events_captured.append(e),
            event_type=EventType.REPLAN_TRIGGERED,
        )

        # Node must be in FAILED status before recording, mirroring real flow
        graph.get_node("B").status = TaskStatus.FAILED
        observer.record_transition("B", TaskStatus.RUNNING, TaskStatus.FAILED)

        replan_events = [
            e for e in events_captured if e.event_type == EventType.REPLAN_TRIGGERED
        ]
        assert len(replan_events) == 1
        assert replan_events[0].data["node_id"] == "B"
        # The new_graph should be a TaskGraph (from the callback)
        assert replan_events[0].data["new_graph"] is not None

    def test_no_replan_for_non_critical_failure(self):
        """Non-critical nodes should not trigger re-plan."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.COMPLETED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        observer = GraphObserver(
            graph,
            critical_threshold=1,
            replan_callback=replanner.as_replan_callback(),
        )

        # C is a leaf node — not critical (no descendants)
        graph.get_node("C").status = TaskStatus.RUNNING
        observer.record_transition("C", TaskStatus.RUNNING, TaskStatus.FAILED)

        # No re-plan should have been triggered
        assert replanner.get_replan_count("C") == 0

    def test_callback_with_observer_policy_controller(self):
        """Full integration: ObservedPolicyController fail triggers re-plan."""
        graph = _make_linear_graph()

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        policy = PolicyController(graph)
        observer = GraphObserver(
            graph,
            critical_threshold=1,
            replan_callback=replanner.as_replan_callback(),
        )
        observed_policy = observer.observe_policy(policy)

        # Assign and complete A
        node_a = observed_policy.assign_next("agent-1")
        assert node_a is not None
        observed_policy.complete_task("A", "done")

        # Assign B, then fail it
        node_b = observed_policy.assign_next("agent-1")
        assert node_b is not None
        observed_policy.fail_task("B", "something went wrong")

        # Re-plan should have fired
        assert replanner.get_replan_count("B") == 1
        replan_ids = [nid for nid in graph.nodes if nid.startswith("replan_B_")]
        assert len(replan_ids) > 0


# ---------------------------------------------------------------------------
# Test: History tracking
# ---------------------------------------------------------------------------


class TestHistory:
    """Tests for re-plan history records."""

    def test_successful_replan_recorded(self):
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        replanner.replan_for_failure(graph, "B")

        history = replanner.history
        assert len(history) == 1
        rec = history[0]
        assert isinstance(rec, RePlanRecord)
        assert rec.failed_node_id == "B"
        assert rec.success is True
        assert rec.error is None
        assert len(rec.replacement_node_ids) > 0

    def test_failed_replan_recorded(self):
        graph = _make_linear_graph()
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(FailingStrategy()))
        replanner.replan_for_failure(graph, "B")

        history = replanner.history
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].replacement_node_ids == []

    def test_multiple_replans_tracked(self):
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=3)

        for _ in range(3):
            graph = _make_linear_graph()
            graph.get_node("A").status = TaskStatus.COMPLETED
            graph.get_node("B").status = TaskStatus.FAILED
            replanner.replan_for_failure(graph, "B")

        assert len(replanner.history) == 3
        assert all(r.failed_node_id == "B" for r in replanner.history)


# ---------------------------------------------------------------------------
# Test: Namespace collision avoidance
# ---------------------------------------------------------------------------


class TestNamespacing:
    """Verify that replacement node IDs are namespaced to avoid collisions."""

    def test_replacement_ids_are_prefixed(self):
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "B")

        assert result is not None
        for nid in result.nodes:
            assert nid.startswith("replan_B_")

    def test_two_replans_no_collision(self):
        """Two re-plans of different nodes should not collide."""
        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=3)

        # Fail B
        graph = _make_diamond_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED
        replanner.replan_for_failure(graph, "B")

        # Now also fail C in the same graph
        graph.get_node("C").status = TaskStatus.FAILED
        replanner.replan_for_failure(graph, "C")

        b_ids = [nid for nid in graph.nodes if nid.startswith("replan_B_")]
        c_ids = [nid for nid in graph.nodes if nid.startswith("replan_C_")]
        assert len(b_ids) > 0
        assert len(c_ids) > 0
        assert set(b_ids).isdisjoint(set(c_ids))


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_replan_root_node_no_upstream(self):
        """Failing a root node (no dependencies) should still work."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "A")

        assert result is not None
        # Replacement root should have no dependencies (A had none)
        replan_ids = [nid for nid in graph.nodes if nid.startswith("replan_A_")]
        for rid in replan_ids:
            assert graph.get_node(rid).dependencies == []

        # B should now depend on the replacement, not A
        assert "A" not in graph.get_node("B").dependencies
        assert any(rid in graph.get_node("B").dependencies for rid in replan_ids)

    def test_replan_leaf_node_no_downstream(self):
        """Failing a leaf node (no successors) should inject without rewiring."""
        graph = _make_linear_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.COMPLETED
        graph.get_node("C").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()))
        result = replanner.replan_for_failure(graph, "C")

        assert result is not None
        # Replacement exists, depends on B
        replan_ids = [nid for nid in graph.nodes if nid.startswith("replan_C_")]
        assert len(replan_ids) > 0
        for rid in replan_ids:
            assert "B" in graph.get_node(rid).dependencies

    def test_graph_remains_valid_dag_after_replan(self):
        """The graph should be a valid DAG after any re-plan."""
        graph = _make_diamond_graph()
        graph.get_node("A").status = TaskStatus.COMPLETED
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(TwoNodeChainStrategy()))
        replanner.replan_for_failure(graph, "B")

        graph.validate_dag()  # Should not raise

    def test_max_retries_zero_disables_replan(self):
        """max_retries=0 means no re-planning ever."""
        graph = _make_linear_graph()
        graph.get_node("B").status = TaskStatus.FAILED

        replanner = DynamicRePlanner(Architect(SingleNodeStrategy()), max_retries=0)
        assert replanner.can_replan(graph.get_node("B")) is False
        result = replanner.replan_for_failure(graph, "B")
        assert result is None
