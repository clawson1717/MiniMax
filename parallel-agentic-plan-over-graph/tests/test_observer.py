"""Comprehensive tests for PAPoG State Observer (ARIES logic)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.models import TaskGraph, TaskNode, TaskStatus
from src.observer import (
    EventType,
    GraphObserver,
    ObservedPolicyController,
    ObserverEvent,
    StateTransition,
)
from src.policy import PolicyController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_diamond_graph() -> TaskGraph:
    """A -> B, A -> C, B -> D, C -> D."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="A", description="root"))
    graph.add_node(TaskNode(id="B", description="left", dependencies=["A"]))
    graph.add_node(TaskNode(id="C", description="right", dependencies=["A"]))
    graph.add_node(TaskNode(id="D", description="sink", dependencies=["B", "C"]))
    return graph


def _make_linear_graph(n: int = 4) -> TaskGraph:
    """task_0 -> task_1 -> ... -> task_{n-1}."""
    graph = TaskGraph()
    for i in range(n):
        deps = [f"task_{i - 1}"] if i > 0 else []
        graph.add_node(TaskNode(id=f"task_{i}", description=f"step {i}", dependencies=deps))
    return graph


def _make_wide_graph() -> TaskGraph:
    """root -> (c1, c2, c3, c4) — root has 4 dependents."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="root", description="root"))
    for i in range(4):
        graph.add_node(TaskNode(id=f"c{i}", description=f"child {i}", dependencies=["root"]))
    return graph


def _make_single_node_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(TaskNode(id="solo", description="lone task"))
    return graph


def _make_empty_graph() -> TaskGraph:
    return TaskGraph()


# ===========================================================================
# StateTransition dataclass
# ===========================================================================


class TestStateTransition:
    def test_immutable(self):
        t = StateTransition(
            node_id="A",
            from_status=TaskStatus.PENDING,
            to_status=TaskStatus.RUNNING,
            timestamp=1000.0,
        )
        with pytest.raises(AttributeError):
            t.node_id = "B"  # type: ignore[misc]

    def test_fields(self):
        t = StateTransition(
            node_id="X",
            from_status=TaskStatus.RUNNING,
            to_status=TaskStatus.COMPLETED,
            timestamp=42.0,
            metadata={"agent": "w1"},
        )
        assert t.node_id == "X"
        assert t.from_status == TaskStatus.RUNNING
        assert t.to_status == TaskStatus.COMPLETED
        assert t.timestamp == 42.0
        assert t.metadata == {"agent": "w1"}

    def test_default_metadata(self):
        t = StateTransition(
            node_id="A",
            from_status=TaskStatus.PENDING,
            to_status=TaskStatus.RUNNING,
            timestamp=0.0,
        )
        assert t.metadata == {}


# ===========================================================================
# ObserverEvent dataclass
# ===========================================================================


class TestObserverEvent:
    def test_fields(self):
        tr = StateTransition(
            node_id="A",
            from_status=TaskStatus.PENDING,
            to_status=TaskStatus.RUNNING,
            timestamp=0.0,
        )
        ev = ObserverEvent(event_type=EventType.STATE_CHANGE, transition=tr)
        assert ev.event_type == EventType.STATE_CHANGE
        assert ev.transition is tr
        assert ev.data == {}

    def test_defaults(self):
        ev = ObserverEvent(event_type=EventType.GRAPH_COMPLETE)
        assert ev.transition is None
        assert ev.data == {}


# ===========================================================================
# State transition recording and querying
# ===========================================================================


class TestTransitionRecording:
    def test_record_single_transition(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)

        tr = obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert isinstance(tr, StateTransition)
        assert tr.node_id == "solo"
        assert tr.from_status == TaskStatus.PENDING
        assert tr.to_status == TaskStatus.RUNNING

    def test_get_transitions_chronological(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)

        obs.record_transition("A", TaskStatus.PENDING, TaskStatus.RUNNING)
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.COMPLETED)
        obs.record_transition("B", TaskStatus.PENDING, TaskStatus.RUNNING)

        transitions = obs.get_transitions()
        assert len(transitions) == 3
        assert transitions[0].node_id == "A"
        assert transitions[0].to_status == TaskStatus.RUNNING
        assert transitions[1].node_id == "A"
        assert transitions[1].to_status == TaskStatus.COMPLETED
        assert transitions[2].node_id == "B"

    def test_get_node_history(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)

        obs.record_transition("A", TaskStatus.PENDING, TaskStatus.RUNNING)
        obs.record_transition("B", TaskStatus.PENDING, TaskStatus.RUNNING)
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.COMPLETED)

        a_history = obs.get_node_history("A")
        assert len(a_history) == 2
        assert a_history[0].to_status == TaskStatus.RUNNING
        assert a_history[1].to_status == TaskStatus.COMPLETED

        b_history = obs.get_node_history("B")
        assert len(b_history) == 1

    def test_get_node_history_unknown_node(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        assert obs.get_node_history("nonexistent") == []

    def test_transition_count(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        assert obs.get_transition_count() == 0

        obs.record_transition("A", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert obs.get_transition_count() == 1

        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.COMPLETED)
        assert obs.get_transition_count() == 2

    def test_transitions_are_copies(self):
        """get_transitions returns a copy, not the internal list."""
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)

        t1 = obs.get_transitions()
        t2 = obs.get_transitions()
        assert t1 is not t2
        assert t1 == t2


# ===========================================================================
# Timestamp accuracy and ordering
# ===========================================================================


class TestTimestamps:
    def test_timestamps_monotonically_increasing(self):
        graph = _make_linear_graph(3)
        obs = GraphObserver(graph)

        obs.record_transition("task_0", TaskStatus.PENDING, TaskStatus.RUNNING)
        obs.record_transition("task_0", TaskStatus.RUNNING, TaskStatus.COMPLETED)
        obs.record_transition("task_1", TaskStatus.PENDING, TaskStatus.RUNNING)

        transitions = obs.get_transitions()
        for i in range(1, len(transitions)):
            assert transitions[i].timestamp >= transitions[i - 1].timestamp

    def test_timestamp_is_realistic(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        before = time.time()
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        after = time.time()

        tr = obs.get_transitions()[0]
        assert before <= tr.timestamp <= after

    def test_metadata_preserved(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        obs.record_transition(
            "solo",
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            metadata={"agent_id": "worker-1", "attempt": 1},
        )
        tr = obs.get_transitions()[0]
        assert tr.metadata["agent_id"] == "worker-1"
        assert tr.metadata["attempt"] == 1


# ===========================================================================
# Graph snapshot
# ===========================================================================


class TestSnapshot:
    def test_snapshot_all_pending(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        snap = obs.get_snapshot()
        assert all(v == "pending" for v in snap.values())
        assert set(snap.keys()) == {"A", "B", "C", "D"}

    def test_snapshot_reflects_mutations(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        graph.get_node("A").mark_running("w1")
        snap = obs.get_snapshot()
        assert snap["A"] == "running"
        assert snap["B"] == "pending"

    def test_snapshot_empty_graph(self):
        graph = _make_empty_graph()
        obs = GraphObserver(graph)
        assert obs.get_snapshot() == {}


# ===========================================================================
# Critical path identification
# ===========================================================================


class TestCriticalPath:
    def test_diamond_critical_nodes(self):
        """In A→B,C→D: A has 3 descendants (B,C,D), B has 1 (D), C has 1 (D), D has 0."""
        graph = _make_diamond_graph()
        obs = GraphObserver(graph, critical_threshold=1)
        critical = obs.critical_node_ids
        # A, B, C all have ≥1 descendant
        assert "A" in critical
        assert "B" in critical
        assert "C" in critical
        # D has 0 descendants
        assert "D" not in critical

    def test_linear_critical_nodes(self):
        """task_0 → task_1 → task_2 → task_3: all but last are critical."""
        graph = _make_linear_graph(4)
        obs = GraphObserver(graph, critical_threshold=1)
        critical = obs.critical_node_ids
        assert "task_0" in critical
        assert "task_1" in critical
        assert "task_2" in critical
        assert "task_3" not in critical

    def test_critical_threshold_filtering(self):
        """With threshold=3, only A (3 descendants) qualifies in diamond."""
        graph = _make_diamond_graph()
        obs = GraphObserver(graph, critical_threshold=3)
        assert obs.critical_node_ids == {"A"}

    def test_descendant_count(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        assert obs.get_descendant_count("A") == 3
        assert obs.get_descendant_count("B") == 1
        assert obs.get_descendant_count("C") == 1
        assert obs.get_descendant_count("D") == 0

    def test_get_critical_path_sorted(self):
        """Critical path returns nodes sorted by descending descendant count."""
        graph = _make_diamond_graph()
        obs = GraphObserver(graph, critical_threshold=1)
        path = obs.get_critical_path()
        assert path[0] == "A"  # 3 descendants
        # B and C tied at 1 descendant each
        assert set(path[1:]) == {"B", "C"}

    def test_wide_graph_root_critical(self):
        graph = _make_wide_graph()
        obs = GraphObserver(graph, critical_threshold=1)
        assert "root" in obs.critical_node_ids
        assert obs.get_descendant_count("root") == 4

    def test_single_node_no_critical(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph, critical_threshold=1)
        assert obs.critical_node_ids == set()

    def test_empty_graph_no_critical(self):
        graph = _make_empty_graph()
        obs = GraphObserver(graph)
        assert obs.critical_node_ids == set()
        assert obs.get_critical_path() == []

    def test_descendant_count_unknown_node(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        assert obs.get_descendant_count("nonexistent") == 0


# ===========================================================================
# Re-plan trigger on critical node failure
# ===========================================================================


class TestReplanTrigger:
    def test_critical_failure_triggers_replan(self):
        graph = _make_diamond_graph()
        replan_mock = MagicMock(return_value=None)
        obs = GraphObserver(graph, replan_callback=replan_mock)

        # A is critical (3 descendants)
        graph.get_node("A").mark_running("w1")
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.FAILED)

        replan_mock.assert_called_once_with("A", graph)

    def test_non_critical_failure_no_replan(self):
        graph = _make_diamond_graph()
        replan_mock = MagicMock(return_value=None)
        obs = GraphObserver(graph, critical_threshold=1, replan_callback=replan_mock)

        # D is NOT critical (0 descendants)
        graph.get_node("D").status = TaskStatus.RUNNING
        obs.record_transition("D", TaskStatus.RUNNING, TaskStatus.FAILED)

        replan_mock.assert_not_called()

    def test_replan_emits_replan_triggered_event(self):
        graph = _make_diamond_graph()
        new_graph = _make_single_node_graph()
        replan_mock = MagicMock(return_value=new_graph)
        events_received: list[ObserverEvent] = []

        obs = GraphObserver(graph, replan_callback=replan_mock)
        obs.add_listener(
            lambda ev: events_received.append(ev),
            EventType.REPLAN_TRIGGERED,
        )

        graph.get_node("A").mark_running("w1")
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.FAILED)

        replan_events = [
            e for e in events_received if e.event_type == EventType.REPLAN_TRIGGERED
        ]
        assert len(replan_events) == 1
        assert replan_events[0].data["node_id"] == "A"
        assert replan_events[0].data["new_graph"] is new_graph

    def test_critical_failure_emits_critical_failure_event(self):
        graph = _make_diamond_graph()
        events_received: list[ObserverEvent] = []

        obs = GraphObserver(graph)
        obs.add_listener(
            lambda ev: events_received.append(ev),
            EventType.CRITICAL_FAILURE,
        )

        graph.get_node("A").mark_running("w1")
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.FAILED)

        assert len(events_received) == 1
        assert events_received[0].data["node_id"] == "A"
        assert events_received[0].data["descendant_count"] == 3

    def test_no_replan_callback_still_emits_critical_failure(self):
        """Even without a replan callback, critical failures emit events."""
        graph = _make_diamond_graph()
        events: list[ObserverEvent] = []

        obs = GraphObserver(graph, replan_callback=None)
        obs.add_listener(lambda ev: events.append(ev), EventType.CRITICAL_FAILURE)

        graph.get_node("A").mark_running("w1")
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.FAILED)

        assert len(events) == 1


# ===========================================================================
# Event callback / listener registration and firing
# ===========================================================================


class TestEventListeners:
    def test_add_listener_specific_type(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        received: list[ObserverEvent] = []

        obs.add_listener(lambda ev: received.append(ev), EventType.STATE_CHANGE)
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)

        assert len(received) == 1
        assert received[0].event_type == EventType.STATE_CHANGE

    def test_add_global_listener(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        received: list[ObserverEvent] = []

        # Global listener (event_type=None) gets all events
        obs.add_listener(lambda ev: received.append(ev))
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)

        assert len(received) >= 1
        types = {e.event_type for e in received}
        assert EventType.STATE_CHANGE in types

    def test_remove_listener(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        received: list[ObserverEvent] = []

        def listener(ev: ObserverEvent) -> None:
            received.append(ev)

        obs.add_listener(listener, EventType.STATE_CHANGE)
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert len(received) == 1

        obs.remove_listener(listener, EventType.STATE_CHANGE)
        obs.record_transition("solo", TaskStatus.RUNNING, TaskStatus.COMPLETED)
        # Should still be 1 — listener was removed
        assert len(received) == 1

    def test_remove_global_listener(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        received: list[ObserverEvent] = []

        def listener(ev: ObserverEvent) -> None:
            received.append(ev)

        obs.add_listener(listener)
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert len(received) >= 1

        count_after_first = len(received)
        obs.remove_listener(listener)
        obs.record_transition("solo", TaskStatus.RUNNING, TaskStatus.COMPLETED)
        assert len(received) == count_after_first

    def test_remove_nonexistent_listener_no_error(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        # Should not raise
        obs.remove_listener(lambda ev: None, EventType.STATE_CHANGE)
        obs.remove_listener(lambda ev: None)

    def test_multiple_listeners_same_type(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        r1: list[ObserverEvent] = []
        r2: list[ObserverEvent] = []

        obs.add_listener(lambda ev: r1.append(ev), EventType.STATE_CHANGE)
        obs.add_listener(lambda ev: r2.append(ev), EventType.STATE_CHANGE)

        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_listener_exception_does_not_crash(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)

        def bad_listener(ev: ObserverEvent) -> None:
            raise RuntimeError("boom")

        received: list[ObserverEvent] = []

        obs.add_listener(bad_listener, EventType.STATE_CHANGE)
        obs.add_listener(lambda ev: received.append(ev), EventType.STATE_CHANGE)

        # Should not raise — the bad listener's exception is swallowed
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        # Second listener still fires
        assert len(received) == 1

    def test_graph_complete_event(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)
        events: list[ObserverEvent] = []

        obs.add_listener(lambda ev: events.append(ev), EventType.GRAPH_COMPLETE)

        # Mutate the node and record transition to terminal
        graph.get_node("solo").mark_running("w1")
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert len(events) == 0  # Not terminal yet

        graph.get_node("solo").mark_completed("done")
        obs.record_transition("solo", TaskStatus.RUNNING, TaskStatus.COMPLETED)
        assert len(events) == 1
        assert events[0].event_type == EventType.GRAPH_COMPLETE

    def test_graph_complete_not_fired_on_partial(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        events: list[ObserverEvent] = []

        obs.add_listener(lambda ev: events.append(ev), EventType.GRAPH_COMPLETE)

        graph.get_node("A").mark_running("w1")
        graph.get_node("A").mark_completed("ok")
        obs.record_transition("A", TaskStatus.RUNNING, TaskStatus.COMPLETED)

        # B, C, D still pending — not complete
        assert len(events) == 0


# ===========================================================================
# Integration with PolicyController
# ===========================================================================


class TestObservedPolicyController:
    def test_observe_policy_creates_wrapper(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)
        assert isinstance(observed, ObservedPolicyController)
        assert observed.policy is policy

    def test_assign_next_records_transition(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        node = observed.assign_next("agent-1")
        assert node is not None
        assert node.id == "A"

        transitions = obs.get_transitions()
        assert len(transitions) == 1
        assert transitions[0].node_id == "A"
        assert transitions[0].from_status == TaskStatus.PENDING
        assert transitions[0].to_status == TaskStatus.RUNNING
        assert transitions[0].metadata["agent_id"] == "agent-1"

    def test_complete_task_records_transition(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        observed.assign_next("agent-1")  # A
        observed.complete_task("A", result="done")

        transitions = obs.get_transitions()
        assert len(transitions) == 2
        assert transitions[1].node_id == "A"
        assert transitions[1].from_status == TaskStatus.RUNNING
        assert transitions[1].to_status == TaskStatus.COMPLETED

    def test_fail_task_records_transition(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        observed.assign_next("agent-1")  # A
        observed.fail_task("A", error="something broke")

        transitions = obs.get_transitions()
        assert len(transitions) == 2
        assert transitions[1].to_status == TaskStatus.FAILED
        assert transitions[1].metadata["error"] == "something broke"

    def test_assign_batch_records_transitions(self):
        graph = _make_wide_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        # First assign root
        observed.assign_next("w0")
        observed.complete_task("root", result="ok")

        # Now children should be ready
        assignments = observed.assign_batch(["w1", "w2", "w3"])
        assert len(assignments) >= 1

        # Count transitions: 1 (assign root) + 1 (complete root) + len(assignments)
        assert obs.get_transition_count() == 2 + len(assignments)

    def test_observed_delegates_read_methods(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        assert observed.graph is graph
        assert len(observed.get_ready_nodes()) == 1
        assert not observed.is_complete()
        assert not observed.is_stuck()
        summary = observed.get_status_summary()
        assert summary["total"] == 4

    def test_observed_full_lifecycle(self):
        """Run a diamond graph through the observed policy, verifying all transitions."""
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        # A: assign → complete
        observed.assign_next("w1")
        observed.complete_task("A", "result_A")

        # B and C in parallel: assign → complete
        observed.assign_next("w1")  # B
        observed.assign_next("w2")  # C
        observed.complete_task("B", "result_B")
        observed.complete_task("C", "result_C")

        # D: assign → complete
        observed.assign_next("w1")
        observed.complete_task("D", "result_D")

        assert observed.is_complete()
        transitions = obs.get_transitions()
        # 4 assigns + 4 completes = 8 transitions
        assert len(transitions) == 8

        # Verify each node has exactly 2 transitions (PENDING→RUNNING, RUNNING→COMPLETED)
        for nid in ["A", "B", "C", "D"]:
            history = obs.get_node_history(nid)
            assert len(history) == 2
            assert history[0].to_status == TaskStatus.RUNNING
            assert history[1].to_status == TaskStatus.COMPLETED

    def test_critical_failure_through_observed_policy(self):
        """Failing a critical node through the observed policy triggers re-plan."""
        graph = _make_diamond_graph()
        replan_mock = MagicMock(return_value=None)
        obs = GraphObserver(graph, replan_callback=replan_mock)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        observed.assign_next("w1")  # A is critical
        observed.fail_task("A", error="crash")

        replan_mock.assert_called_once_with("A", graph)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_graph_observer(self):
        graph = _make_empty_graph()
        obs = GraphObserver(graph)
        assert obs.get_transitions() == []
        assert obs.get_snapshot() == {}
        assert obs.get_critical_path() == []
        assert obs.get_transition_count() == 0

    def test_single_node_full_lifecycle(self):
        graph = _make_single_node_graph()
        obs = GraphObserver(graph)

        graph.get_node("solo").mark_running("w1")
        obs.record_transition("solo", TaskStatus.PENDING, TaskStatus.RUNNING)
        graph.get_node("solo").mark_completed("done")
        obs.record_transition("solo", TaskStatus.RUNNING, TaskStatus.COMPLETED)

        assert obs.get_transition_count() == 2
        snap = obs.get_snapshot()
        assert snap["solo"] == "completed"

    def test_no_observers_registered(self):
        """Recording transitions with no listeners should work fine."""
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        # No listeners added
        obs.record_transition("A", TaskStatus.PENDING, TaskStatus.RUNNING)
        assert obs.get_transition_count() == 1

    def test_concurrent_recording(self):
        """Multiple threads recording transitions simultaneously."""
        graph = _make_wide_graph()
        obs = GraphObserver(graph)
        barrier = threading.Barrier(5)

        def record(node_id: str) -> None:
            barrier.wait()
            obs.record_transition(node_id, TaskStatus.PENDING, TaskStatus.RUNNING)

        threads = []
        for nid in ["root", "c0", "c1", "c2", "c3"]:
            t = threading.Thread(target=record, args=(nid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        assert obs.get_transition_count() == 5

    def test_graph_property(self):
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        assert obs.graph is graph

    def test_assign_next_returns_none_records_nothing(self):
        """When assign_next returns None (no ready nodes), no transition is recorded."""
        graph = _make_diamond_graph()
        obs = GraphObserver(graph)
        policy = PolicyController(graph)
        observed = obs.observe_policy(policy)

        # Assign A (only ready node)
        observed.assign_next("w1")
        assert obs.get_transition_count() == 1

        # Try again — no more ready nodes
        result = observed.assign_next("w2")
        assert result is None
        assert obs.get_transition_count() == 1  # unchanged
