"""State Observer (ARIES logic) for PAPoG.

Provides the ``GraphObserver`` — a reactive observation layer that
records every state transition in a ``TaskGraph``, computes critical-path
information, triggers re-plan callbacks when critical nodes fail, and
exposes a publish/subscribe event system so other components can react
to graph lifecycle events.

The observer integrates with ``PolicyController`` by wrapping its
mutation methods (``assign_next``, ``complete_task``, ``fail_task``) so
that every state change is automatically captured.

Design goals
~~~~~~~~~~~~
* **Immutable transition records** — each status change is stored as a
  ``StateTransition`` dataclass with wall-clock timestamps.
* **Critical-path identification** — nodes whose failure would block the
  most downstream work are surfaced for monitoring and re-plan triggers.
* **Callback/listener pattern** — arbitrary callables can subscribe to
  specific event types (or all events) and are notified synchronously.
* **Thread-safety** — all public methods acquire an internal lock so the
  observer is safe to use from the ``ExecutionEngine``'s thread pool.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import networkx as nx

from src.models import TaskGraph, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Event types and transition records
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Observable event kinds emitted by the observer."""

    STATE_CHANGE = "state_change"
    CRITICAL_FAILURE = "critical_failure"
    REPLAN_TRIGGERED = "replan_triggered"
    GRAPH_COMPLETE = "graph_complete"


@dataclass(frozen=True)
class StateTransition:
    """Immutable record of a single node status change."""

    node_id: str
    from_status: TaskStatus
    to_status: TaskStatus
    timestamp: float  # time.time()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObserverEvent:
    """Payload delivered to event listeners."""

    event_type: EventType
    transition: StateTransition | None = None
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for listener callables
EventListener = Callable[[ObserverEvent], None]


# ---------------------------------------------------------------------------
# GraphObserver
# ---------------------------------------------------------------------------


class GraphObserver:
    """Reactive observer that tracks state transitions in a TaskGraph.

    Parameters
    ----------
    graph:
        The ``TaskGraph`` to observe.
    critical_threshold:
        Minimum number of downstream dependents for a node to be
        considered *critical*.  Defaults to 1 (any node with at least
        one dependent is critical).
    replan_callback:
        Optional callable invoked when a critical node fails.  Signature:
        ``(node_id: str, graph: TaskGraph) -> TaskGraph | None``.  If it
        returns a new ``TaskGraph`` it is treated as a replacement
        sub-graph (the caller is responsible for integration).
    """

    def __init__(
        self,
        graph: TaskGraph,
        critical_threshold: int = 1,
        replan_callback: Callable[[str, TaskGraph], TaskGraph | None] | None = None,
    ) -> None:
        self._graph = graph
        self._critical_threshold = critical_threshold
        self._replan_callback = replan_callback
        self._lock = threading.Lock()

        # Transition history: global list + per-node index
        self._transitions: list[StateTransition] = []
        self._node_transitions: dict[str, list[StateTransition]] = {}

        # Event listeners: event_type -> [listener, ...]
        self._listeners: dict[EventType, list[EventListener]] = {
            et: [] for et in EventType
        }
        # Listeners that want *all* events
        self._global_listeners: list[EventListener] = []

        # Pre-compute critical nodes
        self._critical_node_ids: set[str] = set()
        self._descendant_counts: dict[str, int] = {}
        self._recompute_critical_nodes()

    # -- properties ---------------------------------------------------------

    @property
    def graph(self) -> TaskGraph:
        return self._graph

    @property
    def critical_node_ids(self) -> set[str]:
        """Node IDs currently classified as critical."""
        with self._lock:
            return set(self._critical_node_ids)

    # -- critical-path computation ------------------------------------------

    def _recompute_critical_nodes(self) -> None:
        """Identify nodes whose failure blocks >= *critical_threshold* descendants."""
        nx_graph = self._graph._graph  # noqa: SLF001
        counts: dict[str, int] = {}
        for node_id in self._graph.nodes:
            desc = nx.descendants(nx_graph, node_id)
            counts[node_id] = len(desc)
        self._descendant_counts = counts
        self._critical_node_ids = {
            nid for nid, cnt in counts.items() if cnt >= self._critical_threshold
        }

    def get_descendant_count(self, node_id: str) -> int:
        """Return the number of downstream dependents of *node_id*."""
        with self._lock:
            return self._descendant_counts.get(node_id, 0)

    def get_critical_path(self) -> list[str]:
        """Return critical node IDs sorted by descending impact (most dependents first)."""
        with self._lock:
            return sorted(
                self._critical_node_ids,
                key=lambda nid: self._descendant_counts.get(nid, 0),
                reverse=True,
            )

    # -- state transition recording -----------------------------------------

    def record_transition(
        self,
        node_id: str,
        from_status: TaskStatus,
        to_status: TaskStatus,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Record a state transition and notify listeners.

        Parameters
        ----------
        node_id:
            The node whose status changed.
        from_status:
            Previous status.
        to_status:
            New status.
        metadata:
            Optional extra data to attach to the transition record.

        Returns
        -------
        StateTransition
            The recorded transition.
        """
        transition = StateTransition(
            node_id=node_id,
            from_status=from_status,
            to_status=to_status,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            self._transitions.append(transition)
            self._node_transitions.setdefault(node_id, []).append(transition)

        # Emit state_change event
        event = ObserverEvent(
            event_type=EventType.STATE_CHANGE,
            transition=transition,
        )
        self._emit(event)

        # Check for critical failure
        if to_status == TaskStatus.FAILED and node_id in self._critical_node_ids:
            self._handle_critical_failure(node_id, transition)

        # Check for graph completion
        if self._is_graph_terminal():
            self._emit(ObserverEvent(
                event_type=EventType.GRAPH_COMPLETE,
                transition=transition,
                data={"snapshot": self.get_snapshot()},
            ))

        return transition

    def _handle_critical_failure(
        self, node_id: str, transition: StateTransition
    ) -> None:
        """Handle a critical node failure: emit event and optionally re-plan."""
        crit_event = ObserverEvent(
            event_type=EventType.CRITICAL_FAILURE,
            transition=transition,
            data={
                "node_id": node_id,
                "descendant_count": self._descendant_counts.get(node_id, 0),
            },
        )
        self._emit(crit_event)

        if self._replan_callback is not None:
            new_graph = self._replan_callback(node_id, self._graph)
            replan_event = ObserverEvent(
                event_type=EventType.REPLAN_TRIGGERED,
                transition=transition,
                data={
                    "node_id": node_id,
                    "new_graph": new_graph,
                },
            )
            self._emit(replan_event)

    def _is_graph_terminal(self) -> bool:
        """Return True when every node is in a terminal state."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(
            n.status in terminal for n in self._graph.nodes.values()
        ) and len(self._graph.nodes) > 0

    # -- query methods ------------------------------------------------------

    def get_transitions(self) -> list[StateTransition]:
        """Return all recorded transitions in chronological order."""
        with self._lock:
            return list(self._transitions)

    def get_node_history(self, node_id: str) -> list[StateTransition]:
        """Return transitions for a specific node."""
        with self._lock:
            return list(self._node_transitions.get(node_id, []))

    def get_snapshot(self) -> dict[str, str]:
        """Return a mapping of node_id -> current status value."""
        with self._lock:
            return {
                nid: node.status.value
                for nid, node in self._graph.nodes.items()
            }

    def get_transition_count(self) -> int:
        """Total number of recorded transitions."""
        with self._lock:
            return len(self._transitions)

    # -- event listener system ----------------------------------------------

    def add_listener(
        self,
        callback: EventListener,
        event_type: EventType | None = None,
    ) -> None:
        """Register a listener for a specific event type, or all events.

        Parameters
        ----------
        callback:
            Callable that receives an ``ObserverEvent``.
        event_type:
            If ``None``, the listener is called for every event.
            Otherwise, only for the specified ``EventType``.
        """
        with self._lock:
            if event_type is None:
                self._global_listeners.append(callback)
            else:
                self._listeners[event_type].append(callback)

    def remove_listener(
        self,
        callback: EventListener,
        event_type: EventType | None = None,
    ) -> None:
        """Remove a previously registered listener.

        Silently does nothing if the callback was not registered.
        """
        with self._lock:
            if event_type is None:
                try:
                    self._global_listeners.remove(callback)
                except ValueError:
                    pass
            else:
                try:
                    self._listeners[event_type].remove(callback)
                except ValueError:
                    pass

    def _emit(self, event: ObserverEvent) -> None:
        """Dispatch *event* to all matching listeners."""
        with self._lock:
            targets = list(self._listeners.get(event.event_type, []))
            targets.extend(self._global_listeners)

        for listener in targets:
            try:
                listener(event)
            except Exception:
                # Listeners must not crash the observer
                pass

    # -- PolicyController integration ---------------------------------------

    def observe_policy(self, policy: "PolicyController") -> "ObservedPolicyController":
        """Wrap a ``PolicyController`` so all mutations are auto-observed.

        Returns an ``ObservedPolicyController`` proxy that delegates to
        the original controller but records transitions through this
        observer.
        """
        return ObservedPolicyController(policy, self)


# ---------------------------------------------------------------------------
# ObservedPolicyController — transparent proxy
# ---------------------------------------------------------------------------


class ObservedPolicyController:
    """Wrapper around ``PolicyController`` that records transitions via an observer.

    All read methods delegate directly; mutation methods additionally
    record the state transition through the ``GraphObserver``.
    """

    def __init__(
        self,
        policy: "PolicyController",
        observer: GraphObserver,
    ) -> None:
        self._policy = policy
        self._observer = observer

    @property
    def policy(self) -> "PolicyController":
        return self._policy

    @property
    def graph(self) -> TaskGraph:
        return self._policy.graph

    def get_ready_nodes(self) -> list[TaskNode]:
        return self._policy.get_ready_nodes()

    def assign_next(self, agent_id: str) -> Optional[TaskNode]:
        node = self._policy.assign_next(agent_id)
        if node is not None:
            self._observer.record_transition(
                node_id=node.id,
                from_status=TaskStatus.PENDING,
                to_status=TaskStatus.RUNNING,
                metadata={"agent_id": agent_id},
            )
        return node

    def assign_batch(
        self, agent_ids: list[str]
    ) -> list[tuple[str, TaskNode]]:
        assignments = self._policy.assign_batch(agent_ids)
        for agent_id, node in assignments:
            self._observer.record_transition(
                node_id=node.id,
                from_status=TaskStatus.PENDING,
                to_status=TaskStatus.RUNNING,
                metadata={"agent_id": agent_id},
            )
        return assignments

    def complete_task(self, node_id: str, result: Any = None) -> None:
        self._policy.complete_task(node_id, result)
        self._observer.record_transition(
            node_id=node_id,
            from_status=TaskStatus.RUNNING,
            to_status=TaskStatus.COMPLETED,
            metadata={"result": result},
        )

    def fail_task(self, node_id: str, error: str = "") -> None:
        self._policy.fail_task(node_id, error)
        self._observer.record_transition(
            node_id=node_id,
            from_status=TaskStatus.RUNNING,
            to_status=TaskStatus.FAILED,
            metadata={"error": error},
        )

    def is_complete(self) -> bool:
        return self._policy.is_complete()

    def is_stuck(self) -> bool:
        return self._policy.is_stuck()

    def get_status_summary(self) -> dict[str, Any]:
        return self._policy.get_status_summary()
