"""Policy Controller (Orchestrator) for PAPoG.

Manages task scheduling via a priority queue over the TaskGraph,
handles agent assignment, status transitions, and provides
thread-safe state management for parallel execution.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from src.models import TaskGraph, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Priority strategy interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class PriorityStrategy(Protocol):
    """Protocol for pluggable priority scoring."""

    def score(self, node: TaskNode, graph: TaskGraph) -> tuple[float, str]:
        """Return a sort key ``(priority, tiebreaker)`` for *node*.

        Lower values indicate higher priority (will be scheduled first).
        """
        ...


class PriorityStrategyABC(ABC):
    """Abstract base class alternative for priority strategies."""

    @abstractmethod
    def score(self, node: TaskNode, graph: TaskGraph) -> tuple[float, str]:
        """Return a sort key ``(priority, tiebreaker)`` for *node*."""


# ---------------------------------------------------------------------------
# Built-in priority strategies
# ---------------------------------------------------------------------------


class DepthPriorityStrategy(PriorityStrategyABC):
    """Prioritise nodes by their topological depth (shallowest first).

    Depth is computed as the longest path from any root to the node.
    Ties are broken lexicographically by node ID.
    """

    def score(self, node: TaskNode, graph: TaskGraph) -> tuple[float, str]:
        """Return ``(depth, node_id)`` as the sort key."""
        depth = self._compute_depth(node.id, graph)
        return (float(depth), node.id)

    @staticmethod
    def _compute_depth(node_id: str, graph: TaskGraph) -> int:
        """Compute the longest path from any root to *node_id*."""
        # Use the internal networkx graph for efficient traversal
        nx_graph = graph._graph  # noqa: SLF001
        # longest_path_length from any ancestor
        depth = 0
        for pred in nx_graph.predecessors(node_id):
            pred_depth = DepthPriorityStrategy._compute_depth(pred, graph)
            depth = max(depth, pred_depth + 1)
        return depth


# ---------------------------------------------------------------------------
# Policy Controller
# ---------------------------------------------------------------------------


class PolicyController:
    """Orchestrator that manages task scheduling over a TaskGraph.

    Maintains a priority-ordered queue of ready tasks, assigns them to
    agents, tracks status transitions, and provides graph-state queries.
    All state mutations are protected by a ``threading.Lock``.

    Parameters
    ----------
    graph:
        The ``TaskGraph`` to orchestrate.
    strategy:
        A ``PriorityStrategy`` for scoring ready nodes.  Defaults to
        ``DepthPriorityStrategy``.
    """

    def __init__(
        self,
        graph: TaskGraph,
        strategy: PriorityStrategy | None = None,
    ) -> None:
        self._graph = graph
        self._strategy: PriorityStrategy = strategy or DepthPriorityStrategy()
        self._lock = threading.Lock()
        self._ready_queue: list[TaskNode] = []
        self._refresh_queue()

    # -- internal helpers ---------------------------------------------------

    def _refresh_queue(self) -> None:
        """Rebuild the priority queue from current graph state.

        Must be called with ``_lock`` held (or during ``__init__``).
        """
        ready = self._graph.get_ready_nodes()
        ready.sort(key=lambda n: self._strategy.score(n, self._graph))
        self._ready_queue = ready

    # -- public API ---------------------------------------------------------

    @property
    def graph(self) -> TaskGraph:
        """The underlying task graph."""
        return self._graph

    def get_ready_nodes(self) -> list[TaskNode]:
        """Return the current priority-ordered list of ready nodes.

        Returns a shallow copy so callers cannot mutate the internal queue.
        """
        with self._lock:
            return list(self._ready_queue)

    def assign_next(self, agent_id: str) -> Optional[TaskNode]:
        """Pop the highest-priority ready node and mark it RUNNING.

        Parameters
        ----------
        agent_id:
            Identifier of the agent that will execute the task.

        Returns
        -------
        Optional[TaskNode]
            The assigned node, or ``None`` if no tasks are ready.
        """
        with self._lock:
            if not self._ready_queue:
                return None
            node = self._ready_queue.pop(0)
            node.mark_running(agent_id)
            return node

    def assign_batch(
        self, agent_ids: list[str]
    ) -> list[tuple[str, TaskNode]]:
        """Assign ready nodes to multiple agents in priority order.

        Each agent gets at most one task.  Assignment stops when either
        the agent list or the ready queue is exhausted.

        Parameters
        ----------
        agent_ids:
            List of agent identifiers to assign tasks to.

        Returns
        -------
        list[tuple[str, TaskNode]]
            Pairs of ``(agent_id, assigned_node)``.
        """
        with self._lock:
            assignments: list[tuple[str, TaskNode]] = []
            for agent_id in agent_ids:
                if not self._ready_queue:
                    break
                node = self._ready_queue.pop(0)
                node.mark_running(agent_id)
                assignments.append((agent_id, node))
            return assignments

    def complete_task(self, node_id: str, result: Any = None) -> None:
        """Mark a task as COMPLETED and refresh the ready queue.

        Parameters
        ----------
        node_id:
            ID of the node to complete.
        result:
            Optional result data to store on the node.

        Raises
        ------
        KeyError
            If *node_id* does not exist in the graph.
        ValueError
            If the node is not in RUNNING status.
        """
        with self._lock:
            node = self._graph.get_node(node_id)
            if node.status != TaskStatus.RUNNING:
                raise ValueError(
                    f"Cannot complete node '{node_id}': "
                    f"status is {node.status.value}, expected running"
                )
            node.mark_completed(result)
            self._refresh_queue()

    def fail_task(self, node_id: str, error: str = "") -> None:
        """Mark a task as FAILED and refresh the ready queue.

        Parameters
        ----------
        node_id:
            ID of the node to fail.
        error:
            Error message to store on the node.

        Raises
        ------
        KeyError
            If *node_id* does not exist in the graph.
        ValueError
            If the node is not in RUNNING status.
        """
        with self._lock:
            node = self._graph.get_node(node_id)
            if node.status != TaskStatus.RUNNING:
                raise ValueError(
                    f"Cannot fail node '{node_id}': "
                    f"status is {node.status.value}, expected running"
                )
            node.mark_failed(error)
            self._refresh_queue()

    def is_complete(self) -> bool:
        """Return True when every node is COMPLETED.

        An empty graph is considered complete.
        """
        with self._lock:
            return all(
                n.status == TaskStatus.COMPLETED
                for n in self._graph.nodes.values()
            )

    def is_stuck(self) -> bool:
        """Return True when no nodes are ready but the graph is not complete.

        A graph is stuck when there are no PENDING nodes with all
        dependencies satisfied, no RUNNING nodes, yet not all nodes are
        COMPLETED.  This typically indicates a failure has blocked
        downstream tasks.
        """
        with self._lock:
            if all(
                n.status == TaskStatus.COMPLETED
                for n in self._graph.nodes.values()
            ):
                return False
            has_running = any(
                n.status == TaskStatus.RUNNING
                for n in self._graph.nodes.values()
            )
            has_ready = len(self._ready_queue) > 0
            return not has_running and not has_ready

    def get_status_summary(self) -> dict[str, Any]:
        """Return a summary of node statuses and queue depth.

        Returns
        -------
        dict
            Keys: ``total``, ``pending``, ``running``, ``completed``,
            ``failed``, ``skipped``, ``ready_queue_size``,
            ``is_complete``, ``is_stuck``.
        """
        with self._lock:
            counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
            for n in self._graph.nodes.values():
                counts[n.status.value] += 1

            all_completed = all(
                n.status == TaskStatus.COMPLETED
                for n in self._graph.nodes.values()
            )
            has_running = counts[TaskStatus.RUNNING.value] > 0
            has_ready = len(self._ready_queue) > 0
            stuck = not all_completed and not has_running and not has_ready

            return {
                "total": len(self._graph.nodes),
                "pending": counts[TaskStatus.PENDING.value],
                "running": counts[TaskStatus.RUNNING.value],
                "completed": counts[TaskStatus.COMPLETED.value],
                "failed": counts[TaskStatus.FAILED.value],
                "skipped": counts[TaskStatus.SKIPPED.value],
                "ready_queue_size": len(self._ready_queue),
                "is_complete": all_completed,
                "is_stuck": stuck,
            }
