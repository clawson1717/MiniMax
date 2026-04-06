"""Parallel Execution Engine for PAPoG.

Runs a TaskGraph to completion by dispatching ready nodes to a pool
of concurrent workers via ``concurrent.futures.ThreadPoolExecutor``.
All graph state mutations are routed through the PolicyController,
which is already thread-safe.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from src.models import TaskGraph, TaskNode, TaskStatus
from src.policy import PolicyController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Worker(Protocol):
    """Minimal interface that any worker callable must satisfy.

    A worker receives a ``TaskNode`` and returns a result object.
    It may raise an exception to signal failure.
    """

    def __call__(self, node: TaskNode) -> Any: ...


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Summary of a full graph execution run."""

    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    total_nodes: int = 0
    elapsed_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """True when every node completed successfully."""
        return len(self.failed) == 0 and len(self.skipped) == 0

    @property
    def all_finished(self) -> bool:
        """True when no nodes remain in PENDING or RUNNING state."""
        return (
            len(self.completed) + len(self.failed) + len(self.skipped)
            == self.total_nodes
        )


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Runs a TaskGraph in parallel using a thread pool.

    Parameters
    ----------
    worker:
        A callable matching the ``Worker`` protocol.  Called once per
        ready node with the ``TaskNode`` as the sole argument.
    max_workers:
        Maximum number of threads for the internal ``ThreadPoolExecutor``.
        Defaults to 4.
    poll_interval:
        Seconds to sleep between scheduling sweeps when waiting for
        in-flight futures to resolve.  Defaults to 0.05 s.
    """

    def __init__(
        self,
        worker: Worker,
        max_workers: int = 4,
        poll_interval: float = 0.05,
    ) -> None:
        self._worker = worker
        self._max_workers = max_workers
        self._poll_interval = poll_interval

    # -- public API ---------------------------------------------------------

    def execute(self, graph: TaskGraph) -> ExecutionResult:
        """Execute *graph* to completion, returning an ``ExecutionResult``.

        Creates a fresh ``PolicyController`` internally so the engine
        fully owns scheduling.  Nodes are dispatched in parallel up to
        ``max_workers`` concurrently; dependencies are respected via the
        policy controller's ready-node detection.

        Worker exceptions are caught per-node: the failing node is marked
        FAILED and independent downstream paths continue executing.
        Nodes whose dependencies include a FAILED node are marked SKIPPED.
        """
        t0 = time.monotonic()

        policy = PolicyController(graph)
        result = ExecutionResult(total_nodes=len(graph.nodes))

        if not graph.nodes:
            result.elapsed_seconds = time.monotonic() - t0
            return result

        # Track in-flight futures  {future: node_id}
        in_flight: dict[Future, str] = {}
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while True:
                # --- schedule ready nodes ----------------------------------
                ready = policy.get_ready_nodes()
                for node in ready:
                    assigned = policy.assign_next(f"engine-worker")
                    if assigned is None:
                        break  # another thread grabbed it
                    fut = pool.submit(self._run_node, assigned)
                    with lock:
                        in_flight[fut] = assigned.id

                # --- if nothing in-flight and nothing ready, we're done ----
                with lock:
                    active = len(in_flight)
                if active == 0 and not policy.get_ready_nodes():
                    break

                # --- collect at least one completed future ------------------
                # Copy to avoid mutation during iteration
                with lock:
                    snapshot = dict(in_flight)

                done_futures = [f for f in snapshot if f.done()]
                if not done_futures:
                    time.sleep(self._poll_interval)
                    continue

                for fut in done_futures:
                    with lock:
                        node_id = in_flight.pop(fut)

                    exc = fut.exception()
                    if exc is not None:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            "Node '%s' failed: %s", node_id, error_msg
                        )
                        policy.fail_task(node_id, error_msg)
                        result.failed[node_id] = error_msg
                        # Mark downstream dependents as skipped
                        self._skip_descendants(node_id, graph, policy, result)
                    else:
                        node_result = fut.result()
                        policy.complete_task(node_id, node_result)
                        result.completed.append(node_id)
                        logger.debug("Node '%s' completed", node_id)

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def execute_with_policy(
        self, graph: TaskGraph, policy: PolicyController
    ) -> ExecutionResult:
        """Like ``execute`` but accepts an externally-created policy.

        Useful when the caller wants to supply a custom priority strategy
        or reuse an existing controller.
        """
        t0 = time.monotonic()
        result = ExecutionResult(total_nodes=len(graph.nodes))

        if not graph.nodes:
            result.elapsed_seconds = time.monotonic() - t0
            return result

        in_flight: dict[Future, str] = {}
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while True:
                ready = policy.get_ready_nodes()
                for node in ready:
                    assigned = policy.assign_next("engine-worker")
                    if assigned is None:
                        break
                    fut = pool.submit(self._run_node, assigned)
                    with lock:
                        in_flight[fut] = assigned.id

                with lock:
                    active = len(in_flight)
                if active == 0 and not policy.get_ready_nodes():
                    break

                with lock:
                    snapshot = dict(in_flight)

                done_futures = [f for f in snapshot if f.done()]
                if not done_futures:
                    time.sleep(self._poll_interval)
                    continue

                for fut in done_futures:
                    with lock:
                        node_id = in_flight.pop(fut)

                    exc = fut.exception()
                    if exc is not None:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            "Node '%s' failed: %s", node_id, error_msg
                        )
                        policy.fail_task(node_id, error_msg)
                        result.failed[node_id] = error_msg
                        self._skip_descendants(node_id, graph, policy, result)
                    else:
                        node_result = fut.result()
                        policy.complete_task(node_id, node_result)
                        result.completed.append(node_id)

        result.elapsed_seconds = time.monotonic() - t0
        return result

    # -- internal helpers ---------------------------------------------------

    def _run_node(self, node: TaskNode) -> Any:
        """Invoke the worker on *node* and return its result."""
        return self._worker(node)

    @staticmethod
    def _skip_descendants(
        failed_id: str,
        graph: TaskGraph,
        policy: PolicyController,
        result: ExecutionResult,
    ) -> None:
        """Recursively mark all downstream dependents of a failed node as SKIPPED.

        Only skips nodes that are still PENDING (not already RUNNING or
        finished).  Uses the internal networkx graph for successor traversal.
        """
        nx_graph = graph._graph  # noqa: SLF001
        visited: set[str] = set()
        stack = list(nx_graph.successors(failed_id))

        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)

            node = graph.get_node(nid)
            if node.status == TaskStatus.PENDING:
                # Check if ALL paths to this node are blocked
                # A node should only be skipped if it can never become ready
                # i.e., at least one dependency is FAILED or SKIPPED
                deps_statuses = [
                    graph.get_node(dep).status for dep in node.dependencies
                ]
                has_blocked_dep = any(
                    s in (TaskStatus.FAILED, TaskStatus.SKIPPED)
                    for s in deps_statuses
                )
                if has_blocked_dep:
                    node.status = TaskStatus.SKIPPED
                    result.skipped.append(nid)
                    # Continue to this node's successors
                    stack.extend(nx_graph.successors(nid))
