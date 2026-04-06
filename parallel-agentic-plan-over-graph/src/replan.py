"""Dynamic Re-Plan Logic for PAPoG.

Provides the ``DynamicRePlanner`` — a component that integrates with
``GraphObserver`` to detect critical node failures and automatically
generate replacement sub-graphs using the ``Architect``.  New nodes are
injected into the live ``TaskGraph`` with dependencies rewired so
downstream work continues without disruption to running or completed
nodes.

Key capabilities
~~~~~~~~~~~~~~~~
* **Failure-triggered re-planning** — registers as the observer's
  ``replan_callback`` and fires when a critical node fails.
* **Sub-graph injection** — replacement nodes are inserted into the
  existing graph and dependency edges are rewired atomically.
* **Retry budgeting** — each node has a configurable maximum number of
  re-plan attempts; once exhausted the failure is treated as terminal.
* **History tracking** — every re-plan event is recorded for
  post-mortem analysis.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from src.architect import Architect, DecompositionStrategy
from src.models import TaskGraph, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Re-plan history record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RePlanRecord:
    """Immutable record of a single re-plan attempt."""

    failed_node_id: str
    replacement_node_ids: list[str]
    timestamp: float
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# DynamicRePlanner
# ---------------------------------------------------------------------------


class DynamicRePlanner:
    """Generates replacement sub-graphs for failed critical nodes.

    Parameters
    ----------
    architect:
        The ``Architect`` used to decompose alternative plans.
    max_retries:
        Maximum number of re-plan attempts per original node (not per
        replacement).  Defaults to 3.
    """

    def __init__(
        self,
        architect: Architect,
        max_retries: int = 3,
    ) -> None:
        self._architect = architect
        self._max_retries = max_retries
        self._lock = threading.Lock()

        # node_id -> number of times we've re-planned for it
        self._replan_counts: dict[str, int] = {}
        # Full history of re-plan attempts
        self._history: list[RePlanRecord] = []

    # -- properties ---------------------------------------------------------

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def history(self) -> list[RePlanRecord]:
        with self._lock:
            return list(self._history)

    def get_replan_count(self, node_id: str) -> int:
        """Return how many times *node_id* has been re-planned."""
        with self._lock:
            return self._replan_counts.get(node_id, 0)

    # -- core API -----------------------------------------------------------

    def can_replan(self, node: TaskNode) -> bool:
        """Return True if re-planning is still allowed for *node*.

        A node can be re-planned when:
        1. It is in FAILED status.
        2. The retry budget has not been exhausted.
        """
        if node.status != TaskStatus.FAILED:
            return False
        with self._lock:
            count = self._replan_counts.get(node.id, 0)
        return count < self._max_retries

    def replan_for_failure(
        self, graph: TaskGraph, failed_node_id: str
    ) -> TaskGraph | None:
        """Generate a replacement sub-graph for a failed node.

        Parameters
        ----------
        graph:
            The live ``TaskGraph`` containing the failed node.
        failed_node_id:
            ID of the node that failed.

        Returns
        -------
        TaskGraph | None
            A new sub-graph whose output nodes replace *failed_node_id*
            in downstream dependencies, or ``None`` if re-planning is
            not possible (budget exhausted, node not failed, etc.).

        Side-effects
        ~~~~~~~~~~~~~
        On success the replacement nodes are injected into *graph* and
        downstream dependencies are rewired.  The re-plan count for the
        failed node is incremented and a history record is appended.
        """
        # Validate the node exists and is eligible
        try:
            failed_node = graph.get_node(failed_node_id)
        except KeyError:
            return None

        if not self.can_replan(failed_node):
            return None

        # Attempt to generate an alternative sub-graph
        try:
            goal = f"Alternative approach for: {failed_node.description}"
            subgraph = self._architect.decompose(goal)
        except Exception as exc:
            # Architect itself failed — record and bail
            with self._lock:
                self._replan_counts[failed_node_id] = (
                    self._replan_counts.get(failed_node_id, 0) + 1
                )
                self._history.append(
                    RePlanRecord(
                        failed_node_id=failed_node_id,
                        replacement_node_ids=[],
                        timestamp=time.time(),
                        success=False,
                        error=str(exc),
                    )
                )
            return None

        # Namespace the replacement nodes to avoid ID collisions
        namespaced = self._namespace_subgraph(
            subgraph, prefix=f"replan_{failed_node_id}_"
        )

        # Inject into the live graph
        self.inject_subgraph(graph, namespaced, replacement_for=failed_node_id)

        # Record success
        new_ids = list(namespaced.nodes.keys())
        with self._lock:
            self._replan_counts[failed_node_id] = (
                self._replan_counts.get(failed_node_id, 0) + 1
            )
            self._history.append(
                RePlanRecord(
                    failed_node_id=failed_node_id,
                    replacement_node_ids=new_ids,
                    timestamp=time.time(),
                    success=True,
                )
            )

        return namespaced

    def inject_subgraph(
        self,
        graph: TaskGraph,
        subgraph: TaskGraph,
        replacement_for: str,
    ) -> None:
        """Inject *subgraph* into *graph*, replacing *replacement_for*.

        Steps:
        1. Wire the subgraph's root nodes to the same upstream
           dependencies as *replacement_for*.
        2. Add all subgraph nodes to *graph*.
        3. Rewire every downstream node that depended on
           *replacement_for* to instead depend on the subgraph's
           terminal (leaf/sink) nodes.

        The failed node itself remains in the graph (status FAILED) for
        history, but is no longer on any active execution path.
        """
        failed_node = graph.get_node(replacement_for)
        original_deps = list(failed_node.dependencies)

        # Identify subgraph roots (no internal dependencies) and sinks
        # (no internal successors)
        sub_nx = subgraph._graph  # noqa: SLF001
        root_ids = [n for n in sub_nx.nodes if sub_nx.in_degree(n) == 0]
        sink_ids = [n for n in sub_nx.nodes if sub_nx.out_degree(n) == 0]

        if not sink_ids:
            # Degenerate: every node is both root and sink (no edges)
            sink_ids = list(subgraph.nodes.keys())

        # 1. Wire subgraph roots to the failed node's upstream deps
        #    (only completed/pending ones — skip failed/skipped deps)
        upstream_deps = [
            dep for dep in original_deps
            if dep in graph.nodes
            and graph.get_node(dep).status
            not in (TaskStatus.FAILED, TaskStatus.SKIPPED)
        ]

        for node_id in subgraph.nodes:
            sub_node = subgraph.get_node(node_id)
            if node_id in root_ids:
                # Replace subgraph-internal deps with the failed node's
                # upstream deps
                sub_node.dependencies = list(upstream_deps)
            # Add to the main graph
            graph.add_node(sub_node)

        # Re-add internal subgraph edges into the main graph's nx graph
        for u, v in sub_nx.edges:
            graph._graph.add_edge(u, v)  # noqa: SLF001

        # 2. Rewire downstream: replace replacement_for with sink_ids
        main_nx = graph._graph  # noqa: SLF001
        downstream_ids = list(main_nx.successors(replacement_for))
        for ds_id in downstream_ids:
            ds_node = graph.get_node(ds_id)
            # Remove the failed node from dependencies
            if replacement_for in ds_node.dependencies:
                ds_node.dependencies.remove(replacement_for)
            # Remove the old edge
            if main_nx.has_edge(replacement_for, ds_id):
                main_nx.remove_edge(replacement_for, ds_id)
            # Add new edges from each sink to this downstream node
            for sink_id in sink_ids:
                if sink_id not in ds_node.dependencies:
                    ds_node.dependencies.append(sink_id)
                main_nx.add_edge(sink_id, ds_id)

    # -- observer callback interface ----------------------------------------

    def as_replan_callback(self):
        """Return a callable suitable for ``GraphObserver(replan_callback=...)``.

        The callback signature is ``(node_id, graph) -> TaskGraph | None``.
        """

        def _callback(node_id: str, graph: TaskGraph) -> TaskGraph | None:
            return self.replan_for_failure(graph, node_id)

        return _callback

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _namespace_subgraph(subgraph: TaskGraph, prefix: str) -> TaskGraph:
        """Return a copy of *subgraph* with all node IDs prefixed.

        Internal dependency references are updated to match.
        """
        new_graph = TaskGraph()
        id_map = {nid: f"{prefix}{nid}" for nid in subgraph.nodes}

        for old_id, node in subgraph.nodes.items():
            new_node = TaskNode(
                id=id_map[old_id],
                description=node.description,
                dependencies=[id_map[d] for d in node.dependencies if d in id_map],
                status=TaskStatus.PENDING,
                metadata={**node.metadata, "replanned_from": old_id},
            )
            new_graph.add_node(new_node)

        return new_graph
