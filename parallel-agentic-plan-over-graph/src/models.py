"""Core task graph models for PAPoG.

Defines TaskStatus, TaskNode, and TaskGraph — the backbone of the
parallel agentic plan-over-graph execution model.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle states for a TaskNode."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    """A single unit of work in the task graph."""

    id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    assigned_agent: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_ready(self, completed_ids: set[str]) -> bool:
        """Return True when every dependency has been completed."""
        return all(dep in completed_ids for dep in self.dependencies)

    def mark_running(self, agent_id: str) -> None:
        """Transition to RUNNING and record which agent owns the task."""
        self.status = TaskStatus.RUNNING
        self.assigned_agent = agent_id

    def mark_completed(self, result: Any) -> None:
        """Transition to COMPLETED and store the result."""
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Transition to FAILED and store the error in result."""
        self.status = TaskStatus.FAILED
        self.result = error


class TaskGraph(BaseModel):
    """DAG of TaskNodes backed by networkx.

    Maintains a parallel ``dict[str, TaskNode]`` for fast lookup and a
    ``nx.DiGraph`` for topological queries.
    """

    nodes: dict[str, TaskNode] = Field(default_factory=dict)

    # networkx graph kept in sync but excluded from Pydantic serialisation
    _graph: nx.DiGraph = nx.DiGraph()

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        """Rebuild the internal DiGraph after (de)serialisation."""
        self._graph = nx.DiGraph()
        for node in self.nodes.values():
            self._graph.add_node(node.id)
            for dep in node.dependencies:
                self._graph.add_edge(dep, node.id)

    # -- mutators -----------------------------------------------------------

    def add_node(self, node: TaskNode) -> None:
        """Insert a TaskNode, creating dependency edges."""
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        self._graph.add_node(node.id)
        for dep in node.dependencies:
            self._graph.add_edge(dep, node.id)

    # -- queries ------------------------------------------------------------

    def get_node(self, node_id: str) -> TaskNode:
        """Look up a node by id; raise KeyError if missing."""
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"No node with id '{node_id}'") from None

    def get_ready_nodes(self, completed_ids: set[str] | None = None) -> list[TaskNode]:
        """Return nodes whose dependencies are all satisfied.

        If *completed_ids* is ``None`` it is derived from nodes whose
        status is ``COMPLETED``.
        """
        if completed_ids is None:
            completed_ids = {
                nid for nid, n in self.nodes.items() if n.status == TaskStatus.COMPLETED
            }
        return [
            n
            for n in self.nodes.values()
            if n.status == TaskStatus.PENDING and n.is_ready(completed_ids)
        ]

    def validate_dag(self) -> None:
        """Raise ``ValueError`` if the graph contains a cycle."""
        if not nx.is_directed_acyclic_graph(self._graph):
            cycles = list(nx.simple_cycles(self._graph))
            raise ValueError(f"Graph contains cycle(s): {cycles}")

    def topological_order(self) -> list[str]:
        """Return node IDs in topological (execution) order.

        Raises ``ValueError`` if the graph is not a DAG.
        """
        self.validate_dag()
        return list(nx.topological_sort(self._graph))

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire graph to a plain dict."""
        return {
            "nodes": {nid: n.model_dump() for nid, n in self.nodes.items()}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        """Reconstruct a TaskGraph from a dict produced by ``to_dict``."""
        nodes = {
            nid: TaskNode.model_validate(ndata)
            for nid, ndata in data["nodes"].items()
        }
        return cls(nodes=nodes)
