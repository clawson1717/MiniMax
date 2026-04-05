"""
Graph Pruner for TrajectoryGraph.

Detects and removes cyclic loops and dead branches from a TrajectoryGraph to
support WebClipper-style trajectory pruning (reducing redundant tool calls).

The pruner is non-mutating: it returns a new TrajectoryGraph rather than
modifying the input. It relies only on the Python standard library.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from src.trajectory import TrajectoryGraph


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def detect_cycles(graph: TrajectoryGraph) -> List[List[Any]]:
    """
    Find all simple cycles in the node graph.

    A simple cycle is a path ``v0 -> v1 -> ... -> vk -> v0`` where all
    intermediate nodes are distinct. Each cycle is returned exactly once
    (canonicalised to start at its lexicographically-smallest node to avoid
    rotations).

    Uses Johnson-style DFS over successor adjacency derived from
    ``graph.nodes`` (``{state: [(action, next_state), ...]}``).

    Args:
        graph: A ``TrajectoryGraph`` instance.

    Returns:
        A list of cycles. Each cycle is a list of states in traversal order
        (the first node is repeated implicitly as the closing edge's target).
    """
    # Build plain successor adjacency (drop actions, dedupe targets).
    adj: Dict[Any, List[Any]] = {}
    for src, edges in graph.nodes.items():
        seen: List[Any] = []
        for _action, nxt in edges:
            if nxt not in seen:
                seen.append(nxt)
        adj[src] = seen

    found: List[List[Any]] = []
    seen_signatures: Set[Tuple[Any, ...]] = set()

    def _canonical(cycle: List[Any]) -> Tuple[Any, ...]:
        """Return a rotation-invariant signature for a cycle."""
        # Rotate so the cycle starts at its smallest element (by repr for
        # heterogeneous / unorderable states).
        n = len(cycle)
        best_idx = 0
        best_key = repr(cycle[0])
        for i in range(1, n):
            key = repr(cycle[i])
            if key < best_key:
                best_key = key
                best_idx = i
        return tuple(cycle[best_idx:] + cycle[:best_idx])

    def _dfs(start: Any, current: Any, path: List[Any], on_path: Set[Any]) -> None:
        for nxt in adj.get(current, []):
            if nxt == start and len(path) >= 1:
                # Closed a cycle back to start.
                sig = _canonical(path)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    found.append(list(path))
            elif nxt not in on_path:
                on_path.add(nxt)
                path.append(nxt)
                _dfs(start, nxt, path, on_path)
                path.pop()
                on_path.discard(nxt)

    for node in list(adj.keys()):
        _dfs(node, node, [node], {node})

    return found


# ---------------------------------------------------------------------------
# Dead branch detection
# ---------------------------------------------------------------------------


def detect_dead_branches(
    graph: TrajectoryGraph, goal_states: Set[Any]
) -> List[Any]:
    """
    Find nodes that cannot reach any goal state.

    Performs a reverse BFS from the goal states across the graph's edges and
    returns every node in ``graph.nodes`` that was not visited.

    Args:
        graph: A ``TrajectoryGraph`` instance.
        goal_states: The set of terminal/goal states. If empty, every node is
            considered dead (nothing is reachable from "no goal").

    Returns:
        List of nodes that cannot reach any goal state.
    """
    if not goal_states:
        return list(graph.nodes.keys())

    # Build reverse adjacency: for each edge src -> nxt, record nxt -> src.
    reverse: Dict[Any, List[Any]] = {node: [] for node in graph.nodes}
    for src, edges in graph.nodes.items():
        for _action, nxt in edges:
            reverse.setdefault(nxt, []).append(src)
            reverse.setdefault(src, reverse.get(src, []))

    visited: Set[Any] = set()
    queue: deque = deque()
    for goal in goal_states:
        if goal in graph.nodes or goal in reverse:
            if goal not in visited:
                visited.add(goal)
                queue.append(goal)

    while queue:
        node = queue.popleft()
        for pred in reverse.get(node, []):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)

    dead: List[Any] = [n for n in graph.nodes if n not in visited]
    return dead


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def _find_first_cycle_edge(
    nodes: Dict[Any, List[Tuple[Any, Any]]]
) -> Optional[Tuple[Any, Any, Any]]:
    """
    DFS over the node adjacency looking for a back-edge that closes a cycle.

    Returns ``(src, action, nxt)`` describing the edge to remove, or ``None``
    if the graph is acyclic.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[Any, int] = {n: WHITE for n in nodes}

    # Iterative DFS using an explicit stack to keep recursion bounded.
    for root in list(nodes.keys()):
        if color[root] != WHITE:
            continue
        stack: List[Tuple[Any, int]] = [(root, 0)]
        color[root] = GRAY
        while stack:
            node, idx = stack[-1]
            edges = nodes.get(node, [])
            if idx >= len(edges):
                color[node] = BLACK
                stack.pop()
                continue
            action, nxt = edges[idx]
            # Advance the iterator position on the stack.
            stack[-1] = (node, idx + 1)
            if nxt not in color:
                color[nxt] = WHITE
            if color[nxt] == GRAY:
                # Back-edge: closes a cycle. Remove this edge.
                return (node, action, nxt)
            if color[nxt] == WHITE:
                color[nxt] = GRAY
                stack.append((nxt, 0))
    return None


def _remove_cycles(
    nodes: Dict[Any, List[Tuple[Any, Any]]],
    trajectory: List[Tuple[Any, Any, Any]],
) -> Tuple[Dict[Any, List[Tuple[Any, Any]]], List[Tuple[Any, Any, Any]]]:
    """
    Iteratively remove back-edges until the adjacency graph is acyclic.

    For each back-edge ``(src, action, nxt)`` discovered by DFS:
      * drop it from ``nodes[src]``
      * drop every trajectory step matching ``(src, action, nxt)``

    Operates on copies and returns the new structures.
    """
    new_nodes: Dict[Any, List[Tuple[Any, Any]]] = {
        k: list(v) for k, v in nodes.items()
    }
    new_trajectory: List[Tuple[Any, Any, Any]] = list(trajectory)

    while True:
        edge = _find_first_cycle_edge(new_nodes)
        if edge is None:
            break
        src, action, nxt = edge
        # Remove edge from adjacency.
        new_nodes[src] = [
            (a, n) for (a, n) in new_nodes[src] if not (a == action and n == nxt)
        ]
        # Remove all trajectory steps that used that edge.
        new_trajectory = [
            step for step in new_trajectory
            if not (step[0] == src and step[1] == action and step[2] == nxt)
        ]

    return new_nodes, new_trajectory


def prune(
    graph: TrajectoryGraph, goal_states: Optional[Set[Any]] = None
) -> TrajectoryGraph:
    """
    Return a new pruned ``TrajectoryGraph``.

    Steps:
      1. Remove back-edges so the node graph becomes acyclic. The first path
         discovered by DFS is preserved; edges that loop back to an ancestor
         are dropped (with their matching trajectory steps).
      2. If ``goal_states`` is provided and non-empty, remove any node that
         cannot reach a goal via reverse BFS; also drop trajectory steps that
         touch those nodes.

    The input graph is not modified.

    Args:
        graph: The ``TrajectoryGraph`` to prune.
        goal_states: Optional set of terminal/goal states. If ``None`` or
            empty, only cycle removal is performed.

    Returns:
        A new ``TrajectoryGraph`` containing the pruned structure.
    """
    # Step 1: cycle removal (operate on copies of the input graph).
    nodes_copy: Dict[Any, List[Tuple[Any, Any]]] = {
        k: list(v) for k, v in graph.nodes.items()
    }
    trajectory_copy: List[Tuple[Any, Any, Any]] = list(graph.trajectory)
    new_nodes, new_trajectory = _remove_cycles(nodes_copy, trajectory_copy)

    # Step 2: dead branch removal (only when we have goals).
    if goal_states:
        # Build a temporary TrajectoryGraph view for reuse of reverse BFS.
        temp = TrajectoryGraph()
        temp.nodes = new_nodes
        temp.trajectory = new_trajectory
        dead = set(detect_dead_branches(temp, set(goal_states)))
        if dead:
            # Drop dead nodes from adjacency map.
            new_nodes = {
                n: [(a, nxt) for (a, nxt) in edges if nxt not in dead]
                for n, edges in new_nodes.items()
                if n not in dead
            }
            # Drop trajectory steps that touch a dead node.
            new_trajectory = [
                step for step in new_trajectory
                if step[0] not in dead and step[2] not in dead
            ]

    pruned = TrajectoryGraph()
    pruned.nodes = new_nodes
    pruned.trajectory = new_trajectory
    return pruned


# ---------------------------------------------------------------------------
# Convenience class wrapper
# ---------------------------------------------------------------------------


class TrajectoryPruner:
    """
    Thin OO wrapper around the module-level pruning functions.

    Holds no state; provided for callers that prefer a class-based API.
    """

    def detect_cycles(self, graph: TrajectoryGraph) -> List[List[Any]]:
        return detect_cycles(graph)

    def detect_dead_branches(
        self, graph: TrajectoryGraph, goal_states: Set[Any]
    ) -> List[Any]:
        return detect_dead_branches(graph, goal_states)

    def prune(
        self,
        graph: TrajectoryGraph,
        goal_states: Optional[Set[Any]] = None,
    ) -> TrajectoryGraph:
        return prune(graph, goal_states)
