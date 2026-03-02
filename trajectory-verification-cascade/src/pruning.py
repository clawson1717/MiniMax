from typing import List, Optional, Set
from src.node import TrajectoryNode, NodeStatus
from src.graph import TrajectoryGraph

class PruningPolicy:
    """
    Implements the pruning policy for the Trajectory Verification Cascade.
    Rules for when to prune branches or backtrack in the reasoning process.
    """
    def __init__(self, unproductive_threshold: int = 3, compute_cost_per_node: float = 1.0):
        self.unproductive_threshold = unproductive_threshold
        self.compute_cost_per_node = compute_cost_per_node
        self.nodes_pruned = 0
        self.compute_saved = 0.0

    def detect_cycle_causal_node(self, node_id: str, graph: TrajectoryGraph) -> Optional[str]:
        """
        Detects if a node has caused a cycle in the trajectory leading to it.
        Returns the ID of the node that caused the cycle, or None if no cycle exists.
        """
        visited = set()
        path = []
        curr_id = node_id
        
        while curr_id:
            if curr_id in visited:
                # Cycle detected! The current node_id is part of a cycle.
                # In our case, if we follow parents up, and we hit the same node, it's a cycle.
                return curr_id
            visited.add(curr_id)
            path.append(curr_id)
            node = graph.get_node(curr_id)
            if not node or not node.parent_id:
                break
            curr_id = node.parent_id
        
        return None

    def is_unproductive(self, node_id: str, graph: TrajectoryGraph) -> bool:
        """
        Check if a node belongs to an unproductive branch.
        A branch is unproductive if score or confidence hasn't improved for N steps.
        """
        path = graph.get_path(node_id)
        if len(path) <= self.unproductive_threshold:
            return False
            
        # Check the last N + 1 nodes (including current)
        window = path[-(self.unproductive_threshold + 1):]
        
        # Improvement check: does any node in the window have a higher score/confidence than the base?
        base_node = window[0]
        for node in window[1:]:
            if node.score > base_node.score or node.confidence > base_node.confidence:
                return False
                
        return True

    def decide_prune_vs_backtrack(self, node_id: str, graph: TrajectoryGraph) -> str:
        """
        Decide whether to prune the branch at node_id or just backtrack to find alternatives.
        - Prune: if it's a cycle or unproductive.
        - Backtrack: if it just failed a checklist or failure mode was detected (handled by CascadeEngine).

        Returns "prune", "backtrack", or "none".
        """
        if self.detect_cycle_causal_node(node_id, graph):
            return "prune"
            
        if self.is_unproductive(node_id, graph):
            return "prune"
            
        return "none"

    def apply_pruning(self, node_id: str, graph: TrajectoryGraph):
        """
        Applies pruning to a node and its descendants, updating metrics.
        """
        # Count nodes in the subtree to calculate compute_saved
        nodes_to_prune = self._count_subtree_nodes(node_id, graph)
        
        graph.prune_branch(node_id)
        
        self.nodes_pruned += nodes_to_prune
        self.compute_saved += nodes_to_prune * self.compute_cost_per_node

    def _count_subtree_nodes(self, node_id: str, graph: TrajectoryGraph) -> int:
        count = 1
        node = graph.get_node(node_id)
        if node:
            for child_id in node.children_ids:
                count += self._count_subtree_nodes(child_id, graph)
        return count

    def get_metrics(self):
        return {
            "nodes_pruned": self.nodes_pruned,
            "compute_saved": self.compute_saved
        }
