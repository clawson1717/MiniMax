from typing import List, Set, Dict, Any, Optional
from src.trajectory import TrajectoryLogger

class TrajectoryPruner:
    """
    Identifies and prunes cycles or unproductive branches from the trajectory graph.
    Inspired by WebClipper's graph-based optimization.
    """
    def __init__(self, logger: TrajectoryLogger):
        self.logger = logger
        self.pruned_node_ids: Set[str] = set()
        self.pruned_edge_indices: Set[int] = set()

    def detect_cycles(self) -> List[List[str]]:
        """
        Identifies cycles in the trajectory graph.
        A cycle is a path that leads back to a previously visited state.
        
        Returns:
            A list of cycles, where each cycle is a list of node IDs.
        """
        cycles = []
        adj = self._get_adjacency_list()
        
        # We'll use DFS to find cycles
        visited = set()
        stack = []
        
        def dfs(u, path_stack, visited_in_path):
            visited.add(u)
            path_stack.append(u)
            visited_in_path.add(u)
            
            for v in adj.get(u, []):
                if v in visited_in_path:
                    # Found a cycle!
                    cycle_start_index = path_stack.index(v)
                    cycles.append(path_stack[cycle_start_index:].copy())
                elif v not in visited:
                    dfs(v, path_stack, visited_in_path)
            
            path_stack.pop()
            visited_in_path.remove(u)

        for node_id in self.logger.nodes:
            if node_id not in visited:
                dfs(node_id, [], set())
        
        return cycles

    def prune_unproductive_branches(self, uncertainty_threshold: float):
        """
        Identifies and marks nodes/edges where uncertainty remains high or progress is stagnant.
        
        Args:
            uncertainty_threshold: The threshold above which a node is considered too uncertain.
        """
        for node_id, node in self.logger.nodes.items():
            metadata = node.get("metadata", {})
            uncertainty = metadata.get("uncertainty")
            
            # If uncertainty exists and is above threshold, we consider it a candidate for pruning
            # In a real scenario, we might also check if this node led to any improvement
            if uncertainty is not None and uncertainty > uncertainty_threshold:
                self.pruned_node_ids.add(node_id)
        
        # Also prune edges connected to pruned nodes
        for i, edge in enumerate(self.logger.edges):
            if edge["from"] in self.pruned_node_ids or edge["to"] in self.pruned_node_ids:
                self.pruned_edge_indices.add(i)

    def get_clean_trajectory(self) -> Dict[str, Any]:
        """
        Returns the optimized graph, excluding pruned nodes and edges.
        Also attempts to bypass detected cycles.
        """
        # For simplicity in this step, we just exclude the marked ones
        # A more advanced version would find the shortest non-cyclic path
        
        clean_nodes = [
            node for node_id, node in self.logger.nodes.items() 
            if node_id not in self.pruned_node_ids
        ]
        
        clean_edges = [
            edge for i, edge in enumerate(self.logger.edges)
            if i not in self.pruned_edge_indices
        ]
        
        return {
            "nodes": clean_nodes,
            "edges": clean_edges
        }

    def _get_adjacency_list(self) -> Dict[str, List[str]]:
        adj = {}
        for edge in self.logger.edges:
            u, v = edge["from"], edge["to"]
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
        return adj
