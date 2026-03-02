import heapq
from typing import List, Dict, Optional, Set, Tuple
from src.node import TrajectoryNode, NodeStatus
from src.graph import TrajectoryGraph

class Backtracker:
    """
    Implements backtracking strategies for the verification cascade.
    """
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def find_alternatives(self, 
                          failed_node_id: str, 
                          graph: TrajectoryGraph, 
                          excluded_nodes: Optional[Set[str]] = None) -> List[str]:
        """
        Finds alternative nodes to explore by traversing up the graph.
        
        Ranks alternatives using a priority queue based on:
        1. Number of verified ancestors (more is better).
        2. Path length (shorter is better).
        
        Args:
            failed_node_id: ID of the node that failed verification.
            graph: The trajectory graph.
            excluded_nodes: Set of node IDs to ignore (e.g., already failed or pruned).
            
        Returns:
            A prioritized list of alternative node IDs.
        """
        if excluded_nodes is None:
            excluded_nodes = set()
            
        failed_node = graph.get_node(failed_node_id)
        if not failed_node:
            return []

        priority_queue: List[Tuple[Tuple[int, int], str]] = []
        
        # Backtrack up the graph up to max_depth
        curr_node_id = failed_node.parent_id
        depth = 0
        visited_ancestors = set()
        
        path_to_failure = [n.id for n in graph.get_path(failed_node_id)]
        
        while curr_node_id and depth < self.max_depth:
            if curr_node_id in visited_ancestors:
                break
            visited_ancestors.add(curr_node_id)
            
            parent_node = graph.get_node(curr_node_id)
            if not parent_node:
                break
            
            # Find candidate siblings or descendants of siblings
            for sibling_id in parent_node.children_ids:
                # Skip the node we're backtracking from (it's in the failing path)
                if sibling_id in path_to_failure or sibling_id in excluded_nodes:
                    continue
                
                # Find the first pending nodes in this branch
                pending_alternatives = self._find_pending_front(sibling_id, graph, excluded_nodes)
                
                for alt_id in pending_alternatives:
                    # Calculate priority: (-verified_count, path_length)
                    path_to_alt = graph.get_path(alt_id)
                    path_length = len(path_to_alt)
                    verified_count = sum(1 for n in path_to_alt if n.status == NodeStatus.VERIFIED)
                    
                    priority = (-verified_count, path_length)
                    heapq.heappush(priority_queue, (priority, alt_id))
            
            # Move up to next level
            curr_node_id = parent_node.parent_id
            depth += 1
            
        # Return prioritized alternative node IDs, avoiding duplicates
        seen = set()
        results = []
        while priority_queue:
            _, node_id = heapq.heappop(priority_queue)
            if node_id not in seen:
                results.append(node_id)
                seen.add(node_id)
            
        return results

    def _find_pending_front(self, node_id: str, graph: TrajectoryGraph, excluded_nodes: Set[str]) -> List[str]:
        """
        Helper to find the nearest PENDING nodes in a branch.
        """
        node = graph.get_node(node_id)
        if not node or node_id in excluded_nodes or node.status in [NodeStatus.FAILED, NodeStatus.PRUNED]:
            return []
            
        if node.status == NodeStatus.PENDING:
            return [node_id]
            
        if node.status == NodeStatus.VERIFIED:
            results = []
            for child_id in node.children_ids:
                results.extend(self._find_pending_front(child_id, graph, excluded_nodes))
            return results
        
        return []

    def is_dead_end(self, alternatives: List[str]) -> bool:
        """
        Checks if the current search has reached a dead end.
        """
        return len(alternatives) == 0
