from typing import List, Dict, Optional, Set
from src.node import TrajectoryNode, NodeStatus

class TrajectoryGraph:
    def __init__(self):
        self.nodes: Dict[str, TrajectoryNode] = {}

    def add_node(self, node: TrajectoryNode):
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes[node.id] = node

    def add_edge(self, parent_id: str, child_id: str):
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found.")
        if child_id not in self.nodes:
            raise ValueError(f"Child node {child_id} not found.")
        
        if self._would_create_cycle(parent_id, child_id):
            raise ValueError(f"Adding edge {parent_id} -> {child_id} would create a cycle.")

        parent_node = self.nodes[parent_id]
        child_node = self.nodes[child_id]
        
        if child_id not in parent_node.children_ids:
            parent_node.children_ids.append(child_id)
        
        # Update child's parent reference
        child_node.parent_id = parent_id

    def _would_create_cycle(self, start_node_id: str, target_node_id: str) -> bool:
        # Check if target_node_id is an ancestor of start_node_id
        # This is sufficient for tree-like structures where we only care about the path to root.
        # But let's use a more general reachability check if target can reach start.
        
        if start_node_id == target_node_id:
            return True
            
        # If target_node_id can reach start_node_id, then adding start -> target creates a cycle.
        visited = set()
        stack = [target_node_id]
        
        while stack:
            curr_id = stack.pop()
            if curr_id == start_node_id:
                return True
            if curr_id in visited:
                continue
            visited.add(curr_id)
            node = self.nodes.get(curr_id)
            if node:
                for cid in node.children_ids:
                    stack.append(cid)
        return False

    def get_node(self, node_id: str) -> Optional[TrajectoryNode]:
        return self.nodes.get(node_id)

    def get_path(self, end_node_id: str) -> List[TrajectoryNode]:
        path = []
        curr_id = end_node_id
        visited = set() # Avoid infinite loops just in case
        
        while curr_id and curr_id not in visited:
            node = self.nodes.get(curr_id)
            if not node:
                break
            path.append(node)
            visited.add(curr_id)
            curr_id = node.parent_id
            
        return path[::-1]

    def prune_branch(self, node_id: str):
        node = self.nodes.get(node_id)
        if not node:
            return
        
        node.status = NodeStatus.PRUNED
        for child_id in node.children_ids:
            self.prune_branch(child_id)

    def find_alternatives(self, node_id: str) -> List[str]:
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return []
        
        parent = self.nodes.get(node.parent_id)
        if not parent:
            return []
        
        return [cid for cid in parent.children_ids if cid != node_id]
