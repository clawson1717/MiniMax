"""Trajectory graph for managing agent paths and recovery."""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class TrajectoryNode:
    """A node in the trajectory graph."""
    step_index: int
    action: str
    observation: str = ""
    uncertainty: Optional[float] = None
    children: List[int] = field(default_factory=list)
    parent: Optional[int] = None
    is_pruned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryGraph:
    """
    Graph-based trajectory representation supporting branching and recovery.
    
    Inspired by CM2's trajectory graph for efficient recovery.
    """
    
    def __init__(self):
        """Initialize empty trajectory graph."""
        self.nodes: Dict[int, TrajectoryNode] = {}
        self.current_node_id: int = -1
        self.root_node_id: Optional[int] = None
        
    def add_node(self, action: str, observation: str = "", 
                 uncertainty: Optional[float] = None,
                 parent_id: Optional[int] = None) -> int:
        """Add a new node to the trajectory.
        
        Args:
            action: The action taken.
            observation: The resulting observation.
            uncertainty: Uncertainty score for this step.
            parent_id: Parent node ID (None for root).
            
        Returns:
            The new node ID.
        """
        node_id = len(self.nodes)
        node = TrajectoryNode(
            step_index=node_id,
            action=action,
            observation=observation,
            uncertainty=uncertainty,
            parent=parent_id,
        )
        
        self.nodes[node_id] = node
        
        if parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
            
        if self.root_node_id is None:
            self.root_node_id = node_id
            
        self.current_node_id = node_id
        return node_id
    
    def get_path(self, node_id: Optional[int] = None) -> List[TrajectoryNode]:
        """Get path from root to specified node.
        
        Args:
            node_id: Target node ID. Uses current node if not specified.
            
        Returns:
            List of nodes from root to target.
        """
        if node_id is None:
            node_id = self.current_node_id
            
        path = []
        current = node_id
        
        while current is not None and current in self.nodes:
            path.append(self.nodes[current])
            current = self.nodes[current].parent
            
        return list(reversed(path))
    
    def find_recovery_point(self, threshold: float = 0.7) -> Optional[int]:
        """Find a suitable recovery point based on uncertainty.
        
        Args:
            threshold: Uncertainty threshold.
            
        Returns:
            Node ID to recover to, or None if no recovery needed.
        """
        # TODO: Implement recovery point selection
        # Look for last node with uncertainty below threshold
        path = self.get_path()
        for node in reversed(path):
            if node.uncertainty is not None and node.uncertainty < threshold:
                return node.step_index
        return self.root_node_id
    
    def prune_branch(self, node_id: int) -> Set[int]:
        """Prune a branch starting from specified node.
        
        Args:
            node_id: Node to start pruning from.
            
        Returns:
            Set of pruned node IDs.
        """
        pruned = set()
        
        def _prune_recursive(nid: int):
            if nid not in self.nodes:
                return
            self.nodes[nid].is_pruned = True
            pruned.add(nid)
            for child_id in self.nodes[nid].children:
                _prune_recursive(child_id)
                
        _prune_recursive(node_id)
        return pruned
    
    def get_siblings(self, node_id: Optional[int] = None) -> List[TrajectoryNode]:
        """Get sibling nodes (other children of the same parent).
        
        Args:
            node_id: Node to find siblings for. Uses current node if not specified.
            
        Returns:
            List of sibling nodes.
        """
        if node_id is None:
            node_id = self.current_node_id
            
        if node_id not in self.nodes:
            return []
            
        parent_id = self.nodes[node_id].parent
        if parent_id is None:
            return []
            
        siblings = []
        for child_id in self.nodes[parent_id].children:
            if child_id != node_id:
                siblings.append(self.nodes[child_id])
                
        return siblings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": {
                nid: {
                    "step_index": node.step_index,
                    "action": node.action,
                    "observation": node.observation,
                    "uncertainty": node.uncertainty,
                    "children": node.children,
                    "parent": node.parent,
                    "is_pruned": node.is_pruned,
                    "metadata": node.metadata,
                }
                for nid, node in self.nodes.items()
            },
            "current_node_id": self.current_node_id,
            "root_node_id": self.root_node_id,
        }
    
    def save(self, filepath: str) -> None:
        """Save trajectory graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def reset(self) -> None:
        """Clear all nodes."""
        self.nodes = {}
        self.current_node_id = -1
        self.root_node_id = None