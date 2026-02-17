"""Trajectory Graph Builder - WebClipper approach.

Represents agent trajectories as directed graphs for cycle detection
and branch pruning based on the WebClipper paper concepts.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import hashlib
import json
from collections import deque


@dataclass
class GraphNode:
    """A state node in the trajectory graph.
    
    Represents a state with its observation hash and action taken to reach it.
    """
    state_id: int
    observation_hash: str
    action: str
    observation_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_pruned: bool = False
    timestamp: Optional[float] = None


@dataclass
class GraphEdge:
    """A transition edge between states in the trajectory graph.
    
    Represents the transition from one state to another with associated
    weight and success information.
    """
    edge_id: int
    source: int  # source state_id
    target: int  # target state_id
    weight: float = 1.0
    success: bool = True
    action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryGraph:
    """Graph-based trajectory representation for WebClipper approach.
    
    Implements:
    - State deduplication via hashing
    - Cycle detection using DFS
    - Branch pruning for dead-end/redundant paths
    - Path reconstruction from root
    
    Inspired by WebClipper's trajectory graph management.
    """
    
    def __init__(self):
        """Initialize empty trajectory graph."""
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: Dict[int, GraphEdge] = {}
        self.adjacency: Dict[int, List[int]] = {}  # state_id -> list of edge_ids
        self.reverse_adjacency: Dict[int, List[int]] = {}  # state_id -> list of edge_ids
        self.observation_to_state: Dict[str, int] = {}  # hash -> state_id
        self.root_state_id: Optional[int] = None
        self._next_node_id: int = 0
        self._next_edge_id: int = 0
        
    def get_state_hash(self, observation: Any) -> str:
        """Create deterministic hash for state deduplication.
        
        Args:
            observation: The observation data (string, dict, list, etc.)
            
        Returns:
            Deterministic hash string for the observation.
        """
        # Convert observation to consistent string representation
        if isinstance(observation, str):
            obs_str = observation
        else:
            # Use json.dumps with sorted keys for consistency
            obs_str = json.dumps(observation, sort_keys=True, default=str)
        
        # Create SHA-256 hash
        return hashlib.sha256(obs_str.encode('utf-8')).hexdigest()[:16]
    
    def add_state(self, observation: Any, action: str, 
                  metadata: Optional[Dict[str, Any]] = None,
                  timestamp: Optional[float] = None) -> int:
        """Add a new state node to the graph.
        
        Args:
            observation: The observation data for this state.
            action: The action taken to reach this state.
            metadata: Optional metadata dictionary.
            timestamp: Optional timestamp.
            
        Returns:
            The state_id of the new or existing state.
        """
        obs_hash = self.get_state_hash(observation)
        
        # Check if state already exists (deduplication)
        if obs_hash in self.observation_to_state:
            return self.observation_to_state[obs_hash]
        
        # Create new state
        state_id = self._next_node_id
        self._next_node_id += 1
        
        node = GraphNode(
            state_id=state_id,
            observation_hash=obs_hash,
            action=action,
            observation_data=observation,
            metadata=metadata or {},
            timestamp=timestamp
        )
        
        self.nodes[state_id] = node
        self.adjacency[state_id] = []
        self.reverse_adjacency[state_id] = []
        self.observation_to_state[obs_hash] = state_id
        
        # Set root if this is the first node
        if self.root_state_id is None:
            self.root_state_id = state_id
            
        return state_id
    
    def add_edge(self, from_state: int, to_state: int, 
                 weight: float = 1.0, success: bool = True,
                 action: str = "", metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a transition edge between states.
        
        Args:
            from_state: Source state_id.
            to_state: Target state_id.
            weight: Edge weight (default 1.0).
            success: Whether the transition was successful.
            action: The action that caused this transition.
            metadata: Optional metadata dictionary.
            
        Returns:
            The edge_id of the new edge.
            
        Raises:
            ValueError: If from_state or to_state doesn't exist.
        """
        if from_state not in self.nodes:
            raise ValueError(f"Source state {from_state} does not exist")
        if to_state not in self.nodes:
            raise ValueError(f"Target state {to_state} does not exist")
        
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        
        edge = GraphEdge(
            edge_id=edge_id,
            source=from_state,
            target=to_state,
            weight=weight,
            success=success,
            action=action,
            metadata=metadata or {}
        )
        
        self.edges[edge_id] = edge
        self.adjacency[from_state].append(edge_id)
        self.reverse_adjacency[to_state].append(edge_id)
        
        return edge_id
    
    def detect_cycles(self) -> List[List[int]]:
        """Detect cycles in the graph using DFS.
        
        Returns:
            List of cycles, where each cycle is a list of state_ids.
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(state_id: int) -> None:
            visited.add(state_id)
            rec_stack.add(state_id)
            path.append(state_id)
            
            # Explore neighbors via outgoing edges
            for edge_id in self.adjacency.get(state_id, []):
                edge = self.edges[edge_id]
                neighbor = edge.target
                
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle - extract it from path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(state_id)
        
        # Run DFS from each unvisited node
        for state_id in self.nodes:
            if state_id not in visited:
                dfs(state_id)
        
        return cycles
    
    def get_prunable_branches(self) -> List[int]:
        """Identify redundant/dead-end branches for pruning.
        
        A branch is considered prunable if:
        - It's a dead-end (no outgoing edges from leaf nodes)
        - It's part of a cycle that doesn't lead to goal progress
        - It has been marked as unsuccessful
        
        Returns:
            List of state_ids that are candidates for pruning.
        """
        prunable = set()
        
        # Find dead-end leaf nodes
        for state_id, node in self.nodes.items():
            if node.is_pruned:
                continue
                
            # Check if this is a leaf node (no outgoing edges)
            outgoing = self.adjacency.get(state_id, [])
            if not outgoing:
                prunable.add(state_id)
                continue
            
            # Check if all outgoing edges are unsuccessful
            all_failed = True
            for edge_id in outgoing:
                if self.edges[edge_id].success:
                    all_failed = False
                    break
            if all_failed:
                prunable.add(state_id)
        
        # Find nodes in cycles that don't lead to new states
        cycles = self.detect_cycles()
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle[:-1])  # Exclude duplicate last element
        
        for state_id in cycle_nodes:
            if state_id not in self.nodes or self.nodes[state_id].is_pruned:
                continue
                
            # Check if this node has productive outgoing edges outside all cycles
            has_productive_outgoing = False
            for edge_id in self.adjacency.get(state_id, []):
                edge = self.edges[edge_id]
                if edge.target not in cycle_nodes and edge.success:
                    has_productive_outgoing = True
                    break
            
            if not has_productive_outgoing:
                prunable.add(state_id)
        
        return list(prunable)
    
    def prune_branch(self, state_id: int) -> Set[int]:
        """Remove a branch starting from the specified state.
        
        Args:
            state_id: The state to start pruning from.
            
        Returns:
            Set of state_ids that were pruned.
            
        Raises:
            ValueError: If state_id doesn't exist.
        """
        if state_id not in self.nodes:
            raise ValueError(f"State {state_id} does not exist")
        
        pruned = set()
        
        def prune_recursive(sid: int) -> None:
            if sid in pruned or sid not in self.nodes:
                return
            
            # Mark as pruned
            self.nodes[sid].is_pruned = True
            pruned.add(sid)
            
            # Recursively prune children
            for edge_id in self.adjacency.get(sid, []):
                child_id = self.edges[edge_id].target
                prune_recursive(child_id)
        
        prune_recursive(state_id)
        return pruned
    
    def get_path_to_state(self, state_id: int) -> List[int]:
        """Reconstruct path from root to the given state.
        
        Args:
            state_id: Target state_id.
            
        Returns:
            List of state_ids from root to target.
            
        Raises:
            ValueError: If state_id doesn't exist.
        """
        if state_id not in self.nodes:
            raise ValueError(f"State {state_id} does not exist")
        
        # Use BFS to find shortest path from root
        if self.root_state_id is None:
            return []
        
        # BFS
        queue = deque([(self.root_state_id, [self.root_state_id])])
        visited = {self.root_state_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == state_id:
                return path
            
            for edge_id in self.adjacency.get(current, []):
                neighbor = self.edges[edge_id].target
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        # No path found
        return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph.
        
        Returns:
            Dictionary with node_count, edge_count, cycle_count, and depth.
        """
        node_count = len(self.nodes)
        edge_count = len(self.edges)
        
        # Count cycles
        cycles = self.detect_cycles()
        cycle_count = len(cycles)
        
        # Calculate max depth (longest path from root)
        depth = 0
        if self.root_state_id is not None:
            # BFS to find max depth
            queue = deque([(self.root_state_id, 0)])
            visited = {self.root_state_id}
            
            while queue:
                current, current_depth = queue.popleft()
                depth = max(depth, current_depth)
                
                for edge_id in self.adjacency.get(current, []):
                    neighbor = self.edges[edge_id].target
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
        
        # Count pruned vs active nodes
        pruned_count = sum(1 for node in self.nodes.values() if node.is_pruned)
        active_count = node_count - pruned_count
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "cycle_count": cycle_count,
            "depth": depth,
            "pruned_count": pruned_count,
            "active_count": active_count,
            "root_state_id": self.root_state_id
        }
    
    def get_successors(self, state_id: int) -> List[int]:
        """Get all successor state_ids for a given state.
        
        Args:
            state_id: The state to get successors for.
            
        Returns:
            List of successor state_ids.
        """
        if state_id not in self.nodes:
            return []
        
        successors = []
        for edge_id in self.adjacency.get(state_id, []):
            successors.append(self.edges[edge_id].target)
        return successors
    
    def get_predecessors(self, state_id: int) -> List[int]:
        """Get all predecessor state_ids for a given state.
        
        Args:
            state_id: The state to get predecessors for.
            
        Returns:
            List of predecessor state_ids.
        """
        if state_id not in self.nodes:
            return []
        
        predecessors = []
        for edge_id in self.reverse_adjacency.get(state_id, []):
            predecessors.append(self.edges[edge_id].source)
        return predecessors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": {
                sid: {
                    "state_id": node.state_id,
                    "observation_hash": node.observation_hash,
                    "action": node.action,
                    "is_pruned": node.is_pruned,
                    "metadata": node.metadata,
                    "timestamp": node.timestamp,
                }
                for sid, node in self.nodes.items()
            },
            "edges": {
                eid: {
                    "edge_id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "success": edge.success,
                    "action": edge.action,
                    "metadata": edge.metadata,
                }
                for eid, edge in self.edges.items()
            },
            "root_state_id": self.root_state_id,
            "stats": self.get_graph_stats()
        }
    
    def save(self, filepath: str) -> None:
        """Save trajectory graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def reset(self) -> None:
        """Clear all nodes and edges."""
        self.nodes = {}
        self.edges = {}
        self.adjacency = {}
        self.reverse_adjacency = {}
        self.observation_to_state = {}
        self.root_state_id = None
        self._next_node_id = 0
        self._next_edge_id = 0
