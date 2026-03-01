"""
Reasoning Trajectory Graph for tracking agent reasoning paths.

Implements a directed graph structure where nodes represent reasoning steps
and edges represent transitions between steps. Supports cycle detection,
path finding, and graph traversal operations.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass
class StepNode:
    """
    A node representing a single reasoning step in the trajectory graph.
    
    Attributes:
        step_id: Unique identifier for this step
        content: The reasoning content/text of this step
        agent_id: Identifier of the agent that produced this step
        tokens_used: Number of tokens consumed for this step
        confidence: Confidence score for this step (0.0 to 1.0)
        timestamp: Unix timestamp when this step was created
        metadata: Optional additional data about this step
        in_cycle: Whether this node is part of a detected cycle
    """
    step_id: str
    content: str
    agent_id: str
    tokens_used: int
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_cycle: bool = False
    
    def __post_init__(self):
        """Validate the step node after initialization."""
        if not self.step_id:
            raise ValueError("step_id must not be empty")
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step node to a dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "content": self.content,
            "agent_id": self.agent_id,
            "tokens_used": self.tokens_used,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "in_cycle": self.in_cycle
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepNode":
        """Create a StepNode from a dictionary."""
        return cls(
            step_id=data["step_id"],
            content=data["content"],
            agent_id=data["agent_id"],
            tokens_used=data["tokens_used"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            in_cycle=data.get("in_cycle", False)
        )


class TrajectoryGraph:
    """
    A directed graph representing reasoning trajectories.
    
    Nodes are reasoning steps, and edges represent transitions between steps.
    Supports cycle detection, path finding, and various traversal operations.
    
    Example:
        >>> graph = TrajectoryGraph()
        >>> step1 = graph.add_step("Initial thought", "agent1", 100, 0.8)
        >>> step2 = graph.add_step("Follow-up", "agent2", 50, 0.9, parent_id=step1)
        >>> path = graph.get_path(step1, step2)
    """
    
    def __init__(self):
        """Initialize an empty trajectory graph."""
        self._nodes: Dict[str, StepNode] = {}
        self._edges: Dict[str, Set[str]] = {}  # parent_id -> set of child_ids
        self._reverse_edges: Dict[str, Set[str]] = {}  # child_id -> set of parent_ids
    
    def add_step(
        self,
        content: str,
        agent_id: str,
        tokens: int,
        confidence: float,
        parent_id: Optional[str] = None,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Add a new reasoning step to the graph.
        
        Args:
            content: The reasoning content for this step
            agent_id: ID of the agent producing this step
            tokens: Number of tokens used
            confidence: Confidence score (0.0 to 1.0)
            parent_id: Optional parent step ID to create an edge from
            step_id: Optional custom step ID (auto-generated if not provided)
            metadata: Optional additional metadata
            timestamp: Optional timestamp (current time if not provided)
            
        Returns:
            The step_id of the newly created step
            
        Raises:
            ValueError: If parent_id doesn't exist in the graph
            ValueError: If step_id already exists
        """
        # Generate step ID if not provided
        if step_id is None:
            step_id = str(uuid.uuid4())
        
        # Check for duplicate step_id
        if step_id in self._nodes:
            raise ValueError(f"Step ID '{step_id}' already exists in the graph")
        
        # Validate parent exists if specified
        if parent_id is not None and parent_id not in self._nodes:
            raise ValueError(f"Parent step '{parent_id}' does not exist")
        
        # Create the step node
        node = StepNode(
            step_id=step_id,
            content=content,
            agent_id=agent_id,
            tokens_used=tokens,
            confidence=confidence,
            timestamp=timestamp if timestamp is not None else time.time(),
            metadata=metadata or {}
        )
        
        # Add to graph
        self._nodes[step_id] = node
        self._edges[step_id] = set()
        self._reverse_edges[step_id] = set()
        
        # Create edge from parent if specified
        if parent_id is not None:
            self._edges[parent_id].add(step_id)
            self._reverse_edges[step_id].add(parent_id)
        
        return step_id
    
    def get_step(self, step_id: str) -> Optional[StepNode]:
        """
        Get a step node by its ID.
        
        Args:
            step_id: The ID of the step to retrieve
            
        Returns:
            The StepNode if found, None otherwise
        """
        return self._nodes.get(step_id)
    
    def get_children(self, step_id: str) -> List[StepNode]:
        """
        Get all child nodes of a given step.
        
        Args:
            step_id: The ID of the parent step
            
        Returns:
            List of child StepNodes (empty if step doesn't exist or has no children)
        """
        if step_id not in self._nodes:
            return []
        
        child_ids = self._edges.get(step_id, set())
        return [self._nodes[cid] for cid in child_ids if cid in self._nodes]
    
    def get_parents(self, step_id: str) -> List[StepNode]:
        """
        Get all parent nodes of a given step.
        
        Args:
            step_id: The ID of the child step
            
        Returns:
            List of parent StepNodes (empty if step doesn't exist or has no parents)
        """
        if step_id not in self._nodes:
            return []
        
        parent_ids = self._reverse_edges.get(step_id, set())
        return [self._nodes[pid] for pid in parent_ids if pid in self._nodes]
    
    def get_ancestors(self, step_id: str) -> List[StepNode]:
        """
        Get all ancestor nodes of a given step (transitive parents).
        
        Args:
            step_id: The ID of the step
            
        Returns:
            List of ancestor StepNodes in order from immediate parents to root
        """
        if step_id not in self._nodes:
            return []
        
        ancestors = []
        visited = set()
        queue = list(self._reverse_edges.get(step_id, set()))
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in self._nodes:
                ancestors.append(self._nodes[current_id])
                # Add parents of current to queue
                for parent_id in self._reverse_edges.get(current_id, set()):
                    if parent_id not in visited:
                        queue.append(parent_id)
        
        return ancestors
    
    def get_descendants(self, step_id: str) -> List[StepNode]:
        """
        Get all descendant nodes of a given step (transitive children).
        
        Args:
            step_id: The ID of the step
            
        Returns:
            List of descendant StepNodes in BFS order
        """
        if step_id not in self._nodes:
            return []
        
        descendants = []
        visited = set()
        queue = list(self._edges.get(step_id, set()))
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in self._nodes:
                descendants.append(self._nodes[current_id])
                # Add children of current to queue
                for child_id in self._edges.get(current_id, set()):
                    if child_id not in visited:
                        queue.append(child_id)
        
        return descendants
    
    def get_path(self, start_id: str, end_id: str) -> Optional[List[StepNode]]:
        """
        Find a path between two nodes using BFS.
        
        Args:
            start_id: ID of the starting step
            end_id: ID of the ending step
            
        Returns:
            List of StepNodes representing the path from start to end,
            or None if no path exists
        """
        if start_id not in self._nodes or end_id not in self._nodes:
            return None
        
        if start_id == end_id:
            return [self._nodes[start_id]]
        
        # BFS to find shortest path
        visited = {start_id}
        queue: List[Tuple[str, List[str]]] = [(start_id, [start_id])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            for child_id in self._edges.get(current_id, set()):
                if child_id == end_id:
                    # Found path, convert to nodes
                    full_path = path + [child_id]
                    return [self._nodes[sid] for sid in full_path]
                
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, path + [child_id]))
        
        return None
    
    def get_all_paths(self, start_id: str, end_id: str) -> List[List[StepNode]]:
        """
        Find all paths between two nodes.
        
        Args:
            start_id: ID of the starting step
            end_id: ID of the ending step
            
        Returns:
            List of paths, where each path is a list of StepNodes
        """
        if start_id not in self._nodes or end_id not in self._nodes:
            return []
        
        if start_id == end_id:
            return [[self._nodes[start_id]]]
        
        all_paths = []
        
        def dfs(current_id: str, path: List[str], visited: Set[str]):
            if current_id == end_id:
                all_paths.append([self._nodes[sid] for sid in path])
                return
            
            for child_id in self._edges.get(current_id, set()):
                if child_id not in visited:
                    visited.add(child_id)
                    dfs(child_id, path + [child_id], visited)
                    visited.remove(child_id)
        
        dfs(start_id, [start_id], {start_id})
        return all_paths
    
    def detect_cycles(self) -> List[List[StepNode]]:
        """
        Detect all cycles in the graph using DFS.
        
        Returns:
            List of cycles, where each cycle is a list of StepNode IDs
            forming the cycle path. Also marks nodes that are part of cycles.
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for child_id in self._edges.get(node_id, set()):
                if child_id not in visited:
                    if dfs(child_id):
                        return True
                elif child_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(child_id)
                    cycle = path[cycle_start:] + [child_id]
                    cycles.append(cycle)
                    # Mark nodes in cycle
                    for cid in cycle:
                        if cid in self._nodes:
                            self._nodes[cid].in_cycle = True
                    return True
            
            path.pop()
            rec_stack.remove(node_id)
            return False
        
        # Run DFS from each unvisited node
        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)
        
        # Convert cycle IDs to nodes
        return [[self._nodes[sid] for sid in cycle if sid in self._nodes] 
                for cycle in cycles]
    
    def has_cycles(self) -> bool:
        """
        Check if the graph contains any cycles.
        
        Returns:
            True if cycles exist, False otherwise
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for child_id in self._edges.get(node_id, set()):
                if child_id not in visited:
                    if dfs(child_id):
                        return True
                elif child_id in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self._nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def get_roots(self) -> List[StepNode]:
        """
        Get all root nodes (nodes with no parents).
        
        Returns:
            List of root StepNodes
        """
        roots = []
        for step_id, node in self._nodes.items():
            if not self._reverse_edges.get(step_id):
                roots.append(node)
        return roots
    
    def get_leaves(self) -> List[StepNode]:
        """
        Get all leaf nodes (nodes with no children).
        
        Returns:
            List of leaf StepNodes
        """
        leaves = []
        for step_id, node in self._nodes.items():
            if not self._edges.get(step_id):
                leaves.append(node)
        return leaves
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        return len(self._nodes)
    
    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        return sum(len(children) for children in self._edges.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the graph to a dictionary.
        
        Returns:
            Dictionary representation of the graph
        """
        nodes_list = [node.to_dict() for node in self._nodes.values()]
        
        edges_list = []
        for parent_id, children in self._edges.items():
            for child_id in children:
                edges_list.append({"parent": parent_id, "child": child_id})
        
        return {
            "nodes": nodes_list,
            "edges": edges_list
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryGraph":
        """
        Create a TrajectoryGraph from a dictionary.
        
        Args:
            data: Dictionary with 'nodes' and 'edges' keys
            
        Returns:
            A new TrajectoryGraph instance
        """
        graph = cls()
        
        # Add nodes first (without edges)
        for node_data in data.get("nodes", []):
            node = StepNode.from_dict(node_data)
            graph._nodes[node.step_id] = node
            graph._edges[node.step_id] = set()
            graph._reverse_edges[node.step_id] = set()
        
        # Add edges
        for edge in data.get("edges", []):
            parent_id = edge["parent"]
            child_id = edge["child"]
            if parent_id in graph._nodes and child_id in graph._nodes:
                graph._edges[parent_id].add(child_id)
                graph._reverse_edges[child_id].add(parent_id)
        
        return graph
    
    def clear_cycle_marks(self) -> None:
        """Clear the in_cycle flag from all nodes."""
        for node in self._nodes.values():
            node.in_cycle = False
    
    def get_steps_by_agent(self, agent_id: str) -> List[StepNode]:
        """
        Get all steps produced by a specific agent.
        
        Args:
            agent_id: The agent ID to filter by
            
        Returns:
            List of StepNodes from the specified agent
        """
        return [node for node in self._nodes.values() if node.agent_id == agent_id]
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens used across all steps."""
        return sum(node.tokens_used for node in self._nodes.values())
    
    def get_average_confidence(self) -> float:
        """Get the average confidence across all steps."""
        if not self._nodes:
            return 0.0
        return sum(node.confidence for node in self._nodes.values()) / len(self._nodes)
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)
    
    def __contains__(self, step_id: str) -> bool:
        """Check if a step ID exists in the graph."""
        return step_id in self._nodes
    
    def __iter__(self):
        """Iterate over all step nodes."""
        return iter(self._nodes.values())
