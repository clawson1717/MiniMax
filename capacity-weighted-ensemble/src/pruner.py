"""
Graph-Based Pruner for reasoning trajectories.

Implements the WebClipper technique for pruning unproductive reasoning paths
from trajectory graphs. Detects cycles, low-confidence branches, and 
high-cost/low-value paths.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from copy import deepcopy

from .trajectory import TrajectoryGraph, StepNode


@dataclass
class PruningResult:
    """
    Result of a graph pruning operation.
    
    Attributes:
        original_size: Number of nodes before pruning
        pruned_size: Number of nodes after pruning
        removed_nodes: List of node IDs that were removed
        cycles_found: Number of cycles detected
        strategy: Pruning strategy used
    """
    original_size: int
    pruned_size: int
    removed_nodes: List[str]
    cycles_found: int
    strategy: str
    
    @property
    def nodes_removed(self) -> int:
        """Return the number of nodes removed."""
        return self.original_size - self.pruned_size
    
    @property
    def reduction_ratio(self) -> float:
        """Return the ratio of nodes removed."""
        if self.original_size == 0:
            return 0.0
        return self.nodes_removed / self.original_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_size": self.original_size,
            "pruned_size": self.pruned_size,
            "removed_nodes": self.removed_nodes,
            "cycles_found": self.cycles_found,
            "strategy": self.strategy,
            "nodes_removed": self.nodes_removed,
            "reduction_ratio": self.reduction_ratio
        }


class TrajectoryPruner:
    """
    Prunes unproductive reasoning paths from trajectory graphs.
    
    Uses the WebClipper technique to identify and remove:
    - Cyclic reasoning loops (keep one representative node)
    - Low-confidence branches
    - High-cost branches (low value per token ratio)
    
    Example:
        >>> pruner = TrajectoryPruner(
        ...     min_confidence=0.3,
        ...     min_value_cost_ratio=0.001
        ... )
        >>> result = pruner.prune(graph, strategy="all")
        >>> print(f"Removed {result.nodes_removed} nodes")
    """
    
    def __init__(
        self,
        min_confidence: float = 0.2,
        min_value_cost_ratio: float = 0.0001,
        max_cycle_length: int = 100,
        preserve_roots: bool = True
    ):
        """
        Initialize the trajectory pruner.
        
        Args:
            min_confidence: Minimum confidence threshold for keeping nodes
            min_value_cost_ratio: Minimum value/cost ratio for keeping branches
            max_cycle_length: Maximum cycle length to detect
            preserve_roots: Whether to always keep root nodes
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if min_value_cost_ratio < 0:
            raise ValueError("min_value_cost_ratio must be non-negative")
        if max_cycle_length < 1:
            raise ValueError("max_cycle_length must be at least 1")
        
        self.min_confidence = min_confidence
        self.min_value_cost_ratio = min_value_cost_ratio
        self.max_cycle_length = max_cycle_length
        self.preserve_roots = preserve_roots
    
    def prune(
        self,
        graph: TrajectoryGraph,
        strategy: str = "all"
    ) -> Tuple[TrajectoryGraph, PruningResult]:
        """
        Prune the graph and return a pruned copy.
        
        Args:
            graph: The trajectory graph to prune
            strategy: Pruning strategy to use:
                - "cycles": Only remove cycles
                - "low_confidence": Only remove low-confidence branches
                - "high_cost": Only remove high-cost branches
                - "all": Apply all pruning strategies (default)
        
        Returns:
            Tuple of (pruned_graph, PruningResult)
        """
        if strategy not in ("cycles", "low_confidence", "high_cost", "all"):
            raise ValueError(f"Invalid strategy: {strategy}")
        
        original_size = graph.get_node_count()
        removed_nodes: Set[str] = set()
        
        # Identify cycles first
        cycles = self.identify_cycles(graph)
        cycles_found = len(cycles)
        
        if strategy in ("cycles", "all"):
            cycle_nodes = self._get_cycle_nodes_to_remove(graph, cycles)
            removed_nodes.update(cycle_nodes)
        
        if strategy in ("low_confidence", "all"):
            low_conf_nodes = self._find_low_confidence_nodes(graph)
            removed_nodes.update(low_conf_nodes)
        
        if strategy in ("high_cost", "all"):
            high_cost_nodes = self._find_high_cost_nodes(graph)
            removed_nodes.update(high_cost_nodes)
        
        # Preserve roots if configured
        if self.preserve_roots:
            root_ids = {r.step_id for r in graph.get_roots()}
            removed_nodes -= root_ids
        
        # Create pruned graph
        pruned_graph = self._create_pruned_graph(graph, removed_nodes)
        
        return pruned_graph, PruningResult(
            original_size=original_size,
            pruned_size=pruned_graph.get_node_count(),
            removed_nodes=sorted(list(removed_nodes)),
            cycles_found=cycles_found,
            strategy=strategy
        )
    
    def identify_cycles(self, graph: TrajectoryGraph) -> List[List[str]]:
        """
        Find all cycles in the graph.
        
        Uses DFS to detect back edges and extract cycles.
        
        Args:
            graph: The trajectory graph to analyze
        
        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []
        
        def dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for child_id in graph._edges.get(node_id, set()):
                if len(path) > self.max_cycle_length:
                    continue
                    
                if child_id not in visited:
                    dfs(child_id)
                elif child_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(child_id)
                    cycle = path[cycle_start:] + [child_id]
                    if len(cycle) <= self.max_cycle_length:
                        cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(node_id)
        
        # Run DFS from each unvisited node
        for node_id in list(graph._nodes.keys()):
            if node_id not in visited:
                dfs(node_id)
        
        return cycles
    
    def score_branch(
        self,
        graph: TrajectoryGraph,
        start_node_id: str
    ) -> float:
        """
        Score a branch by its value/cost ratio.
        
        Value = sum of confidence scores in the branch
        Cost = sum of tokens used in the branch
        
        Higher score = more valuable reasoning path.
        
        Args:
            graph: The trajectory graph
            start_node_id: ID of the starting node for the branch
        
        Returns:
            Branch score (confidence_sum / token_sum)
        """
        if start_node_id not in graph._nodes:
            return 0.0
        
        # Get all nodes in the branch (start node + all descendants)
        branch_nodes: List[StepNode] = [graph._nodes[start_node_id]]
        branch_nodes.extend(graph.get_descendants(start_node_id))
        
        if not branch_nodes:
            return 0.0
        
        # Calculate sums
        confidence_sum = sum(node.confidence for node in branch_nodes)
        token_sum = sum(node.tokens_used for node in branch_nodes)
        
        # Avoid division by zero
        if token_sum == 0:
            if confidence_sum > 0:
                return float('inf')
            return 0.0
        
        return confidence_sum / token_sum
    
    def should_prune(
        self,
        node: StepNode,
        score: float
    ) -> bool:
        """
        Determine if a branch should be pruned based on its score.
        
        Args:
            node: The starting node of the branch
            score: The branch score (from score_branch)
        
        Returns:
            True if the branch should be pruned
        """
        # Prune if score is below minimum
        if score < self.min_value_cost_ratio:
            return True
        
        # Prune if node has very low confidence
        if node.confidence < self.min_confidence:
            return True
        
        return False
    
    def _get_cycle_nodes_to_remove(
        self,
        graph: TrajectoryGraph,
        cycles: List[List[str]]
    ) -> Set[str]:
        """
        Determine which nodes to remove from cycles.
        
        Keeps one representative node per cycle (the one with highest confidence).
        
        Args:
            graph: The trajectory graph
            cycles: List of detected cycles
        
        Returns:
            Set of node IDs to remove
        """
        nodes_to_remove: Set[str] = set()
        
        for cycle in cycles:
            if len(cycle) <= 1:
                continue
            
            # Get actual cycle nodes (excluding the repeated node at end)
            cycle_node_ids = set(cycle[:-1])
            
            # Find the node with highest confidence to keep
            best_node_id = None
            best_confidence = -1.0
            
            for node_id in cycle_node_ids:
                if node_id in graph._nodes:
                    node = graph._nodes[node_id]
                    if node.confidence > best_confidence:
                        best_confidence = node.confidence
                        best_node_id = node_id
            
            # Remove all other nodes in the cycle
            for node_id in cycle_node_ids:
                if node_id != best_node_id:
                    nodes_to_remove.add(node_id)
        
        return nodes_to_remove
    
    def _find_low_confidence_nodes(
        self,
        graph: TrajectoryGraph
    ) -> Set[str]:
        """
        Find nodes that should be removed due to low confidence.
        
        A node is removed if:
        - Its confidence is below min_confidence
        - It's not a root (if preserve_roots is True)
        - It has no high-confidence descendants
        
        Args:
            graph: The trajectory graph
        
        Returns:
            Set of node IDs to remove
        """
        nodes_to_remove: Set[str] = set()
        
        for node_id, node in graph._nodes.items():
            if node.confidence < self.min_confidence:
                # Check if any descendant has high confidence
                descendants = graph.get_descendants(node_id)
                has_high_conf_descendant = any(
                    d.confidence >= self.min_confidence 
                    for d in descendants
                )
                
                if not has_high_conf_descendant:
                    nodes_to_remove.add(node_id)
        
        return nodes_to_remove
    
    def _find_high_cost_nodes(
        self,
        graph: TrajectoryGraph
    ) -> Set[str]:
        """
        Find nodes that should be removed due to high cost (low value/cost ratio).
        
        Args:
            graph: The trajectory graph
        
        Returns:
            Set of node IDs to remove
        """
        nodes_to_remove: Set[str] = set()
        
        # Score all branches starting from roots
        roots = graph.get_roots()
        
        for root in roots:
            root_score = self.score_branch(graph, root.step_id)
            
            # Check each child branch
            children = graph.get_children(root.step_id)
            for child in children:
                branch_score = self.score_branch(graph, child.step_id)
                
                if self.should_prune(child, branch_score):
                    # Add child and all its descendants
                    nodes_to_remove.add(child.step_id)
                    for desc in graph.get_descendants(child.step_id):
                        nodes_to_remove.add(desc.step_id)
        
        # Also check leaf nodes individually
        leaves = graph.get_leaves()
        for leaf in leaves:
            if leaf.step_id in nodes_to_remove:
                continue
            # A leaf with very low score on its own
            if leaf.confidence < self.min_confidence:
                nodes_to_remove.add(leaf.step_id)
        
        return nodes_to_remove
    
    def _create_pruned_graph(
        self,
        graph: TrajectoryGraph,
        removed_nodes: Set[str]
    ) -> TrajectoryGraph:
        """
        Create a new graph with specified nodes removed.
        
        Args:
            graph: The original graph
            removed_nodes: Set of node IDs to remove
        
        Returns:
            A new TrajectoryGraph with nodes removed
        """
        pruned = TrajectoryGraph()
        
        # First, add all nodes that should be kept
        for node_id, node in graph._nodes.items():
            if node_id not in removed_nodes:
                pruned._nodes[node_id] = StepNode(
                    step_id=node.step_id,
                    content=node.content,
                    agent_id=node.agent_id,
                    tokens_used=node.tokens_used,
                    confidence=node.confidence,
                    timestamp=node.timestamp,
                    metadata=deepcopy(node.metadata),
                    in_cycle=False
                )
                pruned._edges[node_id] = set()
                pruned._reverse_edges[node_id] = set()
        
        # Then, add edges between remaining nodes
        for parent_id, children in graph._edges.items():
            if parent_id not in removed_nodes:
                for child_id in children:
                    if child_id not in removed_nodes:
                        pruned._edges[parent_id].add(child_id)
                        pruned._reverse_edges[child_id].add(parent_id)
        
        # Reconnect orphaned nodes to their nearest surviving ancestor
        for node_id in pruned._nodes:
            # If node has no parents but had parents in original graph
            if not pruned._reverse_edges.get(node_id):
                original_parents = graph._reverse_edges.get(node_id, set())
                # Find nearest surviving ancestor
                for orig_parent in original_parents:
                    if orig_parent not in removed_nodes:
                        # Direct parent survived, add edge
                        pruned._edges[orig_parent].add(node_id)
                        pruned._reverse_edges[node_id].add(orig_parent)
                        break
                    else:
                        # Parent was removed, find ancestor
                        ancestors = graph.get_ancestors(node_id)
                        for ancestor in ancestors:
                            if ancestor.step_id not in removed_nodes:
                                pruned._edges[ancestor.step_id].add(node_id)
                                pruned._reverse_edges[node_id].add(ancestor.step_id)
                                break
                        if pruned._reverse_edges.get(node_id):
                            break
        
        return pruned
    
    def get_branch_statistics(
        self,
        graph: TrajectoryGraph,
        start_node_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed statistics about a branch.
        
        Args:
            graph: The trajectory graph
            start_node_id: ID of the starting node
        
        Returns:
            Dictionary with branch statistics
        """
        if start_node_id not in graph._nodes:
            return {}
        
        nodes = [graph._nodes[start_node_id]]
        nodes.extend(graph.get_descendants(start_node_id))
        
        confidences = [n.confidence for n in nodes]
        tokens = [n.tokens_used for n in nodes]
        
        return {
            "start_node_id": start_node_id,
            "depth": self._calculate_depth(graph, start_node_id),
            "total_nodes": len(nodes),
            "total_tokens": sum(tokens),
            "total_confidence": sum(confidences),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "value_cost_ratio": self.score_branch(graph, start_node_id)
        }
    
    def _calculate_depth(
        self,
        graph: TrajectoryGraph,
        start_node_id: str
    ) -> int:
        """
        Calculate the maximum depth from a starting node.
        
        Args:
            graph: The trajectory graph
            start_node_id: ID of the starting node
        
        Returns:
            Maximum depth (0 for leaf nodes)
        """
        children = graph.get_children(start_node_id)
        if not children:
            return 0
        
        return 1 + max(
            self._calculate_depth(graph, child.step_id) 
            for child in children
        )
    
    def analyze_graph(
        self,
        graph: TrajectoryGraph
    ) -> Dict[str, Any]:
        """
        Analyze a graph and return comprehensive statistics.
        
        Args:
            graph: The trajectory graph to analyze
        
        Returns:
            Dictionary with graph analysis
        """
        cycles = self.identify_cycles(graph)
        roots = graph.get_roots()
        leaves = graph.get_leaves()
        
        # Score all root branches
        branch_scores = []
        for root in roots:
            score = self.score_branch(graph, root.step_id)
            branch_scores.append({
                "root_id": root.step_id,
                "score": score
            })
        
        # Find low confidence nodes
        low_conf_count = sum(
            1 for node in graph._nodes.values()
            if node.confidence < self.min_confidence
        )
        
        return {
            "total_nodes": graph.get_node_count(),
            "total_edges": graph.get_edge_count(),
            "total_tokens": graph.get_total_tokens(),
            "avg_confidence": graph.get_average_confidence(),
            "num_roots": len(roots),
            "num_leaves": len(leaves),
            "num_cycles": len(cycles),
            "cycle_nodes": sum(len(c) - 1 for c in cycles),  # Exclude duplicate end
            "low_confidence_nodes": low_conf_count,
            "branch_scores": branch_scores,
            "has_cycles": len(cycles) > 0
        }
