"""Advanced Pruning Strategies - Cycle Detection & Pruning.

Implements sophisticated pruning strategies for trajectory graphs:
- Cycle elimination
- Dead-end pruning  
- Redundancy detection
- Composite strategy management

Combines concepts from CATTS (uncertainty-driven), WebClipper (trajectory graphs),
and CM2 (structured exploration).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import deque
import math
import hashlib
import json
from datetime import datetime

from src.trajectory_graph import TrajectoryGraph, GraphNode


@dataclass
class PruningContext:
    """Context information for pruning decisions.
    
    Provides additional information that pruning strategies may use
    to make decisions beyond the graph structure itself.
    """
    task_description: str = ""
    current_step: int = 0
    max_steps: int = 50
    uncertainty_scores: Dict[int, float] = field(default_factory=dict)
    success_history: Dict[int, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_state_uncertainty(self, state_id: int) -> float:
        """Get uncertainty score for a state."""
        return self.uncertainty_scores.get(state_id, 0.5)
    
    def was_successful(self, state_id: int) -> bool:
        """Check if a state led to successful outcomes."""
        return self.success_history.get(state_id, True)


@dataclass
class PruningDecision:
    """Result of a pruning decision.
    
    Contains the decision and reasoning for debugging/auditing.
    """
    state_id: int
    should_prune: bool
    strategy_name: str
    reason: str
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PruningStrategy(ABC):
    """Abstract base class for pruning strategies.
    
    All pruning strategies must implement the should_prune method
    which evaluates whether a given state should be pruned.
    """
    
    def __init__(self, name: str, priority: int = 0):
        """Initialize pruning strategy.
        
        Args:
            name: Human-readable name for this strategy.
            priority: Priority for composite strategies (higher = more important).
        """
        self.name = name
        self.priority = priority
        self._stats: Dict[str, Any] = {
            "evaluations": 0,
            "pruned": 0,
            "kept": 0,
        }
    
    @abstractmethod
    def should_prune(
        self, 
        state_id: int, 
        graph: TrajectoryGraph, 
        context: PruningContext
    ) -> PruningDecision:
        """Evaluate whether a state should be pruned.
        
        Args:
            state_id: The state to evaluate.
            graph: The trajectory graph containing the state.
            context: Additional context for the decision.
            
        Returns:
            PruningDecision with the result and reasoning.
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this strategy."""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset strategy statistics."""
        self._stats["evaluations"] = 0
        self._stats["pruned"] = 0
        self._stats["kept"] = 0
    
    def _record_evaluation(self) -> None:
        """Record that an evaluation was performed."""
        self._stats["evaluations"] += 1
    
    def _record_pruned(self) -> None:
        """Record a prune decision."""
        self._stats["pruned"] += 1
    
    def _record_kept(self) -> None:
        """Record a keep decision."""
        self._stats["kept"] += 1
    
    def _record_decision(self, decision: PruningDecision) -> None:
        """Record a pruning decision and update stats accordingly."""
        self._record_evaluation()
        if decision.should_prune:
            self._record_pruned()
        else:
            self._record_kept()


class CycleEliminationStrategy(PruningStrategy):
    """Prunes nodes in cycles without productive exits.
    
    A cycle is considered unproductive if:
    - All nodes in the cycle only lead to other nodes in the cycle
    - No exit edges lead to states outside the cycle with successful outcomes
    
    This prevents the agent from getting stuck in loops that don't
    make progress toward the goal.
    """
    
    def __init__(self, priority: int = 10):
        """Initialize cycle elimination strategy.
        
        Args:
            priority: Strategy priority (default 10, high priority).
        """
        super().__init__("CycleElimination", priority)
        self._cycle_cache: Optional[List[List[int]]] = None
        self._cache_valid: bool = False
    
    def should_prune(
        self, 
        state_id: int, 
        graph: TrajectoryGraph, 
        context: PruningContext
    ) -> PruningDecision:
        """Check if state is in an unproductive cycle."""
        self._record_evaluation()
        
        # Check if node exists and is not already pruned
        if state_id not in graph.nodes or graph.nodes[state_id].is_pruned:
            self._record_kept()
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name=self.name,
                reason="State does not exist or already pruned",
                priority=self.priority
            )
        
        # Detect cycles (with caching)
        cycles = self._get_cycles(graph)
        
        # Find which cycle(s) this state belongs to
        state_cycles = [c for c in cycles if state_id in c[:-1]]  # Exclude repeated end
        
        if not state_cycles:
            self._record_kept()
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name=self.name,
                reason="State not in any cycle",
                priority=self.priority
            )
        
        # Check if this state specifically has productive exits OUTSIDE the cycle
        # A state should only be pruned if it can't escape the cycle
        for cycle in state_cycles:
            cycle_nodes = set(cycle[:-1])  # Exclude duplicate end
            
            # Check for productive exits from THIS SPECIFIC state
            has_productive_exit = self._state_has_productive_exit(
                state_id, cycle_nodes, graph, context
            )
            
            if has_productive_exit:
                self._record_kept()
                return PruningDecision(
                    state_id=state_id,
                    should_prune=False,
                    strategy_name=self.name,
                    reason=f"Cycle has productive exit from state {state_id}",
                    priority=self.priority
                )
        
        # No productive exits found for this specific state - should prune
        self._record_pruned()
        return PruningDecision(
            state_id=state_id,
            should_prune=True,
            strategy_name=self.name,
            reason="State in cycle without productive exits",
            priority=self.priority,
            metadata={"cycles": state_cycles}
        )
    
    def _get_cycles(self, graph: TrajectoryGraph) -> List[List[int]]:
        """Get cycles from graph (with caching)."""
        if not self._cache_valid or self._cycle_cache is None:
            self._cycle_cache = graph.detect_cycles()
            self._cache_valid = True
        return self._cycle_cache
    
    def _state_has_productive_exit(
        self, 
        state_id: int,
        cycle_nodes: Set[int], 
        graph: TrajectoryGraph, 
        context: PruningContext
    ) -> bool:
        """Check if a specific state has productive exits outside the cycle."""
        outgoing_edges = graph.adjacency.get(state_id, [])
        
        for edge_id in outgoing_edges:
            edge = graph.edges[edge_id]
            target_id = edge.target
            
            # Exit leads outside the cycle
            if target_id not in cycle_nodes:
                # Check if the exit edge is successful
                if edge.success:
                    # Target state is considered productive if it's successful or leads to success
                    if context.was_successful(target_id):
                        return True
                    # Also consider it productive if it's not pruned and has outgoing edges
                    if target_id in graph.nodes and not graph.nodes[target_id].is_pruned:
                        if graph.adjacency.get(target_id, []):
                            return True
        
        return False
    
    def _leads_to_success(
        self, 
        state_id: int, 
        graph: TrajectoryGraph,
        context: PruningContext,
        excluded_nodes: Set[int],
        max_depth: int = 10
    ) -> bool:
        """Check if a state eventually leads to successful outcomes."""
        visited = set()
        queue = deque([(state_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            
            # Check if this state is successful
            if context.was_successful(current_id):
                return True
            
            # Explore successors
            for edge_id in graph.adjacency.get(current_id, []):
                edge = graph.edges[edge_id]
                if edge.success and edge.target not in excluded_nodes:
                    queue.append((edge.target, depth + 1))
        
        return False
    
    def invalidate_cache(self) -> None:
        """Invalidate the cycle cache (call when graph changes)."""
        self._cache_valid = False
        self._cycle_cache = None


class DeadEndStrategy(PruningStrategy):
    """Prunes leaf nodes and failed paths.
    
    Identifies states that:
    - Have no outgoing edges (dead ends)
    - Have only failed outgoing edges
    - Are part of branches that consistently fail
    
    This removes paths that cannot lead to successful outcomes.
    """
    
    def __init__(
        self, 
        priority: int = 5,
        propagate_failures: bool = True,
        failure_threshold: float = 0.8
    ):
        """Initialize dead-end strategy.
        
        Args:
            priority: Strategy priority (default 5).
            propagate_failures: Whether to propagate failure up the tree.
            failure_threshold: Threshold for considering a node a failure.
        """
        super().__init__("DeadEnd", priority)
        self.propagate_failures = propagate_failures
        self.failure_threshold = failure_threshold
    
    def should_prune(
        self, 
        state_id: int, 
        graph: TrajectoryGraph, 
        context: PruningContext
    ) -> PruningDecision:
        """Check if state is a dead end."""
        self._record_evaluation()
        
        if state_id not in graph.nodes or graph.nodes[state_id].is_pruned:
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name=self.name,
                reason="State does not exist or already pruned",
                priority=self.priority
            )
        
        node = graph.nodes[state_id]
        outgoing_edges = graph.adjacency.get(state_id, [])
        
        # Check 1: No outgoing edges (leaf node)
        if not outgoing_edges:
            self._record_pruned()
            return PruningDecision(
                state_id=state_id,
                should_prune=True,
                strategy_name=self.name,
                reason="Leaf node with no outgoing edges",
                priority=self.priority,
                metadata={"reason_type": "leaf"}
            )
        
        # Check 2: All outgoing edges failed
        all_failed = all(
            not graph.edges[edge_id].success 
            for edge_id in outgoing_edges
        )
        if all_failed:
            self._record_pruned()
            return PruningDecision(
                state_id=state_id,
                should_prune=True,
                strategy_name=self.name,
                reason="All outgoing edges failed",
                priority=self.priority,
                metadata={"reason_type": "all_failed"}
            )
        
        # Check 3: Consistently low success rate (if history available)
        if self.propagate_failures:
            failure_rate = self._calculate_failure_rate(state_id, graph, context)
            if failure_rate >= self.failure_threshold:
                self._record_pruned()
                return PruningDecision(
                    state_id=state_id,
                    should_prune=True,
                    strategy_name=self.name,
                    reason=f"High failure rate ({failure_rate:.2f})",
                    priority=self.priority,
                    metadata={"reason_type": "high_failure_rate", "rate": failure_rate}
                )
        
        self._record_kept()
        return PruningDecision(
            state_id=state_id,
            should_prune=False,
            strategy_name=self.name,
            reason="State has productive outgoing edges",
            priority=self.priority
        )
    
    def _calculate_failure_rate(
        self, 
        state_id: int, 
        graph: TrajectoryGraph,
        context: PruningContext
    ) -> float:
        """Calculate the failure rate for paths starting from this state."""
        # Count total descendants and failures
        total = 0
        failures = 0
        
        visited = set()
        stack = [state_id]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            total += 1
            if not context.was_successful(current):
                failures += 1
            
            # Add successors
            for edge_id in graph.adjacency.get(current, []):
                stack.append(graph.edges[edge_id].target)
        
        return failures / total if total > 0 else 0.0


class RedundancyStrategy(PruningStrategy):
    """Prunes states similar to existing ones (within ε distance).
    
    Uses state similarity metrics to detect and remove redundant states:
    - Observation hash comparison
    - Structural similarity
    - Action sequence similarity
    
    This prevents exploring equivalent states multiple times.
    """
    
    def __init__(
        self, 
        priority: int = 3,
        epsilon: float = 0.1,
        use_hash_comparison: bool = True,
        use_structural_similarity: bool = True
    ):
        """Initialize redundancy strategy.
        
        Args:
            priority: Strategy priority (default 3).
            epsilon: Similarity threshold (0-1, higher = more permissive).
            use_hash_comparison: Whether to use observation hash for similarity.
            use_structural_similarity: Whether to use structural similarity.
        """
        super().__init__("Redundancy", priority)
        self.epsilon = epsilon
        self.use_hash_comparison = use_hash_comparison
        self.use_structural_similarity = use_structural_similarity
        self._observation_hashes: Dict[str, int] = {}  # hash -> first state_id
    
    def should_prune(
        self, 
        state_id: int, 
        graph: TrajectoryGraph, 
        context: PruningContext
    ) -> PruningDecision:
        """Check if state is redundant with existing states."""
        self._record_evaluation()
        
        if state_id not in graph.nodes or graph.nodes[state_id].is_pruned:
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name=self.name,
                reason="State does not exist or already pruned",
                priority=self.priority
            )
        
        node = graph.nodes[state_id]
        
        # Check 1: Exact hash match with existing (non-pruned) state
        if self.use_hash_comparison:
            existing_id = self._observation_hashes.get(node.observation_hash)
            if existing_id is not None and existing_id != state_id:
                if existing_id in graph.nodes and not graph.nodes[existing_id].is_pruned:
                    self._record_pruned()
                    return PruningDecision(
                        state_id=state_id,
                        should_prune=True,
                        strategy_name=self.name,
                        reason=f"Duplicate observation hash (similar to state {existing_id})",
                        priority=self.priority,
                        metadata={
                            "reason_type": "hash_match",
                            "similar_state": existing_id,
                            "hash": node.observation_hash
                        }
                    )
            else:
                # Record this hash
                self._observation_hashes[node.observation_hash] = state_id
        
        # Check 2: Structural similarity with nearby states
        if self.use_structural_similarity:
            similar_state = self._find_structurally_similar(state_id, graph)
            if similar_state is not None:
                similarity = self._calculate_similarity(state_id, similar_state, graph)
                if similarity >= (1.0 - self.epsilon):
                    self._record_pruned()
                    return PruningDecision(
                        state_id=state_id,
                        should_prune=True,
                        strategy_name=self.name,
                        reason=f"Structurally similar to state {similar_state} (sim={similarity:.3f})",
                        priority=self.priority,
                        metadata={
                            "reason_type": "structural_similarity",
                            "similar_state": similar_state,
                            "similarity": similarity
                        }
                    )
        
        return PruningDecision(
            state_id=state_id,
            should_prune=False,
            strategy_name=self.name,
            reason="State is unique",
            priority=self.priority
        )
    
    def _find_structurally_similar(
        self, 
        state_id: int, 
        graph: TrajectoryGraph
    ) -> Optional[int]:
        """Find a structurally similar state."""
        # Look at predecessors and their other children
        predecessors = graph.get_predecessors(state_id)
        
        for pred_id in predecessors:
            siblings = graph.get_successors(pred_id)
            for sibling_id in siblings:
                if sibling_id != state_id and sibling_id in graph.nodes:
                    if not graph.nodes[sibling_id].is_pruned:
                        return sibling_id
        
        return None
    
    def _calculate_similarity(
        self, 
        state_id1: int, 
        state_id2: int,
        graph: TrajectoryGraph
    ) -> float:
        """Calculate structural similarity between two states."""
        node1 = graph.nodes[state_id1]
        node2 = graph.nodes[state_id2]
        
        # Start with action similarity
        if node1.action == node2.action:
            similarity = 0.5
        else:
            similarity = 0.0
        
        # Add hash similarity
        if node1.observation_hash == node2.observation_hash:
            similarity += 0.5
        
        # Check successor overlap
        succ1 = set(graph.get_successors(state_id1))
        succ2 = set(graph.get_successors(state_id2))
        
        if succ1 or succ2:
            jaccard = len(succ1 & succ2) / len(succ1 | succ2) if (succ1 | succ2) else 0
            similarity = 0.7 * similarity + 0.3 * jaccard
        
        return similarity
    
    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset_stats()
        self._observation_hashes.clear()


class CompositePruningStrategy(PruningStrategy):
    """Combines multiple strategies with priority-based evaluation.
    
    Evaluates states using multiple strategies and combines decisions
    based on priority. Can use different combination modes:
    - ANY: Prune if any strategy says prune
    - ALL: Prune only if all strategies say prune
    - PRIORITY: Use highest priority strategy that makes a decision
    """
    
    COMBINATION_ANY = "any"
    COMBINATION_ALL = "all"
    COMBINATION_PRIORITY = "priority"
    
    def __init__(
        self, 
        strategies: Optional[List[PruningStrategy]] = None,
        combination_mode: str = "priority",
        name: str = "Composite"
    ):
        """Initialize composite strategy.
        
        Args:
            strategies: List of strategies to combine.
            combination_mode: How to combine decisions ('any', 'all', 'priority').
            name: Name for this composite strategy.
        """
        super().__init__(name, priority=0)
        self.strategies = strategies or []
        self.combination_mode = combination_mode
        self._sort_strategies()
    
    def add_strategy(self, strategy: PruningStrategy) -> None:
        """Add a strategy to the composite."""
        self.strategies.append(strategy)
        self._sort_strategies()
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy by name."""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                del self.strategies[i]
                return True
        return False
    
    def _sort_strategies(self) -> None:
        """Sort strategies by priority (highest first)."""
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def should_prune(
        self, 
        state_id: int, 
        graph: TrajectoryGraph, 
        context: PruningContext
    ) -> PruningDecision:
        """Evaluate using combined strategies."""
        if not self.strategies:
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name=self.name,
                reason="No strategies configured",
                priority=self.priority
            )
        
        decisions = []
        
        if self.combination_mode == self.COMBINATION_PRIORITY:
            # Use highest priority strategy that makes a positive decision
            for strategy in self.strategies:
                decision = strategy.should_prune(state_id, graph, context)
                if decision.should_prune:
                    return PruningDecision(
                        state_id=state_id,
                        should_prune=True,
                        strategy_name=self.name,
                        reason=f"Priority decision from {strategy.name}: {decision.reason}",
                        priority=strategy.priority,
                        metadata={
                            "source_decision": decision,
                            "combination_mode": self.combination_mode
                        }
                    )
            
            # No strategy said to prune
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name=self.name,
                reason="No high-priority strategy recommended pruning",
                priority=self.priority
            )
        
        # ANY or ALL mode - collect all decisions
        for strategy in self.strategies:
            decision = strategy.should_prune(state_id, graph, context)
            decisions.append(decision)
        
        if self.combination_mode == self.COMBINATION_ANY:
            # Prune if any strategy says prune
            prune_decisions = [d for d in decisions if d.should_prune]
            if prune_decisions:
                highest = max(prune_decisions, key=lambda d: d.priority)
                return PruningDecision(
                    state_id=state_id,
                    should_prune=True,
                    strategy_name=self.name,
                    reason=f"ANY mode: {len(prune_decisions)} strategies recommended pruning",
                    priority=highest.priority,
                    metadata={
                        "all_decisions": decisions,
                        "combination_mode": self.combination_mode
                    }
                )
        
        elif self.combination_mode == self.COMBINATION_ALL:
            # Prune only if all strategies say prune
            if all(d.should_prune for d in decisions):
                highest = max(decisions, key=lambda d: d.priority)
                return PruningDecision(
                    state_id=state_id,
                    should_prune=True,
                    strategy_name=self.name,
                    reason="ALL mode: All strategies recommended pruning",
                    priority=highest.priority,
                    metadata={
                        "all_decisions": decisions,
                        "combination_mode": self.combination_mode
                    }
                )
        
        # Default: don't prune
        return PruningDecision(
            state_id=state_id,
            should_prune=False,
            strategy_name=self.name,
            reason=f"{self.combination_mode.upper()} mode: No consensus to prune",
            priority=self.priority,
            metadata={
                "all_decisions": decisions,
                "combination_mode": self.combination_mode
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all strategies."""
        stats = super().get_stats()
        stats["sub_strategies"] = {
            strategy.name: strategy.get_stats() 
            for strategy in self.strategies
        }
        return stats
    
    def reset_stats(self) -> None:
        """Reset all strategy statistics."""
        super().reset_stats()
        for strategy in self.strategies:
            strategy.reset_stats()


class PruningManager:
    """Manages pruning strategies and tracks pruning history.
    
    Provides:
    - Registration of multiple strategies
    - Evaluation of pruning decisions
    - History tracking for debugging/auditing
    - Execution-time pruning hooks
    """
    
    def __init__(self):
        """Initialize pruning manager."""
        self.strategies: Dict[str, PruningStrategy] = {}
        self.default_strategy: Optional[str] = None
        self._history: List[PruningDecision] = []
        self._pruned_states: Set[int] = set()
        self._hooks: List[Callable[[PruningDecision], None]] = []
        self._enabled: bool = True
    
    def register_strategy(
        self, 
        strategy: PruningStrategy, 
        set_as_default: bool = False
    ) -> None:
        """Register a pruning strategy.
        
        Args:
            strategy: The strategy to register.
            set_as_default: Whether to set as the default strategy.
        """
        self.strategies[strategy.name] = strategy
        if set_as_default or self.default_strategy is None:
            self.default_strategy = strategy.name
    
    def unregister_strategy(self, name: str) -> bool:
        """Unregister a strategy by name.
        
        Args:
            name: Name of the strategy to remove.
            
        Returns:
            True if strategy was found and removed.
        """
        if name in self.strategies:
            del self.strategies[name]
            if self.default_strategy == name:
                self.default_strategy = next(iter(self.strategies), None)
            return True
        return False
    
    def get_strategy(self, name: Optional[str] = None) -> PruningStrategy:
        """Get a strategy by name (or default).
        
        Args:
            name: Strategy name, or None for default.
            
        Returns:
            The requested strategy.
            
        Raises:
            ValueError: If strategy not found.
        """
        name = name or self.default_strategy
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found")
        return self.strategies[name]
    
    def evaluate(
        self, 
        state_id: int, 
        graph: TrajectoryGraph,
        context: Optional[PruningContext] = None,
        strategy_name: Optional[str] = None
    ) -> PruningDecision:
        """Evaluate whether a state should be pruned.
        
        Args:
            state_id: The state to evaluate.
            graph: The trajectory graph.
            context: Optional pruning context.
            strategy_name: Strategy to use (or default).
            
        Returns:
            The pruning decision.
        """
        if not self._enabled:
            return PruningDecision(
                state_id=state_id,
                should_prune=False,
                strategy_name="None",
                reason="Pruning is disabled",
                priority=0
            )
        
        context = context or PruningContext()
        strategy = self.get_strategy(strategy_name)
        
        decision = strategy.should_prune(state_id, graph, context)
        
        # Record in history
        self._history.append(decision)
        
        # Track pruned states
        if decision.should_prune:
            self._pruned_states.add(state_id)
        
        # Execute hooks
        for hook in self._hooks:
            hook(decision)
        
        return decision
    
    def prune_if_needed(
        self, 
        state_id: int, 
        graph: TrajectoryGraph,
        context: Optional[PruningContext] = None,
        strategy_name: Optional[str] = None
    ) -> bool:
        """Evaluate and execute pruning if recommended.
        
        Args:
            state_id: The state to evaluate.
            graph: The trajectory graph.
            context: Optional pruning context.
            strategy_name: Strategy to use (or default).
            
        Returns:
            True if state was pruned.
        """
        decision = self.evaluate(state_id, graph, context, strategy_name)
        
        if decision.should_prune and state_id in graph.nodes:
            graph.prune_branch(state_id)
            return True
        
        return False
    
    def add_hook(self, hook: Callable[[PruningDecision], None]) -> None:
        """Add a hook to be called on each pruning decision.
        
        Args:
            hook: Function to call with the pruning decision.
        """
        self._hooks.append(hook)
    
    def remove_hook(self, hook: Callable[[PruningDecision], None]) -> bool:
        """Remove a pruning hook.
        
        Args:
            hook: The hook function to remove.
            
        Returns:
            True if hook was found and removed.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)
            return True
        return False
    
    def get_history(
        self, 
        state_id: Optional[int] = None
    ) -> List[PruningDecision]:
        """Get pruning decision history.
        
        Args:
            state_id: Filter by state_id, or None for all.
            
        Returns:
            List of pruning decisions.
        """
        if state_id is not None:
            return [d for d in self._history if d.state_id == state_id]
        return self._history.copy()
    
    def get_pruned_states(self) -> Set[int]:
        """Get set of all pruned state IDs."""
        return self._pruned_states.copy()
    
    def clear_history(self) -> None:
        """Clear pruning history."""
        self._history.clear()
        self._pruned_states.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all strategies."""
        return {
            name: strategy.get_stats()
            for name, strategy in self.strategies.items()
        }
    
    def reset_stats(self) -> None:
        """Reset all strategy statistics."""
        for strategy in self.strategies.values():
            strategy.reset_stats()
    
    def enable(self) -> None:
        """Enable pruning."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable pruning."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if pruning is enabled."""
        return self._enabled
    
    def create_default_strategy(self) -> CompositePruningStrategy:
        """Create a default composite strategy with standard strategies."""
        composite = CompositePruningStrategy(
            strategies=[
                CycleEliminationStrategy(priority=10),
                DeadEndStrategy(priority=5),
                RedundancyStrategy(priority=3, epsilon=0.1)
            ],
            combination_mode="priority",
            name="DefaultComposite"
        )
        return composite
    
    def setup_default(self) -> None:
        """Set up default pruning configuration."""
        default = self.create_default_strategy()
        self.register_strategy(default, set_as_default=True)


# Utility functions for common pruning patterns

def prune_graph_batch(
    graph: TrajectoryGraph,
    manager: PruningManager,
    context: Optional[PruningContext] = None,
    strategy_name: Optional[str] = None
) -> List[int]:
    """Prune all eligible states in a graph.
    
    Args:
        graph: The trajectory graph.
        manager: The pruning manager.
        context: Optional pruning context.
        strategy_name: Strategy to use.
        
    Returns:
        List of state IDs that were pruned.
    """
    pruned = []
    context = context or PruningContext()
    
    # Evaluate all non-pruned states
    for state_id in list(graph.nodes.keys()):
        if not graph.nodes[state_id].is_pruned:
            decision = manager.evaluate(state_id, graph, context, strategy_name)
            if decision.should_prune:
                graph.prune_branch(state_id)
                pruned.append(state_id)
    
    return pruned


def get_pruning_recommendations(
    graph: TrajectoryGraph,
    manager: PruningManager,
    context: Optional[PruningContext] = None,
    strategy_name: Optional[str] = None
) -> List[PruningDecision]:
    """Get pruning recommendations without executing.
    
    Args:
        graph: The trajectory graph.
        manager: The pruning manager.
        context: Optional pruning context.
        strategy_name: Strategy to use.
        
    Returns:
        List of pruning decisions recommending pruning.
    """
    recommendations = []
    context = context or PruningContext()
    
    for state_id in graph.nodes:
        if not graph.nodes[state_id].is_pruned:
            decision = manager.evaluate(state_id, graph, context, strategy_name)
            if decision.should_prune:
                recommendations.append(decision)
    
    return recommendations
