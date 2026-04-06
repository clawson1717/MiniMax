"""
Pruned Ensemble Coordinator for capacity-weighted ensemble coordination.

Orchestrates the full pipeline: generate → prune → vote → scale.
Integrates all components for iterative capacity-weighted ensemble processing.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .capacity import CapacityEstimator, CapacityResult
from .trajectory import TrajectoryGraph, StepNode
from .pruner import TrajectoryPruner, PruningResult
from .uncertainty import UncertaintyEstimator, UncertaintyResult
from .allocator import ComputeAllocator, AllocationResult
from .voting import CapacityWeightedVoter, VoteResult as VotingVoteResult, DisagreementResult


@dataclass
class CoordinationResult:
    """
    Result of a full coordination run.
    
    Attributes:
        final_response: The final selected response after all iterations
        trajectory_graph: The trajectory graph containing all reasoning steps
        pruning_stats: Statistics from the pruning operations
        capacity_distribution: Distribution of capacity estimates across agents
        iterations: Number of coordination iterations performed
        converged: Whether the coordination converged (vs budget exhaustion)
        total_tokens: Total tokens used across all agents
        uncertainty_history: Uncertainty scores from each iteration
        metadata: Additional metadata about the coordination process
    """
    final_response: str
    trajectory_graph: TrajectoryGraph
    pruning_stats: List[PruningResult]
    capacity_distribution: Dict[str, float]
    iterations: int
    converged: bool
    total_tokens: int
    uncertainty_history: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_response": self.final_response,
            "iterations": self.iterations,
            "converged": self.converged,
            "total_tokens": self.total_tokens,
            "uncertainty_history": self.uncertainty_history,
            "pruning_stats": [p.to_dict() for p in self.pruning_stats],
            "capacity_distribution": self.capacity_distribution,
            "metadata": self.metadata
        }


class EnsembleCoordinator:
    """
    Coordinates the full capacity-weighted ensemble pipeline.
    
    Orchestrates: generate → prune → vote → scale loop until convergence
    or budget exhaustion. Integrates all components for iterative processing.
    
    The coordinator maintains state across iterations, building up a trajectory
    graph of reasoning steps, pruning unproductive paths, and allocating compute
    based on agent capacity and uncertainty.
    
    Example:
        >>> coordinator = EnsembleCoordinator(
        ...     agents=[agent1, agent2, agent3],
        ...     capacity_estimator=CapacityEstimator(),
        ...     max_iterations=5,
        ...     compute_budget=1000
        ... )
        >>> result = coordinator.coordinate("What is 2+2?")
        >>> print(result.final_response)
    """
    
    def __init__(
        self,
        agents: List[Any],
        capacity_estimator: CapacityEstimator,
        trajectory_pruner: Optional[TrajectoryPruner] = None,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        compute_allocator: Optional[ComputeAllocator] = None,
        voter: Optional[CapacityWeightedVoter] = None,
        max_iterations: int = 3,
        compute_budget: int = 1000,
        convergence_threshold: float = 0.8,
        min_agents: int = 1
    ):
        """
        Initialize the ensemble coordinator.
        
        Args:
            agents: List of agents (callables, or objects with generate/respond methods)
            capacity_estimator: CapacityEstimator instance for measuring agent capacity
            trajectory_pruner: Optional TrajectoryPruner for pruning trajectories
            uncertainty_estimator: Optional UncertaintyEstimator for measuring disagreement
            compute_allocator: Optional ComputeAllocator for test-time compute allocation
            voter: Optional CapacityWeightedVoter for capacity-weighted voting
            max_iterations: Maximum number of coordination iterations
            compute_budget: Total compute budget for the coordination
            convergence_threshold: Agreement score threshold for convergence (0.0-1.0)
            min_agents: Minimum number of agents required for coordination
            
        Raises:
            ValueError: If agents list is empty or parameters are invalid
        """
        if not agents:
            raise ValueError("agents list cannot be empty")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
        if compute_budget < 0:
            raise ValueError(f"compute_budget must be non-negative, got {compute_budget}")
        if not 0.0 <= convergence_threshold <= 1.0:
            raise ValueError(f"convergence_threshold must be between 0.0 and 1.0, got {convergence_threshold}")
        if min_agents < 1:
            raise ValueError(f"min_agents must be at least 1, got {min_agents}")
        
        self._agents: Dict[str, Any] = {}
        self.capacity_estimator = capacity_estimator
        
        # Initialize optional components with defaults
        self._pruner = trajectory_pruner or TrajectoryPruner()
        self._uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self._compute_allocator = compute_allocator or ComputeAllocator(total_budget=compute_budget)
        self._voter = voter or CapacityWeightedVoter()
        
        # Configuration
        self.max_iterations = max_iterations
        self.compute_budget = compute_budget
        self.convergence_threshold = convergence_threshold
        self.min_agents = min_agents
        
        # Register agents with IDs
        for i, agent in enumerate(agents):
            agent_id = self._get_agent_id(agent, i)
            self._agents[agent_id] = agent
        
        # State tracking
        self._trajectory_graph: Optional[TrajectoryGraph] = None
        self._iteration_count: int = 0
        self._total_tokens: int = 0
        self._uncertainty_history: List[float] = []
        self._pruning_stats: List[PruningResult] = []
        self._capacity_distribution: Dict[str, float] = {}
        self._last_vote_result: Optional[VotingVoteResult] = None
        self._last_allocation: Optional[Dict[str, int]] = None
        
        # Compute tracking
        self._compute_used: int = 0
    
    def _get_agent_id(self, agent: Any, index: int) -> str:
        """
        Get or generate an agent ID.
        
        Args:
            agent: The agent object
            index: Index in the agents list (used for default ID)
            
        Returns:
            Agent ID string
        """
        if hasattr(agent, 'id') and agent.id:
            return str(agent.id)
        if hasattr(agent, 'agent_id') and agent.agent_id:
            return str(agent.agent_id)
        if hasattr(agent, 'name') and agent.name:
            return str(agent.name)
        return f"agent_{index}"
    
    def _call_agent(self, agent: Any, task: str) -> Tuple[str, int]:
        """
        Call an agent with a task and return response with token estimate.
        
        Args:
            agent: The agent object
            task: The task/prompt string
            
        Returns:
            Tuple of (response, tokens_used)
        """
        try:
            if hasattr(agent, 'generate') and callable(agent.generate):
                result = agent.generate(task)
                response = str(result) if result is not None else ""
            elif hasattr(agent, 'respond') and callable(agent.respond):
                result = agent.respond(task)
                response = str(result) if result is not None else ""
            elif callable(agent):
                result = agent(task)
                response = str(result) if result is not None else ""
            else:
                response = str(agent)
            
            # Estimate tokens (rough: len/4 for English)
            tokens = len(response.split()) if response else 0
            
            return response, tokens
        except Exception as e:
            # Return empty on failure
            return "", 0
    
    def coordinate(self, task: str, context: Optional[Dict[str, Any]] = None) -> CoordinationResult:
        """
        Run the full generate → prune → vote → scale coordination loop.
        
        This is the main entry point for using the coordinator. It orchestrates
        all components in an iterative loop until convergence or budget exhaustion.
        
        Args:
            task: The task/prompt string to coordinate
            context: Optional context dictionary for capacity estimation
            
        Returns:
            CoordinationResult with the final response and metadata
            
        Raises:
            RuntimeError: If no agents respond successfully
        """
        context = context or {}
        task_context = {"query": task, **context}
        
        # Reset state for new coordination
        self._reset_state()
        
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and self.should_continue():
            iteration += 1
            self._iteration_count = iteration
            
            # 1. GENERATE: Get responses from all agents
            agent_responses, tokens_used = self._generate_responses(task)
            
            if not agent_responses:
                raise RuntimeError("No agents produced valid responses")
            
            self._total_tokens += tokens_used
            
            # 2. ESTIMATE CAPACITY: Measure agent information capacity
            capacities = self._estimate_capacities(agent_responses, task_context)
            
            # 3. UPDATE TRAJECTORY: Add steps to the trajectory graph
            self._update_trajectory(agent_responses, capacities)
            
            # 4. PRUNE: Remove unproductive reasoning paths
            pruned_graph, pruning_result = self._pruner.prune(
                self._trajectory_graph, 
                strategy="all"
            )
            self._trajectory_graph = pruned_graph
            self._pruning_stats.append(pruning_result)
            
            # 5. VOTE: Conduct capacity-weighted voting
            responses_list = list(agent_responses.values())
            capacities_list = [capacities.get(aid, 1.0) for aid in agent_responses.keys()]
            vote_result = self._voter.weighted_vote(responses_list, capacities_list)
            self._last_vote_result = vote_result
            
            # 6. ESTIMATE UNCERTAINTY: Measure disagreement
            uncertainty_result = self._uncertainty_estimator.estimate_uncertainty(responses_list)
            self._uncertainty_history.append(uncertainty_result.uncertainty_score)
            
            # 7. ALLOCATION: Decide compute allocation for next iteration
            uncertainties = {aid: uncertainty_result.uncertainty_score for aid in agent_responses}
            self._last_allocation = self._compute_allocator.allocate(
                agents=list(agent_responses.keys()),
                capacities=capacities,
                uncertainties=uncertainties
            )
            
            # Check for convergence
            if vote_result.disagreement_score < (1.0 - self.convergence_threshold):
                converged = True
                break
            
            # Use compute budget
            self._compute_used += len(agent_responses) * 10  # Rough estimate
        
        # Build final response from last vote result or default
        if self._last_vote_result:
            final_response = self._last_vote_result.winning_response
        else:
            # Fallback: run one generation if no iterations happened
            agent_responses, _ = self._generate_responses(task)
            if agent_responses:
                final_response = list(agent_responses.values())[0]
            else:
                final_response = ""
        
        return CoordinationResult(
            final_response=final_response,
            trajectory_graph=self._trajectory_graph,
            pruning_stats=self._pruning_stats,
            capacity_distribution=self._capacity_distribution,
            iterations=self._iteration_count,
            converged=converged,
            total_tokens=self._total_tokens,
            uncertainty_history=self._uncertainty_history,
            metadata={
                "max_iterations": self.max_iterations,
                "compute_budget": self.compute_budget,
                "num_agents": len(self._agents)
            }
        )
    
    def _reset_state(self) -> None:
        """Reset coordinator state for a new coordination run."""
        self._trajectory_graph = TrajectoryGraph()
        self._iteration_count = 0
        self._total_tokens = 0
        self._uncertainty_history = []
        self._pruning_stats = []
        self._capacity_distribution = {}
        self._last_vote_result = None
        self._last_allocation = None
        self._compute_used = 0
        self._compute_allocator.reset()
    
    def _generate_responses(self, task: str) -> Tuple[Dict[str, str], int]:
        """
        Generate responses from all agents.
        
        Args:
            task: The task/prompt string
            
        Returns:
            Tuple of (agent_responses dict, total_tokens_used)
        """
        agent_responses: Dict[str, str] = {}
        total_tokens = 0
        
        for agent_id, agent in self._agents.items():
            response, tokens = self._call_agent(agent, task)
            if response.strip():  # Only add non-empty responses
                agent_responses[agent_id] = response
                total_tokens += tokens
        
        return agent_responses, total_tokens
    
    def _estimate_capacities(
        self,
        agent_responses: Dict[str, str],
        task_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate capacity for each agent.
        
        Args:
            agent_responses: Dictionary of agent_id -> response
            task_context: Task context for capacity estimation
            
        Returns:
            Dictionary of agent_id -> capacity_bits
        """
        capacities: Dict[str, float] = {}
        
        for agent_id, agent in self._agents.items():
            if agent_id in agent_responses:
                try:
                    result = self.capacity_estimator.estimate_capacity(
                        agent, task_context
                    )
                    capacities[agent_id] = result.capacity_bits
                except Exception:
                    # Default capacity if estimation fails
                    capacities[agent_id] = 1.0
            else:
                capacities[agent_id] = 0.0
        
        self._capacity_distribution = capacities
        return capacities
    
    def _update_trajectory(
        self,
        agent_responses: Dict[str, str],
        capacities: Dict[str, float]
    ) -> None:
        """
        Update the trajectory graph with new reasoning steps.
        
        Args:
            agent_responses: Dictionary of agent_id -> response
            capacities: Dictionary of agent_id -> capacity
        """
        if self._trajectory_graph is None:
            self._trajectory_graph = TrajectoryGraph()
        
        # Add a node for each agent's response
        parent_id = None
        for agent_id, response in agent_responses.items():
            tokens = len(response.split()) if response else 0
            confidence = min(1.0, capacities.get(agent_id, 1.0) / 10.0)
            
            step_id = self._trajectory_graph.add_step(
                content=response,
                agent_id=agent_id,
                tokens=tokens,
                confidence=confidence,
                parent_id=parent_id,
                metadata={
                    "capacity": capacities.get(agent_id, 0.0),
                    "iteration": self._iteration_count
                }
            )
            parent_id = step_id
    
    def should_continue(self) -> bool:
        """
        Decide whether to continue allocating more compute.
        
        This method checks:
        1. If the compute budget has been exhausted
        2. If the maximum number of iterations has been reached
        3. If the current iteration count is below max_iterations
        4. If there's remaining compute budget
        
        Returns:
            True if more compute should be allocated, False otherwise
        """
        # Check iteration limit
        if self._iteration_count >= self.max_iterations:
            return False
        
        # Check compute budget
        if self._compute_used >= self.compute_budget:
            return False
        
        # Check if we have agents available
        if len(self._agents) < self.min_agents:
            return False
        
        return True
    
    def should_scale(self, uncertainty: Optional[float] = None) -> bool:
        """
        Determine if compute should be scaled up.
        
        Uses the uncertainty estimator to decide if more agents should be invoked.
        If uncertainty is not provided, uses the most recent uncertainty estimate.
        
        Args:
            uncertainty: Optional uncertainty score to use instead of latest estimate
            
        Returns:
            True if scaling up is recommended
        """
        if uncertainty is None:
            if not self._uncertainty_history:
                return True  # Default to scaling if no history
            uncertainty = self._uncertainty_history[-1]
        
        return self._uncertainty_estimator.should_scale(uncertainty)
    
    def get_scale_recommendation(
        self,
        current_agents: Optional[int] = None,
        max_agents: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a recommendation for compute scaling.
        
        Args:
            current_agents: Current number of agents (defaults to registered count)
            max_agents: Maximum agents available (defaults to registered count)
            
        Returns:
            Dictionary with scaling recommendation
        """
        if current_agents is None:
            current_agents = len(self._agents)
        if max_agents is None:
            max_agents = len(self._agents)
        
        latest_uncertainty = self._uncertainty_history[-1] if self._uncertainty_history else 0.5
        
        return self._uncertainty_estimator.get_scale_recommendation(
            uncertainty=latest_uncertainty,
            current_agents=current_agents,
            max_agents=max_agents
        )
    
    @property
    def trajectory_graph(self) -> Optional[TrajectoryGraph]:
        """Get the current trajectory graph."""
        return self._trajectory_graph
    
    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count
    
    @property
    def total_tokens(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens
    
    @property
    def uncertainty_history(self) -> List[float]:
        """Get the history of uncertainty scores."""
        return list(self._uncertainty_history)
    
    @property
    def pruning_stats(self) -> List[PruningResult]:
        """Get pruning statistics from all iterations."""
        return list(self._pruning_stats)
    
    @property
    def capacity_distribution(self) -> Dict[str, float]:
        """Get the current capacity distribution."""
        return dict(self._capacity_distribution)
    
    @property
    def last_vote_result(self) -> Optional[VotingVoteResult]:
        """Get the result from the last vote."""
        return self._last_vote_result
    
    @property
    def agent_ids(self) -> List[str]:
        """Get list of registered agent IDs."""
        return list(self._agents.keys())
    
    @property
    def num_agents(self) -> int:
        """Get number of registered agents."""
        return len(self._agents)
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of pruning statistics across all iterations.
        
        Returns:
            Dictionary with aggregated pruning statistics
        """
        if not self._pruning_stats:
            return {
                "total_iterations": 0,
                "total_nodes_removed": 0,
                "total_cycles_found": 0,
                "avg_reduction_ratio": 0.0
            }
        
        total_removed = sum(s.nodes_removed for s in self._pruning_stats)
        total_cycles = sum(s.cycles_found for s in self._pruning_stats)
        avg_reduction = sum(s.reduction_ratio for s in self._pruning_stats) / len(self._pruning_stats)
        
        return {
            "total_iterations": len(self._pruning_stats),
            "total_nodes_removed": total_removed,
            "total_cycles_found": total_cycles,
            "avg_reduction_ratio": avg_reduction
        }
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the coordination process.
        
        Returns:
            Dictionary with coordination summary
        """
        return {
            "num_agents": len(self._agents),
            "iteration_count": self._iteration_count,
            "total_tokens": self._total_tokens,
            "compute_used": self._compute_used,
            "compute_budget": self.compute_budget,
            "uncertainty_history": self.uncertainty_history,
            "pruning_summary": self.get_pruning_summary(),
            "capacity_distribution": self.capacity_distribution,
            "last_vote_agreement": (
                1.0 - self._last_vote_result.disagreement_score
                if self._last_vote_result else None
            ),
            "should_continue": self.should_continue()
        }
    
    def add_agent(self, agent: Any, agent_id: Optional[str] = None) -> str:
        """
        Add an agent to the coordinator.
        
        Args:
            agent: Agent object to add
            agent_id: Optional agent ID (auto-generated if not provided)
            
        Returns:
            The agent ID (assigned or generated)
        """
        if agent_id is None:
            agent_id = self._get_agent_id(agent, len(self._agents))
        
        self._agents[agent_id] = agent
        return agent_id
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the coordinator.
        
        Args:
            agent_id: ID of the agent to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent ID to look up
            
        Returns:
            Agent object or None if not found
        """
        return self._agents.get(agent_id)
    
    def reset(self) -> None:
        """
        Reset the coordinator to initial state.
        
        Clears all iteration state, trajectory graph, and statistics.
        Does not remove registered agents.
        """
        self._reset_state()
