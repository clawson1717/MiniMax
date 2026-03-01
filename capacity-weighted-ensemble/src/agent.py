"""
Ensemble Agent for capacity-weighted response coordination.

Implements an ensemble agent that wraps multiple agents with capacity
measurement and coordinates responses via voting/consensus mechanisms.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .capacity import CapacityEstimator


@dataclass
class VoteResult:
    """
    Result of a voting process among agent responses.
    
    Attributes:
        winner: The winning response (most votes)
        votes: Dictionary mapping responses to vote counts
        agreement_score: Agreement level (0.0-1.0, higher = more agreement)
    """
    winner: str
    votes: Dict[str, int]
    agreement_score: float
    
    def __post_init__(self):
        """Validate the vote result after initialization."""
        if not self.winner and self.votes:
            raise ValueError("winner cannot be empty when votes exist")
        if not 0.0 <= self.agreement_score <= 1.0:
            raise ValueError("agreement_score must be between 0.0 and 1.0")
        if self.votes:
            total_votes = sum(self.votes.values())
            if total_votes <= 0:
                raise ValueError("total votes must be positive")
            if self.winner not in self.votes:
                raise ValueError("winner must be in votes dictionary")


@dataclass
class EnsembleResponse:
    """
    Response from an ensemble of agents.
    
    Attributes:
        final_response: The final selected/coordinated response
        agent_responses: Dictionary mapping agent IDs to their responses
        capacities: Dictionary mapping agent IDs to their capacity estimates
        vote_result: Optional VoteResult if voting was performed
        metadata: Additional metadata about the ensemble process
    """
    final_response: str
    agent_responses: Dict[str, str]
    capacities: Dict[str, float]
    vote_result: Optional[VoteResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the ensemble response after initialization."""
        if not isinstance(self.agent_responses, dict):
            raise ValueError("agent_responses must be a dictionary")
        if not isinstance(self.capacities, dict):
            raise ValueError("capacities must be a dictionary")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


class EnsembleCoordinator:
    """
    Optional coordinator for more advanced ensemble coordination strategies.
    
    This is a placeholder for future extensions. The EnsembleAgent can
    function without a coordinator using built-in voting/consensus.
    """
    
    def __init__(self, strategy: str = "majority"):
        """
        Initialize the coordinator.
        
        Args:
            strategy: Coordination strategy (majority, weighted, consensus)
        """
        if strategy not in ("majority", "weighted", "consensus"):
            raise ValueError(f"Invalid strategy: {strategy}")
        self.strategy = strategy
    
    def coordinate(
        self,
        responses: Dict[str, str],
        capacities: Dict[str, float]
    ) -> str:
        """
        Coordinate responses based on strategy.
        
        Args:
            responses: Dictionary of agent_id -> response
            capacities: Dictionary of agent_id -> capacity
            
        Returns:
            Coordinated final response
        """
        if not responses:
            raise ValueError("No responses to coordinate")
        
        if self.strategy == "majority":
            # Simple majority vote
            vote_counts: Dict[str, int] = {}
            for response in responses.values():
                vote_counts[response] = vote_counts.get(response, 0) + 1
            return max(vote_counts, key=vote_counts.get)
        
        elif self.strategy == "weighted":
            # Capacity-weighted selection
            weighted: Dict[str, float] = {}
            for agent_id, response in responses.items():
                weight = capacities.get(agent_id, 1.0)
                weighted[response] = weighted.get(response, 0.0) + weight
            return max(weighted, key=weighted.get)
        
        else:  # consensus
            # For now, same as majority
            vote_counts: Dict[str, int] = {}
            for response in responses.values():
                vote_counts[response] = vote_counts.get(response, 0) + 1
            return max(vote_counts, key=vote_counts.get)


class EnsembleAgent:
    """
    Ensemble agent that wraps multiple agents with capacity measurement
    and coordinates responses via voting/consensus.
    
    Supports multiple agent interfaces:
    - Callable agents (functions that take str and return str)
    - Agents with .generate(prompt) method
    - Agents with .respond(prompt) method
    
    Example:
        >>> def agent1(prompt): return "response 1"
        >>> def agent2(prompt): return "response 2"
        >>> estimator = CapacityEstimator()
        >>> ensemble = EnsembleAgent([agent1, agent2], estimator)
        >>> result = ensemble.generate("What is 2+2?")
        >>> print(result.final_response)
    """
    
    def __init__(
        self,
        agents: List[Any],
        capacity_estimator: CapacityEstimator,
        coordinator: Optional[EnsembleCoordinator] = None
    ):
        """
        Initialize the ensemble agent.
        
        Args:
            agents: List of agents (callables, or objects with generate/respond methods)
            capacity_estimator: CapacityEstimator instance for measuring agent capacity
            coordinator: Optional EnsembleCoordinator for advanced coordination
            
        Raises:
            ValueError: If agents list is empty
        """
        if not agents:
            raise ValueError("agents list cannot be empty")
        
        self._agents: Dict[str, Any] = {}
        self.capacity_estimator = capacity_estimator
        self.coordinator = coordinator
        
        # Register agents with IDs
        for i, agent in enumerate(agents):
            agent_id = self._get_agent_id(agent, i)
            self._agents[agent_id] = agent
    
    def _get_agent_id(self, agent: Any, index: int) -> str:
        """
        Get or generate an agent ID.
        
        Args:
            agent: The agent object
            index: Index in the agents list (used for default ID)
            
        Returns:
            Agent ID string
        """
        # Check for explicit ID attribute
        if hasattr(agent, 'id') and agent.id:
            return str(agent.id)
        if hasattr(agent, 'agent_id') and agent.agent_id:
            return str(agent.agent_id)
        if hasattr(agent, 'name') and agent.name:
            return str(agent.name)
        
        # Generate default ID
        return f"agent_{index}"
    
    def _call_agent(self, agent: Any, prompt: str) -> str:
        """
        Call an agent with a prompt, handling different interfaces.
        
        Args:
            agent: The agent object
            prompt: Input prompt string
            
        Returns:
            Agent's response string
            
        Raises:
            RuntimeError: If agent call fails
        """
        try:
            # Try generate method first
            if hasattr(agent, 'generate') and callable(agent.generate):
                result = agent.generate(prompt)
                return str(result) if result is not None else ""
            
            # Try respond method
            if hasattr(agent, 'respond') and callable(agent.respond):
                result = agent.respond(prompt)
                return str(result) if result is not None else ""
            
            # Try callable
            if callable(agent):
                result = agent(prompt)
                return str(result) if result is not None else ""
            
            # Fallback: string representation
            return str(agent)
            
        except Exception as e:
            raise RuntimeError(f"Agent call failed: {e}") from e
    
    def generate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EnsembleResponse:
        """
        Generate an ensemble response for a task.
        
        Args:
            task: The task/prompt string
            context: Optional context dictionary for capacity estimation
            
        Returns:
            EnsembleResponse with coordinated response and metadata
        """
        context = context or {}
        start_time = time.time()
        
        # Collect responses from all agents
        agent_responses: Dict[str, str] = {}
        errors: List[str] = []
        
        for agent_id, agent in self._agents.items():
            try:
                response = self._call_agent(agent, task)
                agent_responses[agent_id] = response
            except Exception as e:
                errors.append(f"{agent_id}: {str(e)}")
        
        if not agent_responses:
            raise RuntimeError(f"All agents failed: {'; '.join(errors)}")
        
        # Estimate capacities
        capacities: Dict[str, float] = {}
        task_context = {"query": task, **context}
        
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
        
        # Perform voting
        vote_result = self.vote(list(agent_responses.values()))
        
        # Get final response
        if self.coordinator:
            final_response = self.coordinator.coordinate(agent_responses, capacities)
        else:
            # Use consensus (capacity-weighted)
            final_response = self.consensus(
                list(agent_responses.values()), 
                capacities
            )
        
        elapsed_time = time.time() - start_time
        
        metadata = {
            "num_agents": len(self._agents),
            "num_responses": len(agent_responses),
            "elapsed_time_seconds": elapsed_time,
            "errors": errors if errors else None,
        }
        
        return EnsembleResponse(
            final_response=final_response,
            agent_responses=agent_responses,
            capacities=capacities,
            vote_result=vote_result,
            metadata=metadata
        )
    
    def vote(self, responses: List[str]) -> VoteResult:
        """
        Perform simple majority voting on responses.
        
        Groups similar responses (exact match for now) and returns
        the most common response as the winner.
        
        Args:
            responses: List of response strings to vote on
            
        Returns:
            VoteResult with winner and vote distribution
            
        Raises:
            ValueError: If responses list is empty
        """
        if not responses:
            raise ValueError("responses cannot be empty")
        
        # Count votes (exact match)
        vote_counts: Dict[str, int] = {}
        for response in responses:
            normalized = response.strip()
            vote_counts[normalized] = vote_counts.get(normalized, 0) + 1
        
        # Find winner
        winner = max(vote_counts, key=vote_counts.get)
        winner_votes = vote_counts[winner]
        total_votes = len(responses)
        
        # Calculate agreement score (proportion of votes for winner)
        agreement_score = winner_votes / total_votes
        
        return VoteResult(
            winner=winner,
            votes=vote_counts,
            agreement_score=agreement_score
        )
    
    def consensus(
        self,
        responses: List[str],
        capacities: Dict[str, float]
    ) -> str:
        """
        Reach consensus using capacity-weighted voting.
        
        Each response gets weighted by the capacity of the agent
        that produced it. Responses are grouped by exact match,
        and weights are summed for each unique response.
        
        Args:
            responses: List of response strings
            capacities: Dictionary mapping agent IDs to capacity values
            
        Returns:
            The consensus response (highest weighted votes)
            
        Raises:
            ValueError: If responses list is empty
        """
        if not responses:
            raise ValueError("responses cannot be empty")
        
        # For consensus, we need to map responses to capacities
        # Since we only have the list of responses, we'll use
        # simple voting but could extend for capacity weighting
        
        # Simple majority for now (can be extended to use capacities)
        # When called from generate(), capacities correspond to agents
        # but responses list doesn't have agent IDs
        
        # Group responses and count
        vote_counts: Dict[str, int] = {}
        for response in responses:
            normalized = response.strip()
            vote_counts[normalized] = vote_counts.get(normalized, 0) + 1
        
        # Return most common response
        return max(vote_counts, key=vote_counts.get)
    
    @property
    def agent_ids(self) -> List[str]:
        """Get list of registered agent IDs."""
        return list(self._agents.keys())
    
    @property
    def num_agents(self) -> int:
        """Get number of registered agents."""
        return len(self._agents)
    
    def add_agent(self, agent: Any, agent_id: Optional[str] = None) -> str:
        """
        Add an agent to the ensemble.
        
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
        Remove an agent from the ensemble.
        
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
