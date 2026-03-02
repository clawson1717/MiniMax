"""
Iterative Refinement Loop for Multi-Agent Systems

Implements a refinement loop where agents iteratively improve their responses
based on peer feedback. Higher capacity agents have more influence on the
refinement process through capacity-weighted feedback aggregation.
"""

import random
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

from src.environment import (
    MultiAgentEnvironment,
    MessageType,
    Message,
    Task,
    AgentRole,
)
from src.routing import (
    MessageRouter,
    RouteMode,
)


class RefinementPhase(Enum):
    """Phases in the refinement loop."""
    INITIAL = "initial"
    COLLECT_RESPONSES = "collect_responses"
    DISTRIBUTE_FEEDBACK = "distribute_feedback"
    REFINE = "refine"
    AGGREGATE = "aggregate"
    COMPLETE = "complete"


@dataclass
class AgentResponse:
    """A response from an agent in the refinement loop."""
    
    agent_id: str
    content: Any
    round_num: int
    confidence: float = 0.5
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PeerFeedback:
    """Feedback from one agent to another."""
    
    sender_id: str
    receiver_id: str
    feedback: Any
    weight: float = 1.0  # Capacity-weighted influence
    round_num: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class RefinementRound:
    """State for a single refinement round."""
    
    round_num: int
    phase: RefinementPhase
    responses: Dict[str, AgentResponse] = field(default_factory=dict)
    feedback: Dict[str, List[PeerFeedback]] = field(default_factory=dict)
    aggregated_feedback: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Any] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class RefinementStats:
    """Statistics for the refinement process."""
    
    total_rounds: int = 0
    total_responses: int = 0
    total_feedback: int = 0
    avg_response_confidence: float = 0.0
    convergence_round: Optional[int] = None
    capacity_influence_total: Dict[str, float] = field(default_factory=dict)
    
    def update_from_round(self, round_data: RefinementRound) -> None:
        """Update stats from a completed round."""
        self.total_rounds = round_data.round_num + 1
        self.total_responses += len(round_data.responses)
        self.total_feedback += sum(len(fb) for fb in round_data.feedback.values())
        
        # Track capacity influence
        for agent_id, feedbacks in round_data.feedback.items():
            for fb in feedbacks:
                if fb.sender_id not in self.capacity_influence_total:
                    self.capacity_influence_total[fb.sender_id] = 0.0
                self.capacity_influence_total[fb.sender_id] += fb.weight
        
        # Update average confidence
        if round_data.responses:
            self.avg_response_confidence = sum(
                r.confidence for r in round_data.responses.values()
            ) / len(round_data.responses)


class RefinementLoop:
    """
    Iterative refinement loop for multi-agent collaboration.
    
    Agents refine their responses based on peer feedback, with higher
    capacity agents having more influence on the refinement process.
    
    The loop proceeds through phases:
    1. INITIAL: Task is distributed to agents
    2. COLLECT_RESPONSES: Agents submit initial responses
    3. DISTRIBUTE_FEEDBACK: Feedback is shared between agents
    4. REFINE: Agents refine their responses based on feedback
    5. AGGREGATE: Final results are aggregated
    6. COMPLETE: Refinement is finished
    """
    
    def __init__(
        self,
        environment: MultiAgentEnvironment,
        router: Optional[MessageRouter] = None,
        max_rounds: int = 5,
        convergence_threshold: float = 0.95,
        min_participants: int = 2,
        feedback_distributor: Optional[Callable[[Dict[str, AgentResponse]], Dict[str, List[PeerFeedback]]]] = None,
        response_aggregator: Optional[Callable[[Dict[str, AgentResponse]], Any]] = None,
    ):
        """
        Initialize the refinement loop.
        
        Args:
            environment: Multi-agent environment
            router: Message router (created if None)
            max_rounds: Maximum refinement rounds
            convergence_threshold: Confidence threshold for early stopping
            min_participants: Minimum agents required for refinement
            feedback_distributor: Custom feedback distribution function
            response_aggregator: Custom response aggregation function
        """
        self.environment = environment
        self.router = router or MessageRouter(
            environment=environment,
            default_mode=RouteMode.CAPACITY_WEIGHTED,
        )
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.min_participants = min_participants
        self.feedback_distributor = feedback_distributor
        self.response_aggregator = response_aggregator
        
        # State tracking
        self.current_round: Optional[RefinementRound] = None
        self.rounds_history: List[RefinementRound] = []
        self.stats = RefinementStats()
        self._task: Optional[Task] = None
        self._participant_ids: List[str] = []
        self._is_running = False
        
        # Callbacks
        self.on_round_start: Optional[Callable[[int], None]] = None
        self.on_round_complete: Optional[Callable[[RefinementRound], None]] = None
        self.on_refinement_complete: Optional[Callable[[Any], None]] = None
    
    def initialize(self, task: Task, participant_ids: Optional[List[str]] = None) -> None:
        """
        Initialize the refinement loop with a task.
        
        Args:
            task: Task to refine
            participant_ids: Specific agents to include (all if None)
        """
        self._task = task
        self._is_running = True
        
        # Determine participants
        if participant_ids:
            self._participant_ids = [
                pid for pid in participant_ids 
                if pid in self.environment.agents
            ]
        else:
            self._participant_ids = list(self.environment.agents.keys())
        
        # Ensure minimum participants
        if len(self._participant_ids) < self.min_participants:
            raise ValueError(
                f"Need at least {self.min_participants} participants, "
                f"got {len(self._participant_ids)}"
            )
        
        # Initialize first round
        self.current_round = RefinementRound(
            round_num=0,
            phase=RefinementPhase.INITIAL,
        )
        self.rounds_history = []
        self.stats = RefinementStats()
    
    def get_capacity_weight(self, agent_id: str) -> float:
        """
        Get the capacity-based weight for an agent.
        
        Higher capacity agents have more influence.
        
        Args:
            agent_id: Agent to get weight for
            
        Returns:
            Capacity weight (normalized across all participants)
        """
        if agent_id not in self._participant_ids:
            return 0.0
        
        if agent_id not in self.environment.agent_states:
            return 0.0
        
        agent_capacity = self.environment.agent_states[agent_id].capacity
        
        # Sum capacities of all participants
        total_capacity = sum(
            self.environment.agent_states[pid].capacity
            for pid in self._participant_ids
            if pid in self.environment.agent_states
        )
        
        if total_capacity == 0:
            return 1.0 / len(self._participant_ids)
        
        return agent_capacity / total_capacity
    
    def submit_response(
        self,
        agent_id: str,
        content: Any,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentResponse]:
        """
        Submit a response from an agent.
        
        Args:
            agent_id: Agent submitting the response
            content: Response content
            confidence: Agent's confidence in their response
            metadata: Optional metadata
            
        Returns:
            AgentResponse object if successful
        """
        if not self._is_running or not self.current_round:
            return None
        
        if agent_id not in self._participant_ids:
            return None
        
        response = AgentResponse(
            agent_id=agent_id,
            content=content,
            round_num=self.current_round.round_num,
            confidence=confidence,
            metadata=metadata or {},
        )
        
        self.current_round.responses[agent_id] = response
        return response
    
    def provide_feedback(
        self,
        sender_id: str,
        receiver_id: str,
        feedback: Any,
    ) -> Optional[PeerFeedback]:
        """
        Provide feedback from one agent to another.
        
        The feedback weight is determined by the sender's capacity.
        
        Args:
            sender_id: Agent providing feedback
            receiver_id: Agent receiving feedback
            feedback: Feedback content
            
        Returns:
            PeerFeedback object if successful
        """
        if not self._is_running or not self.current_round:
            return None
        
        if sender_id not in self._participant_ids:
            return None
        
        if receiver_id not in self._participant_ids:
            return None
        
        # Capacity-weighted influence
        weight = self.get_capacity_weight(sender_id)
        
        peer_feedback = PeerFeedback(
            sender_id=sender_id,
            receiver_id=receiver_id,
            feedback=feedback,
            weight=weight,
            round_num=self.current_round.round_num,
        )
        
        if receiver_id not in self.current_round.feedback:
            self.current_round.feedback[receiver_id] = []
        
        self.current_round.feedback[receiver_id].append(peer_feedback)
        return peer_feedback
    
    def collect_all_responses(self) -> Dict[str, AgentResponse]:
        """
        Get all responses from the current round.
        
        Returns:
            Dictionary of agent_id -> AgentResponse
        """
        if not self.current_round:
            return {}
        return dict(self.current_round.responses)
    
    def get_feedback_for_agent(self, agent_id: str) -> List[PeerFeedback]:
        """
        Get all feedback addressed to an agent.
        
        Args:
            agent_id: Agent to get feedback for
            
        Returns:
            List of PeerFeedback objects
        """
        if not self.current_round:
            return []
        return self.current_round.feedback.get(agent_id, [])
    
    def aggregate_feedback(self, agent_id: str) -> Dict[str, Any]:
        """
        Aggregate all feedback for an agent, weighted by capacity.
        
        Higher capacity agents' feedback has more weight.
        
        Args:
            agent_id: Agent to aggregate feedback for
            
        Returns:
            Aggregated feedback dictionary
        """
        feedbacks = self.get_feedback_for_agent(agent_id)
        
        if not feedbacks:
            return {}
        
        # Weight feedback by capacity
        weighted_feedback = []
        total_weight = 0.0
        
        for fb in feedbacks:
            weighted_feedback.append({
                "sender_id": fb.sender_id,
                "feedback": fb.feedback,
                "weight": fb.weight,
            })
            total_weight += fb.weight
        
        # Normalize weights
        if total_weight > 0:
            for wf in weighted_feedback:
                wf["normalized_weight"] = wf["weight"] / total_weight
        
        # Store aggregated feedback
        self.current_round.aggregated_feedback[agent_id] = {
            "weighted_feedback": weighted_feedback,
            "total_weight": total_weight,
            "feedback_count": len(feedbacks),
        }
        
        return self.current_round.aggregated_feedback[agent_id]
    
    def distribute_feedback_via_router(self) -> List[Message]:
        """
        Distribute feedback to agents using the message router.
        
        Uses capacity-weighted routing to deliver feedback.
        
        Returns:
            List of messages sent
        """
        messages = []
        
        for agent_id in self._participant_ids:
            feedback = self.get_feedback_for_agent(agent_id)
            if feedback:
                aggregated = self.aggregate_feedback(agent_id)
                
                # Send feedback via router
                msg = self.environment.send_message(
                    sender_id="refinement_loop",
                    receiver_id=agent_id,
                    message_type=MessageType.FEEDBACK,
                    content=aggregated,
                    metadata={"round": self.current_round.round_num},
                )
                messages.append(msg)
        
        return messages
    
    def check_convergence(self) -> bool:
        """
        Check if refinement has converged.
        
        Returns:
            True if all agents have high confidence or max rounds reached
        """
        if not self.current_round:
            return False
        
        # Check if max rounds reached
        if self.current_round.round_num >= self.max_rounds - 1:
            return True
        
        # Check if all participants have responded with high confidence
        responses = self.current_round.responses
        
        if len(responses) < len(self._participant_ids):
            return False
        
        # Check average confidence
        avg_confidence = sum(r.confidence for r in responses.values()) / len(responses)
        
        if avg_confidence >= self.convergence_threshold:
            self.stats.convergence_round = self.current_round.round_num
            return True
        
        return False
    
    def advance_round(self) -> RefinementRound:
        """
        Advance to the next refinement round.
        
        Returns:
            The new round object
        """
        if not self._is_running:
            raise RuntimeError("Refinement loop not initialized")
        
        # Complete current round
        if self.current_round:
            self.current_round.phase = RefinementPhase.COMPLETE
            self.stats.update_from_round(self.current_round)
            self.rounds_history.append(self.current_round)
            
            if self.on_round_complete:
                self.on_round_complete(self.current_round)
        
        # Check for convergence
        if self.check_convergence():
            return self.finalize()
        
        # Start new round
        new_round = RefinementRound(
            round_num=len(self.rounds_history),
            phase=RefinementPhase.COLLECT_RESPONSES,
        )
        self.current_round = new_round
        
        if self.on_round_start:
            self.on_round_start(new_round.round_num)
        
        return new_round
    
    def finalize(self) -> RefinementRound:
        """
        Finalize the refinement loop and compute the final result.
        
        Returns:
            The final round with aggregated result
        """
        final_round = self.current_round or RefinementRound(
            round_num=0,
            phase=RefinementPhase.AGGREGATE,
        )
        
        final_round.phase = RefinementPhase.AGGREGATE
        
        # Aggregate responses
        if self.response_aggregator:
            final_result = self.response_aggregator(final_round.responses)
        else:
            final_result = self._default_aggregate_responses(final_round.responses)
        
        final_round.final_result = final_result
        final_round.phase = RefinementPhase.COMPLETE
        
        self._is_running = False
        self.current_round = final_round
        self.rounds_history.append(final_round)
        self.stats.update_from_round(final_round)
        
        if self.on_refinement_complete:
            self.on_refinement_complete(final_result)
        
        return final_round
    
    def _default_aggregate_responses(self, responses: Dict[str, AgentResponse]) -> Any:
        """
        Default aggregation: capacity-weighted voting.
        
        Args:
            responses: Agent responses
            
        Returns:
            Aggregated result
        """
        if not responses:
            return None
        
        # For text responses, return the highest capacity agent's response
        # For numeric responses, return capacity-weighted average
        # This is a simple heuristic; customize via response_aggregator
        
        weighted_responses = []
        total_weight = 0.0
        
        for agent_id, response in responses.items():
            weight = self.get_capacity_weight(agent_id) * response.confidence
            weighted_responses.append({
                "agent_id": agent_id,
                "content": response.content,
                "weight": weight,
                "capacity": self.environment.agent_states[agent_id].capacity if agent_id in self.environment.agent_states else 0,
            })
            total_weight += weight
        
        # Return structured result
        return {
            "responses": weighted_responses,
            "total_weight": total_weight,
            "participant_count": len(responses),
            "rounds_completed": len(self.rounds_history),
        }
    
    def run_complete_loop(
        self,
        task: Task,
        participant_ids: Optional[List[str]] = None,
        response_generator: Optional[Callable[[str, Task, Dict[str, Any]], Tuple[Any, float]]] = None,
        feedback_generator: Optional[Callable[[str, str, AgentResponse], Any]] = None,
    ) -> Any:
        """
        Run a complete refinement loop with optional generators.
        
        This is a convenience method for running the full loop with
        simulated or real agent responses.
        
        Args:
            task: Task to refine
            participant_ids: Specific agents to include
            response_generator: Function to generate responses (agent_id, task, context) -> (content, confidence)
            feedback_generator: Function to generate feedback (sender_id, receiver_id, response) -> feedback
            
        Returns:
            Final aggregated result
        """
        self.initialize(task, participant_ids)
        
        # Default generators if not provided
        if not response_generator:
            response_generator = self._default_response_generator
        
        if not feedback_generator:
            feedback_generator = self._default_feedback_generator
        
        while self._is_running:
            # Collect responses
            for agent_id in self._participant_ids:
                context = {
                    "round": self.current_round.round_num,
                    "previous_feedback": self.current_round.aggregated_feedback.get(agent_id),
                }
                content, confidence = response_generator(agent_id, task, context)
                self.submit_response(agent_id, content, confidence)
            
            # Generate and distribute feedback
            for sender_id in self._participant_ids:
                for receiver_id in self._participant_ids:
                    if sender_id != receiver_id:
                        if receiver_id in self.current_round.responses:
                            response = self.current_round.responses[receiver_id]
                            feedback = feedback_generator(sender_id, receiver_id, response)
                            self.provide_feedback(sender_id, receiver_id, feedback)
            
            # Advance to next round
            self.advance_round()
        
        return self.current_round.final_result if self.current_round else None
    
    def _default_response_generator(
        self,
        agent_id: str,
        task: Task,
        context: Dict[str, Any],
    ) -> Tuple[Any, float]:
        """Default response generator (random for testing)."""
        # Simulate response based on capacity
        capacity = self.get_capacity_weight(agent_id)
        confidence = min(1.0, capacity + random.uniform(0, 0.3))
        content = f"Response from {agent_id} for task {task.task_id}"
        return content, confidence
    
    def _default_feedback_generator(
        self,
        sender_id: str,
        receiver_id: str,
        response: AgentResponse,
    ) -> Any:
        """Default feedback generator."""
        return {
            "type": "general",
            "comment": f"Feedback from {sender_id}",
            "score": random.uniform(0.5, 1.0),
        }
    
    def get_refinement_stats(self) -> Dict[str, Any]:
        """Get statistics about the refinement process."""
        return {
            "total_rounds": self.stats.total_rounds,
            "total_responses": self.stats.total_responses,
            "total_feedback": self.stats.total_feedback,
            "avg_response_confidence": self.stats.avg_response_confidence,
            "convergence_round": self.stats.convergence_round,
            "capacity_influence_total": dict(self.stats.capacity_influence_total),
            "is_running": self._is_running,
            "participant_count": len(self._participant_ids),
        }
    
    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get history of all rounds."""
        return [
            {
                "round_num": r.round_num,
                "phase": r.phase.value,
                "response_count": len(r.responses),
                "feedback_count": sum(len(fb) for fb in r.feedback.values()),
                "has_final_result": r.final_result is not None,
            }
            for r in self.rounds_history
        ]
    
    def get_influential_agents(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get agents ranked by their total influence in the refinement.
        
        Influence is based on capacity-weighted feedback provided.
        
        Args:
            top_k: Number of top agents to return
            
        Returns:
            List of (agent_id, influence_score) tuples
        """
        sorted_influence = sorted(
            self.stats.capacity_influence_total.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_influence[:top_k]
    
    def reset(self) -> None:
        """Reset the refinement loop state."""
        self.current_round = None
        self.rounds_history = []
        self.stats = RefinementStats()
        self._task = None
        self._participant_ids = []
        self._is_running = False


def create_refinement_loop(
    environment: MultiAgentEnvironment,
    max_rounds: int = 5,
    convergence_threshold: float = 0.95,
) -> RefinementLoop:
    """
    Factory function to create a refinement loop.
    
    Args:
        environment: Multi-agent environment
        max_rounds: Maximum refinement rounds
        convergence_threshold: Confidence threshold for convergence
        
    Returns:
        Configured RefinementLoop instance
    """
    return RefinementLoop(
        environment=environment,
        max_rounds=max_rounds,
        convergence_threshold=convergence_threshold,
    )


if __name__ == "__main__":
    # Demo usage
    from src.environment import create_collaboration_environment
    
    env = create_collaboration_environment(num_workers=3)
    loop = RefinementLoop(environment=env, max_rounds=3)
    
    task = Task(task_id="demo_task", prompt="Solve this problem")
    
    print(f"Agents: {list(env.agents.keys())}")
    print(f"Capacities: {[(aid, env.agent_states[aid].capacity) for aid in env.agents]}")
    
    # Run complete loop
    result = loop.run_complete_loop(task)
    
    print(f"\nRefinement completed!")
    print(f"Stats: {loop.get_refinement_stats()}")
    print(f"Rounds: {len(loop.rounds_history)}")
