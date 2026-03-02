"""
Message Routing for Multi-Agent Systems

Implements capacity-weighted message routing that preferentially delivers
messages to agents with higher information capacity. Supports multiple
routing modes for different communication patterns.
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
    AgentRole,
)


class RouteMode(Enum):
    """Routing modes for message delivery."""
    BROADCAST = "broadcast"  # Send to all agents
    TARGETED = "targeted"  # Send to specific agent(s)
    CAPACITY_WEIGHTED = "capacity_weighted"  # Route based on capacity weights


@dataclass
class RoutingDecision:
    """Records a routing decision made by the router."""
    
    message_id: str
    route_mode: RouteMode
    sender_id: str
    target_agents: List[str]
    capacity_weights: Dict[str, float]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingStats:
    """Statistics for message routing."""
    
    total_messages_routed: int = 0
    broadcast_count: int = 0
    targeted_count: int = 0
    capacity_weighted_count: int = 0
    messages_by_sender: Dict[str, int] = field(default_factory=dict)
    messages_by_receiver: Dict[str, int] = field(default_factory=dict)
    
    def record_routing(self, decision: RoutingDecision) -> None:
        """Update stats with a new routing decision."""
        self.total_messages_routed += 1
        
        if decision.route_mode == RouteMode.BROADCAST:
            self.broadcast_count += 1
        elif decision.route_mode == RouteMode.TARGETED:
            self.targeted_count += 1
        elif decision.route_mode == RouteMode.CAPACITY_WEIGHTED:
            self.capacity_weighted_count += 1
        
        # Track sender
        if decision.sender_id not in self.messages_by_sender:
            self.messages_by_sender[decision.sender_id] = 0
        self.messages_by_sender[decision.sender_id] += 1
        
        # Track receivers
        for agent_id in decision.target_agents:
            if agent_id not in self.messages_by_receiver:
                self.messages_by_receiver[agent_id] = 0
            self.messages_by_receiver[agent_id] += 1


class MessageRouter:
    """
    Routes messages in a multi-agent environment based on capacity.
    
    Supports three routing modes:
    - BROADCAST: Send to all agents (except sender)
    - TARGETED: Send to specific target agent(s)
    - CAPACITY_WEIGHTED: Route preferentially to high-capacity agents
    
    The capacity-weighted routing uses softmax over agent capacities to
    determine delivery probabilities, ensuring high-capacity agents receive
    priority while still allowing lower-capacity agents to participate.
    """
    
    def __init__(
        self,
        environment: MultiAgentEnvironment,
        default_mode: RouteMode = RouteMode.CAPACITY_WEIGHTED,
        temperature: float = 1.0,
        min_capacity_threshold: float = 0.0,
        routing_history_limit: int = 500,
    ):
        """
        Initialize the message router.
        
        Args:
            environment: The multi-agent environment to route messages in
            default_mode: Default routing mode when not specified
            temperature: Temperature for capacity-weighted softmax (higher = more uniform)
            min_capacity_threshold: Minimum capacity to receive routed messages
            routing_history_limit: Maximum routing decisions to keep in history
        """
        self.environment = environment
        self.default_mode = default_mode
        self.temperature = temperature
        self.min_capacity_threshold = min_capacity_threshold
        self.routing_history_limit = routing_history_limit
        
        # Tracking
        self.routing_history: List[RoutingDecision] = []
        self.stats = RoutingStats()
        
        # Callbacks
        self.on_routed: Optional[Callable[[RoutingDecision], None]] = None
    
    def route(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.TASK,
        mode: Optional[RouteMode] = None,
        target_ids: Optional[List[str]] = None,
        exclude_sender: bool = True,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """
        Route a message according to the specified or default mode.
        
        Args:
            sender_id: ID of the sending agent
            content: Message content
            message_type: Type of message
            mode: Routing mode (uses default if None)
            target_ids: Specific targets for TARGETED mode
            exclude_sender: Whether to exclude sender from broadcast
            top_k: Limit to top K agents (for capacity-weighted)
            metadata: Additional message metadata
            
        Returns:
            List of created Message objects
        """
        mode = mode or self.default_mode
        
        if mode == RouteMode.BROADCAST:
            return self._route_broadcast(
                sender_id, content, message_type, exclude_sender, metadata
            )
        elif mode == RouteMode.TARGETED:
            return self._route_targeted(
                sender_id, content, message_type, target_ids, metadata
            )
        elif mode == RouteMode.CAPACITY_WEIGHTED:
            return self._route_capacity_weighted(
                sender_id, content, message_type, top_k, metadata
            )
        else:
            raise ValueError(f"Unknown routing mode: {mode}")
    
    def _route_broadcast(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType,
        exclude_sender: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> List[Message]:
        """Broadcast to all agents."""
        target_agents = list(self.environment.agents.keys())
        if exclude_sender and sender_id in target_agents:
            target_agents.remove(sender_id)
        
        # For broadcast, all agents have equal weight
        weights = {aid: 1.0 / len(target_agents) for aid in target_agents} if target_agents else {}
        
        # Create the broadcast message
        msg = self.environment.broadcast(
            sender_id=sender_id,
            message_type=message_type,
            content=content,
            metadata=metadata,
        )
        
        # Record decision
        decision = RoutingDecision(
            message_id=f"{sender_id}_{msg.timestamp}",
            route_mode=RouteMode.BROADCAST,
            sender_id=sender_id,
            target_agents=target_agents,
            capacity_weights=weights,
            metadata=metadata or {},
        )
        self._record_decision(decision)
        
        return [msg]
    
    def _route_targeted(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType,
        target_ids: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> List[Message]:
        """Route to specific target agents."""
        if not target_ids:
            raise ValueError("TARGETED mode requires target_ids")
        
        # Validate targets exist
        valid_targets = [
            tid for tid in target_ids
            if tid in self.environment.agents
        ]
        
        if not valid_targets:
            return []
        
        # Get capacities for weights
        weights = self._get_capacity_weights(valid_targets)
        
        messages = []
        for target_id in valid_targets:
            msg = self.environment.send_message(
                sender_id=sender_id,
                receiver_id=target_id,
                message_type=message_type,
                content=content,
                metadata=metadata,
            )
            messages.append(msg)
        
        # Record decision
        decision = RoutingDecision(
            message_id=f"{sender_id}_{datetime.now().timestamp()}",
            route_mode=RouteMode.TARGETED,
            sender_id=sender_id,
            target_agents=valid_targets,
            capacity_weights=weights,
            metadata=metadata or {},
        )
        self._record_decision(decision)
        
        return messages
    
    def _route_capacity_weighted(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType,
        top_k: Optional[int],
        metadata: Optional[Dict[str, Any]],
    ) -> List[Message]:
        """Route based on capacity weights (preferentially to high-capacity agents)."""
        # Get eligible agents (excluding sender, above threshold)
        eligible_agents = [
            aid for aid, state in self.environment.agent_states.items()
            if aid != sender_id and state.capacity >= self.min_capacity_threshold
        ]
        
        if not eligible_agents:
            return []
        
        # Apply top_k if specified
        if top_k is not None and top_k > 0:
            # Sort by capacity and take top K
            sorted_agents = sorted(
                eligible_agents,
                key=lambda aid: self.environment.agent_states[aid].capacity,
                reverse=True
            )
            eligible_agents = sorted_agents[:top_k]
        
        # Compute softmax weights based on capacity
        weights = self._compute_softmax_weights(eligible_agents)
        
        # Route to all eligible agents with priority (high capacity first)
        messages = []
        for agent_id in eligible_agents:
            msg = self.environment.send_message(
                sender_id=sender_id,
                receiver_id=agent_id,
                message_type=message_type,
                content=content,
                metadata={
                    **(metadata or {}),
                    "capacity_weight": weights.get(agent_id, 0.0),
                    "routing_priority": self.environment.agent_states[agent_id].capacity,
                },
            )
            messages.append(msg)
        
        # Record decision
        decision = RoutingDecision(
            message_id=f"{sender_id}_{datetime.now().timestamp()}",
            route_mode=RouteMode.CAPACITY_WEIGHTED,
            sender_id=sender_id,
            target_agents=eligible_agents,
            capacity_weights=weights,
            metadata=metadata or {},
        )
        self._record_decision(decision)
        
        return messages
    
    def _get_capacity_weights(self, agent_ids: List[str]) -> Dict[str, float]:
        """Get normalized capacity weights for a list of agents."""
        weights = {}
        for aid in agent_ids:
            if aid in self.environment.agent_states:
                weights[aid] = self.environment.agent_states[aid].capacity
            else:
                weights[aid] = 0.0
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _compute_softmax_weights(self, agent_ids: List[str]) -> Dict[str, float]:
        """
        Compute softmax weights over agent capacities.
        
        Higher temperature makes distribution more uniform,
        lower temperature makes it more peaked on high capacity.
        """
        if not agent_ids:
            return {}
        
        # Get capacities
        capacities = []
        for aid in agent_ids:
            cap = self.environment.agent_states[aid].capacity if aid in self.environment.agent_states else 0.0
            capacities.append((aid, cap))
        
        # Compute softmax
        # Add small epsilon to avoid division by zero
        max_cap = max(c for _, c in capacities) if capacities else 0.0
        
        exp_values = []
        for aid, cap in capacities:
            # Numerical stability: subtract max
            exp_val = pow(2.718281828, (cap - max_cap) / self.temperature)
            exp_values.append((aid, exp_val))
        
        total = sum(ev for _, ev in exp_values)
        
        if total == 0:
            # Fallback to uniform
            return {aid: 1.0 / len(agent_ids) for aid in agent_ids}
        
        return {aid: ev / total for aid, ev in exp_values}
    
    def _record_decision(self, decision: RoutingDecision) -> None:
        """Record a routing decision and update stats."""
        self.routing_history.append(decision)
        self.stats.record_routing(decision)
        
        # Trim history
        if len(self.routing_history) > self.routing_history_limit:
            self.routing_history = self.routing_history[-self.routing_history_limit:]
        
        # Callback
        if self.on_routed:
            self.on_routed(decision)
    
    def route_to_high_capacity(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.TASK,
        top_k: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """
        Convenience method to route to top K highest capacity agents.
        
        Args:
            sender_id: Sender's ID
            content: Message content
            message_type: Type of message
            top_k: Number of high-capacity agents to target
            metadata: Optional metadata
            
        Returns:
            List of created messages
        """
        return self.route(
            sender_id=sender_id,
            content=content,
            message_type=message_type,
            mode=RouteMode.CAPACITY_WEIGHTED,
            top_k=top_k,
            metadata=metadata,
        )
    
    def route_to_role(
        self,
        sender_id: str,
        content: Any,
        role: AgentRole,
        message_type: MessageType = MessageType.TASK,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """
        Route to all agents with a specific role.
        
        Args:
            sender_id: Sender's ID
            content: Message content
            role: Target role
            message_type: Type of message
            metadata: Optional metadata
            
        Returns:
            List of created messages
        """
        target_ids = self.environment.get_agents_by_role(role)
        # Exclude sender if they have the same role
        if sender_id in target_ids:
            target_ids.remove(sender_id)
        
        if not target_ids:
            return []
        
        return self.route(
            sender_id=sender_id,
            content=content,
            message_type=message_type,
            mode=RouteMode.TARGETED,
            target_ids=target_ids,
            metadata=metadata,
        )
    
    def sample_by_capacity(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.TASK,
        num_recipients: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """
        Sample recipients based on capacity weights (probabilistic routing).
        
        Unlike route(), this randomly selects recipients weighted by capacity,
        rather than sending to all eligible agents.
        
        Args:
            sender_id: Sender's ID
            content: Message content
            message_type: Type of message
            num_recipients: Number of recipients to sample
            metadata: Optional metadata
            
        Returns:
            List of created messages
        """
        # Get eligible agents
        eligible_agents = [
            aid for aid in self.environment.agents.keys()
            if aid != sender_id
        ]
        
        if not eligible_agents:
            return []
        
        # Compute weights
        weights_dict = self._compute_softmax_weights(eligible_agents)
        
        # Sample without replacement
        agents = list(weights_dict.keys())
        weights = [weights_dict[a] for a in agents]
        
        # Normalize weights for sampling
        total = sum(weights)
        if total == 0:
            weights = [1.0 / len(agents)] * len(agents)
        else:
            weights = [w / total for w in weights]
        
        # Sample
        num_recipients = min(num_recipients, len(agents))
        sampled = random.choices(agents, weights=weights, k=num_recipients)
        
        # Remove duplicates (for without replacement behavior)
        sampled = list(dict.fromkeys(sampled))[:num_recipients]
        
        # Send messages
        messages = []
        for target_id in sampled:
            msg = self.environment.send_message(
                sender_id=sender_id,
                receiver_id=target_id,
                message_type=message_type,
                content=content,
                metadata={
                    **(metadata or {}),
                    "sampled_by_capacity": True,
                    "capacity_weight": weights_dict.get(target_id, 0.0),
                },
            )
            messages.append(msg)
        
        # Record decision
        decision = RoutingDecision(
            message_id=f"{sender_id}_{datetime.now().timestamp()}",
            route_mode=RouteMode.CAPACITY_WEIGHTED,
            sender_id=sender_id,
            target_agents=sampled,
            capacity_weights=weights_dict,
            metadata={"sampled": True, **(metadata or {})},
        )
        self._record_decision(decision)
        
        return messages
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_messages_routed": self.stats.total_messages_routed,
            "broadcast_count": self.stats.broadcast_count,
            "targeted_count": self.stats.targeted_count,
            "capacity_weighted_count": self.stats.capacity_weighted_count,
            "routing_history_size": len(self.routing_history),
            "messages_by_sender": dict(self.stats.messages_by_sender),
            "messages_by_receiver": dict(self.stats.messages_by_receiver),
            "default_mode": self.default_mode.value,
            "temperature": self.temperature,
            "min_capacity_threshold": self.min_capacity_threshold,
        }
    
    def get_most_routed_agents(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get agents who received the most routed messages."""
        sorted_receivers = sorted(
            self.stats.messages_by_receiver.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_receivers[:top_k]
    
    def get_most_active_senders(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get agents who sent the most routed messages."""
        sorted_senders = sorted(
            self.stats.messages_by_sender.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_senders[:top_k]
    
    def set_temperature(self, temperature: float) -> None:
        """Update the temperature for capacity-weighted routing."""
        self.temperature = temperature
    
    def set_min_capacity_threshold(self, threshold: float) -> None:
        """Update the minimum capacity threshold for routing."""
        self.min_capacity_threshold = threshold
    
    def clear_history(self) -> None:
        """Clear routing history and reset stats."""
        self.routing_history.clear()
        self.stats = RoutingStats()
    
    def get_recent_routing_decisions(self, limit: int = 10) -> List[RoutingDecision]:
        """Get the most recent routing decisions."""
        return self.routing_history[-limit:]


def create_router(
    environment: MultiAgentEnvironment,
    mode: str = "capacity_weighted",
    temperature: float = 1.0,
) -> MessageRouter:
    """
    Factory function to create a message router.
    
    Args:
        environment: Multi-agent environment
        mode: Default routing mode
        temperature: Temperature for capacity-weighted routing
        
    Returns:
        Configured MessageRouter instance
    """
    route_mode = RouteMode(mode)
    return MessageRouter(
        environment=environment,
        default_mode=route_mode,
        temperature=temperature,
    )


if __name__ == "__main__":
    # Demo usage
    from src.environment import create_collaboration_environment
    
    env = create_collaboration_environment(num_workers=3)
    router = MessageRouter(env, default_mode=RouteMode.CAPACITY_WEIGHTED)
    
    print(f"Agents in environment: {list(env.agents.keys())}")
    print(f"Agent capacities: {[(aid, env.agent_states[aid].capacity) for aid in env.agents]}")
    
    # Route a message using capacity weighting
    messages = router.route(
        sender_id="coordinator_0",
        content="Please analyze this problem",
        mode=RouteMode.CAPACITY_WEIGHTED,
        top_k=2,
    )
    
    print(f"\nRouted {len(messages)} messages")
    print(f"Routing stats: {router.get_routing_stats()}")
