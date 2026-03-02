"""
Multi-Agent Environment

An environment where multiple reasoning agents can communicate and collaborate on tasks.
Supports message passing, task distribution, and result aggregation.
"""

import random
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict


class AgentRole(Enum):
    """Roles that agents can take in the environment."""
    WORKER = "worker"
    COORDINATOR = "coordinator"
    EVALUATOR = "evaluator"


class MessageType(Enum):
    """Types of messages agents can exchange."""
    TASK = "task"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    QUERY = "query"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """A message exchanged between agents."""
    
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Any
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_broadcast(self) -> bool:
        return self.receiver_id is None


@dataclass
class Task:
    """A task to be solved by the multi-agent system."""
    
    task_id: str
    prompt: str
    task_type: str = "reasoning"
    difficulty: float = 1.0
    max_rounds: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """State tracking for an agent in the environment."""
    
    agent_id: str
    role: AgentRole
    capacity: float = 0.0
    compute_budget: float = 1.0
    messages_sent: int = 0
    messages_received: int = 0
    tasks_completed: int = 0
    last_activity: Optional[float] = None


class MultiAgentEnvironment:
    """
    Environment for multi-agent reasoning and collaboration.
    
    Supports:
    - Agent registration with roles and capacities
    - Message passing between agents
    - Task distribution and result collection
    - Round-based collaboration
    """
    
    def __init__(
        self,
        name: str = "multi-agent-env",
        max_agents: int = 10,
        message_history_limit: int = 1000,
    ):
        """
        Initialize the multi-agent environment.
        
        Args:
            name: Environment name
            max_agents: Maximum number of agents
            message_history_limit: Max messages to keep in history
        """
        self.name = name
        self.max_agents = max_agents
        self.message_history_limit = message_history_limit
        
        # Agent registry
        self.agents: Dict[str, Any] = {}  # agent_id -> agent instance
        self.agent_states: Dict[str, AgentState] = {}
        
        # Message handling
        self.message_queue: List[Message] = []
        self.message_history: List[Message] = []
        
        # Task management
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self.on_message_sent: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        
        # Track broadcast message delivery
        self._broadcast_seen: Dict[tuple, set] = defaultdict(set)
    
    def register_agent(
        self,
        agent_id: str,
        agent: Any,
        role: AgentRole = AgentRole.WORKER,
        capacity: float = 0.0,
    ) -> bool:
        """
        Register an agent in the environment.
        
        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
            role: Agent's role in the environment
            capacity: Initial information capacity
            
        Returns:
            True if registered successfully
        """
        if len(self.agents) >= self.max_agents:
            return False
        
        if agent_id in self.agents:
            return False
        
        self.agents[agent_id] = agent
        self.agent_states[agent_id] = AgentState(
            agent_id=agent_id,
            role=role,
            capacity=capacity,
        )
        
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the environment."""
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        del self.agent_states[agent_id]
        
        return True
    
    def update_capacity(self, agent_id: str, capacity: float) -> None:
        """Update an agent's information capacity."""
        if agent_id in self.agent_states:
            self.agent_states[agent_id].capacity = capacity
    
    def update_compute_budget(self, agent_id: str, budget: float) -> None:
        """Update an agent's compute budget."""
        if agent_id in self.agent_states:
            self.agent_states[agent_id].compute_budget = budget
    
    def send_message(
        self,
        sender_id: str,
        receiver_id: Optional[str],
        message_type: MessageType,
        content: Any,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """
        Send a message from one agent to another (or broadcast).
        
        Args:
            sender_id: Sender's agent ID
            receiver_id: Receiver's agent ID (None for broadcast)
            message_type: Type of message
            content: Message content
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        
        self.message_queue.append(message)
        self.message_history.append(message)
        
        # Trim history if needed
        if len(self.message_history) > self.message_history_limit:
            self.message_history = self.message_history[-self.message_history_limit:]
        
        # Update sender stats
        if sender_id in self.agent_states:
            self.agent_states[sender_id].messages_sent += 1
            self.agent_states[sender_id].last_activity = message.timestamp
        
        # Callback
        if self.on_message_sent:
            self.on_message_sent(message)
        
        return message
    
    def broadcast(
        self,
        sender_id: str,
        message_type: MessageType,
        content: Any,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """Broadcast a message to all agents."""
        return self.send_message(
            sender_id=sender_id,
            receiver_id=None,
            message_type=message_type,
            content=content,
            metadata=metadata,
        )
    
    def get_messages_for_agent(self, agent_id: str) -> List[Message]:
        """
        Get all pending messages for an agent.
        
        Broadcast messages remain in the queue until all agents have received them.
        
        Args:
            agent_id: Agent to get messages for
            
        Returns:
            List of messages addressed to this agent
        """
        messages = []
        remaining = []
        
        # Track which broadcast messages this agent has seen
        if not hasattr(self, '_broadcast_seen'):
            self._broadcast_seen = defaultdict(set)
        
        for msg in self.message_queue:
            if msg.receiver_id == agent_id:
                # Direct message - always include and remove from queue
                messages.append(msg)
            elif msg.is_broadcast() and msg.sender_id != agent_id:
                # Broadcast - include if agent hasn't seen it (sender doesn't receive their own)
                msg_key = (msg.sender_id, msg.timestamp, id(msg))
                if agent_id not in self._broadcast_seen[msg_key]:
                    messages.append(msg)
                    self._broadcast_seen[msg_key].add(agent_id)
                    # Keep in queue until all agents have seen it
                    if len(self._broadcast_seen[msg_key]) < len(self.agents) - 1:
                        # -1 because sender doesn't receive their own broadcast
                        remaining.append(msg)
                    else:
                        # All agents have seen this broadcast, remove it
                        del self._broadcast_seen[msg_key]
            else:
                remaining.append(msg)
        
        self.message_queue = remaining
        
        # Update receiver stats
        if agent_id in self.agent_states:
            self.agent_states[agent_id].messages_received += len(messages)
        
        return messages
    
    def submit_task(self, task: Task) -> str:
        """
        Submit a task to the environment.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        self.active_tasks[task.task_id] = task
        return task.task_id
    
    def complete_task(
        self,
        task_id: str,
        result: Any,
        agent_id: str,
    ) -> bool:
        """Mark a task as completed with a result."""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks.pop(task_id)
        
        self.completed_tasks[task_id] = {
            "task": task,
            "result": result,
            "completed_by": agent_id,
            "completed_at": datetime.now().timestamp(),
        }
        
        if agent_id in self.agent_states:
            self.agent_states[agent_id].tasks_completed += 1
        
        if self.on_task_complete:
            self.on_task_complete(task_id, result, agent_id)
        
        return True
    
    def get_active_task(self, task_id: str) -> Optional[Task]:
        """Get an active task by ID."""
        return self.active_tasks.get(task_id)
    
    def get_agents_by_role(self, role: AgentRole) -> List[str]:
        """Get all agent IDs with a specific role."""
        return [
            aid for aid, state in self.agent_states.items()
            if state.role == role
        ]
    
    def get_agents_by_capacity(
        self,
        min_capacity: Optional[float] = None,
        max_capacity: Optional[float] = None,
        sort_desc: bool = True,
    ) -> List[str]:
        """
        Get agents filtered by capacity.
        
        Args:
            min_capacity: Minimum capacity filter
            max_capacity: Maximum capacity filter
            sort_desc: Sort by capacity descending
            
        Returns:
            List of agent IDs matching criteria
        """
        agents = []
        
        for aid, state in self.agent_states.items():
            if min_capacity is not None and state.capacity < min_capacity:
                continue
            if max_capacity is not None and state.capacity > max_capacity:
                continue
            agents.append((aid, state.capacity))
        
        if sort_desc:
            agents.sort(key=lambda x: x[1], reverse=True)
        else:
            agents.sort(key=lambda x: x[1])
        
        return [aid for aid, _ in agents]
    
    def get_high_capacity_agents(self, top_k: int = 3) -> List[str]:
        """Get the top K agents by capacity."""
        return self.get_agents_by_capacity()[:top_k]
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get statistics about the environment."""
        total_messages = sum(s.messages_sent for s in self.agent_states.values())
        total_tasks = sum(s.tasks_completed for s in self.agent_states.values())
        
        capacities = [s.capacity for s in self.agent_states.values()]
        avg_capacity = sum(capacities) / len(capacities) if capacities else 0.0
        
        return {
            "name": self.name,
            "num_agents": len(self.agents),
            "max_agents": self.max_agents,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_messages": len(self.message_queue),
            "total_messages_sent": total_messages,
            "total_tasks_completed": total_tasks,
            "average_capacity": avg_capacity,
            "agents_by_role": {
                role.value: len(self.get_agents_by_role(role))
                for role in AgentRole
            },
        }
    
    def reset(self) -> None:
        """Reset the environment state."""
        self.message_queue.clear()
        self.message_history.clear()
        self.active_tasks.clear()
        self.completed_tasks.clear()
        
        for state in self.agent_states.values():
            state.messages_sent = 0
            state.messages_received = 0
            state.tasks_completed = 0
            state.last_activity = None


def create_collaboration_environment(
    num_workers: int = 3,
    num_coordinators: int = 1,
    num_evaluators: int = 1,
) -> MultiAgentEnvironment:
    """
    Create a standard collaboration environment.
    
    Args:
        num_workers: Number of worker agents
        num_coordinators: Number of coordinator agents
        num_evaluators: Number of evaluator agents
        
    Returns:
        Configured MultiAgentEnvironment
    """
    env = MultiAgentEnvironment(
        name="collaboration-env",
        max_agents=num_workers + num_coordinators + num_evaluators + 5,
    )
    
    # Placeholder agents (in real use, these would be actual agent instances)
    for i in range(num_workers):
        env.register_agent(
            agent_id=f"worker_{i}",
            agent=None,
            role=AgentRole.WORKER,
            capacity=random.uniform(0.5, 1.0),
        )
    
    for i in range(num_coordinators):
        env.register_agent(
            agent_id=f"coordinator_{i}",
            agent=None,
            role=AgentRole.COORDINATOR,
            capacity=random.uniform(0.7, 1.0),
        )
    
    for i in range(num_evaluators):
        env.register_agent(
            agent_id=f"evaluator_{i}",
            agent=None,
            role=AgentRole.EVALUATOR,
            capacity=random.uniform(0.6, 1.0),
        )
    
    return env


if __name__ == "__main__":
    # Demo usage
    env = create_collaboration_environment(num_workers=3)
    
    print(f"Environment: {env.name}")
    print(f"Agents: {env.get_environment_stats()['num_agents']}")
    print(f"High capacity agents: {env.get_high_capacity_agents(top_k=2)}")
    
    # Send a broadcast
    msg = env.broadcast(
        sender_id="coordinator_0",
        message_type=MessageType.TASK,
        content="Analyze the following problem...",
    )
    print(f"\nBroadcast message: {msg.message_type.value}")
    
    # Get messages for a worker
    worker_msgs = env.get_messages_for_agent("worker_0")
    print(f"Worker 0 received {len(worker_msgs)} messages")
