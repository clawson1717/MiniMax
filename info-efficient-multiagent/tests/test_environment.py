"""
Tests for Multi-Agent Environment
"""

import pytest
import time
from unittest.mock import Mock

from src.environment import (
    AgentRole,
    MessageType,
    Message,
    Task,
    AgentState,
    MultiAgentEnvironment,
    create_collaboration_environment,
)


class TestMessage:
    """Tests for Message dataclass."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=MessageType.TASK,
            content="Test content",
        )
        
        assert msg.sender_id == "agent_1"
        assert msg.receiver_id == "agent_2"
        assert msg.message_type == MessageType.TASK
        assert msg.content == "Test content"
        assert msg.timestamp > 0
    
    def test_broadcast_message(self):
        """Test broadcast message (no receiver)."""
        msg = Message(
            sender_id="agent_1",
            receiver_id=None,
            message_type=MessageType.BROADCAST,
            content="Broadcast content",
        )
        
        assert msg.is_broadcast() == True
    
    def test_direct_message(self):
        """Test direct message (has receiver)."""
        msg = Message(
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=MessageType.RESPONSE,
            content="Direct content",
        )
        
        assert msg.is_broadcast() == False


class TestTask:
    """Tests for Task dataclass."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id="task_1",
            prompt="Solve this problem",
        )
        
        assert task.task_id == "task_1"
        assert task.prompt == "Solve this problem"
        assert task.task_type == "reasoning"
        assert task.difficulty == 1.0
        assert task.max_rounds == 5
    
    def test_task_with_options(self):
        """Test task with custom options."""
        task = Task(
            task_id="task_2",
            prompt="Complex problem",
            task_type="math",
            difficulty=0.8,
            max_rounds=10,
            metadata={"source": "benchmark"},
        )
        
        assert task.task_type == "math"
        assert task.difficulty == 0.8
        assert task.max_rounds == 10
        assert task.metadata["source"] == "benchmark"


class TestAgentState:
    """Tests for AgentState dataclass."""
    
    def test_agent_state_creation(self):
        """Test basic agent state creation."""
        state = AgentState(
            agent_id="agent_1",
            role=AgentRole.WORKER,
        )
        
        assert state.agent_id == "agent_1"
        assert state.role == AgentRole.WORKER
        assert state.capacity == 0.0
        assert state.compute_budget == 1.0
        assert state.messages_sent == 0
        assert state.messages_received == 0
    
    def test_agent_state_with_capacity(self):
        """Test agent state with initial capacity."""
        state = AgentState(
            agent_id="agent_1",
            role=AgentRole.COORDINATOR,
            capacity=0.85,
        )
        
        assert state.capacity == 0.85


class TestMultiAgentEnvironment:
    """Tests for MultiAgentEnvironment."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh environment."""
        return MultiAgentEnvironment(name="test-env")
    
    def test_init(self, env):
        """Test environment initialization."""
        assert env.name == "test-env"
        assert env.max_agents == 10
        assert len(env.agents) == 0
    
    def test_register_agent(self, env):
        """Test agent registration."""
        mock_agent = Mock()
        result = env.register_agent(
            agent_id="agent_1",
            agent=mock_agent,
            role=AgentRole.WORKER,
            capacity=0.75,
        )
        
        assert result == True
        assert "agent_1" in env.agents
        assert env.agent_states["agent_1"].capacity == 0.75
    
    def test_register_agent_limit(self):
        """Test max agent limit."""
        env = MultiAgentEnvironment(max_agents=2)
        
        env.register_agent("agent_1", Mock())
        env.register_agent("agent_2", Mock())
        result = env.register_agent("agent_3", Mock())
        
        assert result == False
        assert len(env.agents) == 2
    
    def test_register_duplicate_agent(self, env):
        """Test registering duplicate agent ID."""
        env.register_agent("agent_1", Mock())
        result = env.register_agent("agent_1", Mock())
        
        assert result == False
    
    def test_unregister_agent(self, env):
        """Test agent unregistration."""
        env.register_agent("agent_1", Mock())
        result = env.unregister_agent("agent_1")
        
        assert result == True
        assert "agent_1" not in env.agents
    
    def test_unregister_nonexistent(self, env):
        """Test unregistering nonexistent agent."""
        result = env.unregister_agent("nonexistent")
        assert result == False
    
    def test_update_capacity(self, env):
        """Test updating agent capacity."""
        env.register_agent("agent_1", Mock(), capacity=0.5)
        env.update_capacity("agent_1", 0.9)
        
        assert env.agent_states["agent_1"].capacity == 0.9
    
    def test_update_compute_budget(self, env):
        """Test updating agent compute budget."""
        env.register_agent("agent_1", Mock())
        env.update_compute_budget("agent_1", 2.5)
        
        assert env.agent_states["agent_1"].compute_budget == 2.5
    
    def test_send_message(self, env):
        """Test sending a message."""
        env.register_agent("agent_1", Mock())
        
        msg = env.send_message(
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=MessageType.TASK,
            content="Hello",
        )
        
        assert len(env.message_queue) == 1
        assert msg.sender_id == "agent_1"
        assert env.agent_states["agent_1"].messages_sent == 1
    
    def test_broadcast_message(self, env):
        """Test broadcasting a message."""
        env.register_agent("agent_1", Mock())
        
        msg = env.broadcast(
            sender_id="agent_1",
            message_type=MessageType.BROADCAST,
            content="Hello everyone",
        )
        
        assert msg.is_broadcast() == True
        assert msg.receiver_id is None
    
    def test_get_messages_for_agent(self, env):
        """Test retrieving messages for an agent."""
        env.register_agent("agent_1", Mock())
        env.register_agent("agent_2", Mock())
        
        # Send direct and broadcast messages
        env.send_message("agent_1", "agent_2", MessageType.TASK, "Direct")
        env.broadcast("agent_1", MessageType.BROADCAST, "Broadcast")
        
        messages = env.get_messages_for_agent("agent_2")
        
        assert len(messages) == 2
        assert env.agent_states["agent_2"].messages_received == 2
    
    def test_message_queue_clearing(self, env):
        """Test that retrieved messages are removed from queue."""
        env.register_agent("agent_1", Mock())
        env.register_agent("agent_2", Mock())
        
        env.send_message("agent_1", "agent_2", MessageType.TASK, "Test")
        env.get_messages_for_agent("agent_2")
        
        assert len(env.message_queue) == 0
    
    def test_submit_task(self, env):
        """Test task submission."""
        task = Task(task_id="task_1", prompt="Test task")
        task_id = env.submit_task(task)
        
        assert task_id == "task_1"
        assert "task_1" in env.active_tasks
    
    def test_complete_task(self, env):
        """Test task completion."""
        env.register_agent("agent_1", Mock())
        task = Task(task_id="task_1", prompt="Test task")
        env.submit_task(task)
        
        result = env.complete_task("task_1", "Result", "agent_1")
        
        assert result == True
        assert "task_1" not in env.active_tasks
        assert "task_1" in env.completed_tasks
        assert env.agent_states["agent_1"].tasks_completed == 1
    
    def test_get_agents_by_role(self, env):
        """Test filtering agents by role."""
        env.register_agent("worker_1", Mock(), role=AgentRole.WORKER)
        env.register_agent("worker_2", Mock(), role=AgentRole.WORKER)
        env.register_agent("coord_1", Mock(), role=AgentRole.COORDINATOR)
        
        workers = env.get_agents_by_role(AgentRole.WORKER)
        coords = env.get_agents_by_role(AgentRole.COORDINATOR)
        
        assert len(workers) == 2
        assert len(coords) == 1
    
    def test_get_agents_by_capacity(self, env):
        """Test filtering agents by capacity."""
        env.register_agent("agent_1", Mock(), capacity=0.9)
        env.register_agent("agent_2", Mock(), capacity=0.5)
        env.register_agent("agent_3", Mock(), capacity=0.7)
        
        high_cap = env.get_agents_by_capacity(min_capacity=0.6)
        low_cap = env.get_agents_by_capacity(max_capacity=0.6)
        
        assert len(high_cap) == 2
        assert len(low_cap) == 1
    
    def test_get_agents_by_capacity_sorting(self, env):
        """Test capacity sorting."""
        env.register_agent("agent_1", Mock(), capacity=0.5)
        env.register_agent("agent_2", Mock(), capacity=0.9)
        env.register_agent("agent_3", Mock(), capacity=0.7)
        
        desc = env.get_agents_by_capacity(sort_desc=True)
        asc = env.get_agents_by_capacity(sort_desc=False)
        
        assert desc[0] == "agent_2"  # Highest first
        assert asc[0] == "agent_1"   # Lowest first
    
    def test_get_high_capacity_agents(self, env):
        """Test getting top K agents by capacity."""
        env.register_agent("agent_1", Mock(), capacity=0.9)
        env.register_agent("agent_2", Mock(), capacity=0.5)
        env.register_agent("agent_3", Mock(), capacity=0.8)
        
        top = env.get_high_capacity_agents(top_k=2)
        
        assert len(top) == 2
        assert top[0] == "agent_1"
        assert top[1] == "agent_3"
    
    def test_get_environment_stats(self, env):
        """Test environment statistics."""
        env.register_agent("worker_1", Mock(), role=AgentRole.WORKER, capacity=0.7)
        env.register_agent("coord_1", Mock(), role=AgentRole.COORDINATOR, capacity=0.9)
        
        stats = env.get_environment_stats()
        
        assert stats["num_agents"] == 2
        assert stats["agents_by_role"]["worker"] == 1
        assert stats["agents_by_role"]["coordinator"] == 1
        assert 0.7 < stats["average_capacity"] < 0.9
    
    def test_reset(self, env):
        """Test environment reset."""
        env.register_agent("agent_1", Mock())
        env.send_message("agent_1", "agent_2", MessageType.TASK, "Test")
        
        env.reset()
        
        assert len(env.message_queue) == 0
        assert len(env.active_tasks) == 0
        assert env.agent_states["agent_1"].messages_sent == 0
    
    def test_message_history_limit(self):
        """Test message history trimming."""
        env = MultiAgentEnvironment(message_history_limit=10)
        env.register_agent("agent_1", Mock())
        
        # Send more messages than the limit
        for i in range(15):
            env.send_message("agent_1", None, MessageType.TASK, f"msg_{i}")
        
        assert len(env.message_history) == 10


class TestCreateCollaborationEnvironment:
    """Tests for factory function."""
    
    def test_create_default(self):
        """Test creating default collaboration environment."""
        env = create_collaboration_environment()
        
        stats = env.get_environment_stats()
        assert stats["num_agents"] == 5  # 3 workers + 1 coord + 1 eval
    
    def test_create_custom(self):
        """Test creating custom collaboration environment."""
        env = create_collaboration_environment(
            num_workers=5,
            num_coordinators=2,
            num_evaluators=2,
        )
        
        stats = env.get_environment_stats()
        assert stats["num_agents"] == 9
        assert stats["agents_by_role"]["worker"] == 5
        assert stats["agents_by_role"]["coordinator"] == 2
        assert stats["agents_by_role"]["evaluator"] == 2
    
    def test_agents_have_capacity(self):
        """Test that created agents have capacity values."""
        env = create_collaboration_environment()
        
        for state in env.agent_states.values():
            assert 0.0 <= state.capacity <= 1.0


class TestEnvironmentIntegration:
    """Integration tests for environment."""
    
    def test_full_collaboration_flow(self):
        """Test a complete multi-agent collaboration flow."""
        env = MultiAgentEnvironment()
        
        # Register agents
        env.register_agent("worker_1", Mock(), role=AgentRole.WORKER, capacity=0.8)
        env.register_agent("worker_2", Mock(), role=AgentRole.WORKER, capacity=0.6)
        env.register_agent("coord", Mock(), role=AgentRole.COORDINATOR, capacity=0.9)
        
        # Coordinator broadcasts a task
        task = Task(task_id="task_1", prompt="Solve this problem")
        env.submit_task(task)
        env.broadcast("coord", MessageType.TASK, {"task_id": "task_1"})
        
        # Workers receive the broadcast
        w1_msgs = env.get_messages_for_agent("worker_1")
        w2_msgs = env.get_messages_for_agent("worker_2")
        
        assert len(w1_msgs) == 1
        assert len(w2_msgs) == 1
        
        # Worker 1 responds (higher capacity)
        env.send_message("worker_1", "coord", MessageType.RESPONSE, "My answer")
        
        # Coordinator receives response
        coord_msgs = env.get_messages_for_agent("coord")
        assert len(coord_msgs) == 1
        
        # Complete the task
        env.complete_task("task_1", "Final answer", "coord")
        
        stats = env.get_environment_stats()
        assert stats["completed_tasks"] == 1
    
    def test_capacity_based_routing_simulation(self):
        """Simulate capacity-based message routing."""
        env = MultiAgentEnvironment()
        
        # Register agents with varying capacities
        env.register_agent("agent_1", Mock(), capacity=0.95)
        env.register_agent("agent_2", Mock(), capacity=0.70)
        env.register_agent("agent_3", Mock(), capacity=0.45)
        
        # Get high capacity agents
        high_cap = env.get_high_capacity_agents(top_k=2)
        
        # Route task to high capacity agents
        for agent_id in high_cap:
            env.send_message(
                sender_id="system",
                receiver_id=agent_id,
                message_type=MessageType.TASK,
                content="Priority task",
            )
        
        assert len(env.message_queue) == 2