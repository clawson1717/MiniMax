"""
Tests for Message Routing

Tests the capacity-weighted message routing system for multi-agent environments.
"""

import pytest
from unittest.mock import Mock
import math

from src.routing import (
    RouteMode,
    RoutingDecision,
    RoutingStats,
    MessageRouter,
    create_router,
)
from src.environment import (
    MultiAgentEnvironment,
    MessageType,
    Message,
    AgentRole,
    create_collaboration_environment,
)


class TestRouteMode:
    """Tests for RouteMode enum."""
    
    def test_route_mode_values(self):
        """Test RouteMode enum values."""
        assert RouteMode.BROADCAST.value == "broadcast"
        assert RouteMode.TARGETED.value == "targeted"
        assert RouteMode.CAPACITY_WEIGHTED.value == "capacity_weighted"
    
    def test_route_mode_count(self):
        """Test that we have exactly three routing modes."""
        assert len(RouteMode) == 3


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""
    
    def test_routing_decision_creation(self):
        """Test basic routing decision creation."""
        decision = RoutingDecision(
            message_id="msg_1",
            route_mode=RouteMode.BROADCAST,
            sender_id="agent_1",
            target_agents=["agent_2", "agent_3"],
            capacity_weights={"agent_2": 0.5, "agent_3": 0.5},
        )
        
        assert decision.message_id == "msg_1"
        assert decision.route_mode == RouteMode.BROADCAST
        assert decision.sender_id == "agent_1"
        assert len(decision.target_agents) == 2
        assert decision.timestamp > 0
    
    def test_routing_decision_with_metadata(self):
        """Test routing decision with metadata."""
        decision = RoutingDecision(
            message_id="msg_2",
            route_mode=RouteMode.TARGETED,
            sender_id="agent_1",
            target_agents=["agent_2"],
            capacity_weights={"agent_2": 1.0},
            metadata={"priority": "high"},
        )
        
        assert decision.metadata["priority"] == "high"


class TestRoutingStats:
    """Tests for RoutingStats dataclass."""
    
    def test_routing_stats_initialization(self):
        """Test initial routing stats."""
        stats = RoutingStats()
        
        assert stats.total_messages_routed == 0
        assert stats.broadcast_count == 0
        assert stats.targeted_count == 0
        assert stats.capacity_weighted_count == 0
    
    def test_routing_stats_record_broadcast(self):
        """Test recording broadcast routing."""
        stats = RoutingStats()
        decision = RoutingDecision(
            message_id="msg_1",
            route_mode=RouteMode.BROADCAST,
            sender_id="agent_1",
            target_agents=["agent_2"],
            capacity_weights={"agent_2": 1.0},
        )
        
        stats.record_routing(decision)
        
        assert stats.total_messages_routed == 1
        assert stats.broadcast_count == 1
        assert stats.targeted_count == 0
    
    def test_routing_stats_record_targeted(self):
        """Test recording targeted routing."""
        stats = RoutingStats()
        decision = RoutingDecision(
            message_id="msg_1",
            route_mode=RouteMode.TARGETED,
            sender_id="agent_1",
            target_agents=["agent_2"],
            capacity_weights={"agent_2": 1.0},
        )
        
        stats.record_routing(decision)
        
        assert stats.total_messages_routed == 1
        assert stats.targeted_count == 1
    
    def test_routing_stats_record_capacity_weighted(self):
        """Test recording capacity-weighted routing."""
        stats = RoutingStats()
        decision = RoutingDecision(
            message_id="msg_1",
            route_mode=RouteMode.CAPACITY_WEIGHTED,
            sender_id="agent_1",
            target_agents=["agent_2"],
            capacity_weights={"agent_2": 1.0},
        )
        
        stats.record_routing(decision)
        
        assert stats.total_messages_routed == 1
        assert stats.capacity_weighted_count == 1
    
    def test_routing_stats_tracks_sender(self):
        """Test that stats track messages by sender."""
        stats = RoutingStats()
        decision = RoutingDecision(
            message_id="msg_1",
            route_mode=RouteMode.BROADCAST,
            sender_id="sender_1",
            target_agents=["agent_2"],
            capacity_weights={},
        )
        
        stats.record_routing(decision)
        
        assert stats.messages_by_sender["sender_1"] == 1
    
    def test_routing_stats_tracks_receivers(self):
        """Test that stats track messages by receiver."""
        stats = RoutingStats()
        decision = RoutingDecision(
            message_id="msg_1",
            route_mode=RouteMode.TARGETED,
            sender_id="sender_1",
            target_agents=["receiver_1", "receiver_2"],
            capacity_weights={},
        )
        
        stats.record_routing(decision)
        
        assert stats.messages_by_receiver["receiver_1"] == 1
        assert stats.messages_by_receiver["receiver_2"] == 1


class TestMessageRouter:
    """Tests for MessageRouter class."""
    
    @pytest.fixture
    def env(self):
        """Create a test environment with agents."""
        env = MultiAgentEnvironment(name="test-env")
        env.register_agent("agent_1", Mock(), capacity=0.9)
        env.register_agent("agent_2", Mock(), capacity=0.7)
        env.register_agent("agent_3", Mock(), capacity=0.5)
        return env
    
    @pytest.fixture
    def router(self, env):
        """Create a router for the test environment."""
        return MessageRouter(env)
    
    def test_router_initialization(self, router, env):
        """Test router initialization."""
        assert router.environment == env
        assert router.default_mode == RouteMode.CAPACITY_WEIGHTED
        assert router.temperature == 1.0
        assert len(router.routing_history) == 0
    
    def test_router_with_custom_defaults(self, env):
        """Test router with custom default settings."""
        router = MessageRouter(
            env,
            default_mode=RouteMode.BROADCAST,
            temperature=0.5,
            min_capacity_threshold=0.3,
        )
        
        assert router.default_mode == RouteMode.BROADCAST
        assert router.temperature == 0.5
        assert router.min_capacity_threshold == 0.3
    
    def test_route_broadcast(self, router, env):
        """Test broadcast routing mode."""
        messages = router.route(
            sender_id="agent_1",
            content="Broadcast message",
            mode=RouteMode.BROADCAST,
        )
        
        # Broadcast creates one message that goes to all
        assert len(messages) == 1
        assert messages[0].is_broadcast()
        assert len(router.routing_history) == 1
        assert router.stats.broadcast_count == 1
    
    def test_route_targeted(self, router, env):
        """Test targeted routing mode."""
        messages = router.route(
            sender_id="agent_1",
            content="Targeted message",
            mode=RouteMode.TARGETED,
            target_ids=["agent_2"],
        )
        
        assert len(messages) == 1
        assert messages[0].receiver_id == "agent_2"
        assert router.stats.targeted_count == 1
    
    def test_route_targeted_multiple(self, router, env):
        """Test targeted routing to multiple agents."""
        messages = router.route(
            sender_id="agent_1",
            content="Multi-target message",
            mode=RouteMode.TARGETED,
            target_ids=["agent_2", "agent_3"],
        )
        
        assert len(messages) == 2
        assert router.stats.targeted_count == 1
    
    def test_route_targeted_no_targets_raises(self, router, env):
        """Test that targeted mode without targets raises error."""
        with pytest.raises(ValueError, match="requires target_ids"):
            router.route(
                sender_id="agent_1",
                content="Test",
                mode=RouteMode.TARGETED,
                target_ids=None,
            )
    
    def test_route_capacity_weighted(self, router, env):
        """Test capacity-weighted routing mode."""
        messages = router.route(
            sender_id="agent_1",
            content="Capacity weighted message",
            mode=RouteMode.CAPACITY_WEIGHTED,
        )
        
        # Should route to all eligible agents (agent_2 and agent_3)
        assert len(messages) == 2
        assert router.stats.capacity_weighted_count == 1
    
    def test_route_capacity_weighted_top_k(self, router, env):
        """Test capacity-weighted routing with top_k limit."""
        messages = router.route(
            sender_id="agent_1",
            content="Top K message",
            mode=RouteMode.CAPACITY_WEIGHTED,
            top_k=1,
        )
        
        # Should only route to top 1 (agent_2 with 0.7 capacity)
        assert len(messages) == 1
    
    def test_route_with_default_mode(self, router, env):
        """Test routing with default mode."""
        router.default_mode = RouteMode.BROADCAST
        messages = router.route(
            sender_id="agent_1",
            content="Default mode message",
        )
        
        assert router.stats.broadcast_count == 1
    
    def test_capacity_weights_sum_to_one(self, router, env):
        """Test that softmax capacity weights sum to 1."""
        router.route(
            sender_id="agent_1",
            content="Test message",
            mode=RouteMode.CAPACITY_WEIGHTED,
        )
        
        decision = router.routing_history[0]
        weights = list(decision.capacity_weights.values())
        
        # Weights should sum to approximately 1
        assert abs(sum(weights) - 1.0) < 0.01
    
    def test_higher_capacity_gets_higher_weight(self, router, env):
        """Test that higher capacity agents get higher weights."""
        router.route(
            sender_id="agent_1",
            content="Test message",
            mode=RouteMode.CAPACITY_WEIGHTED,
        )
        
        decision = router.routing_history[0]
        weights = decision.capacity_weights
        
        # agent_2 (0.7) should have higher weight than agent_3 (0.5)
        assert weights["agent_2"] > weights["agent_3"]
    
    def test_route_to_high_capacity_convenience(self, router, env):
        """Test route_to_high_capacity convenience method."""
        messages = router.route_to_high_capacity(
            sender_id="agent_1",
            content="High capacity priority",
            top_k=2,
        )
        
        # Should route to top 2 by capacity (agent_2=0.7, agent_3=0.5)
        assert len(messages) == 2
    
    def test_route_to_role(self):
        """Test routing to agents with specific role."""
        env = MultiAgentEnvironment()
        env.register_agent("worker_1", Mock(), role=AgentRole.WORKER, capacity=0.8)
        env.register_agent("worker_2", Mock(), role=AgentRole.WORKER, capacity=0.6)
        env.register_agent("coord_1", Mock(), role=AgentRole.COORDINATOR, capacity=0.9)
        
        router = MessageRouter(env)
        
        messages = router.route_to_role(
            sender_id="coord_1",
            content="Task for workers",
            role=AgentRole.WORKER,
        )
        
        assert len(messages) == 2
        # Verify all targets are workers
        for msg in messages:
            assert msg.receiver_id.startswith("worker_")
    
    def test_sample_by_capacity(self, router, env):
        """Test probabilistic sampling by capacity."""
        messages = router.sample_by_capacity(
            sender_id="agent_1",
            content="Sampled message",
            num_recipients=1,
        )
        
        assert len(messages) == 1
        assert messages[0].receiver_id in ["agent_2", "agent_3"]
    
    def test_min_capacity_threshold(self, env):
        """Test minimum capacity threshold filtering."""
        router = MessageRouter(env, min_capacity_threshold=0.6)
        
        messages = router.route(
            sender_id="agent_1",
            content="Filtered message",
            mode=RouteMode.CAPACITY_WEIGHTED,
        )
        
        # Only agent_2 (0.7) should receive, agent_3 (0.5) is below threshold
        assert len(messages) == 1
        assert messages[0].receiver_id == "agent_2"
    
    def test_routing_stats(self, router, env):
        """Test getting routing statistics."""
        router.route("agent_1", "msg1", mode=RouteMode.BROADCAST)
        router.route("agent_1", "msg2", mode=RouteMode.TARGETED, target_ids=["agent_2"])
        
        stats = router.get_routing_stats()
        
        assert stats["total_messages_routed"] == 2
        assert stats["broadcast_count"] == 1
        assert stats["targeted_count"] == 1
    
    def test_get_most_routed_agents(self, router, env):
        """Test getting most routed agents."""
        # Route multiple times to agent_2
        router.route("agent_1", "msg1", mode=RouteMode.TARGETED, target_ids=["agent_2"])
        router.route("agent_1", "msg2", mode=RouteMode.TARGETED, target_ids=["agent_2"])
        router.route("agent_1", "msg3", mode=RouteMode.TARGETED, target_ids=["agent_3"])
        
        most_routed = router.get_most_routed_agents()
        
        assert most_routed[0][0] == "agent_2"
        assert most_routed[0][1] == 2
    
    def test_get_most_active_senders(self, router, env):
        """Test getting most active senders."""
        router.route("agent_1", "msg1", mode=RouteMode.BROADCAST)
        router.route("agent_1", "msg2", mode=RouteMode.BROADCAST)
        router.route("agent_2", "msg3", mode=RouteMode.BROADCAST)
        
        most_active = router.get_most_active_senders()
        
        assert most_active[0][0] == "agent_1"
        assert most_active[0][1] == 2
    
    def test_set_temperature(self, router):
        """Test updating temperature."""
        router.set_temperature(0.5)
        assert router.temperature == 0.5
    
    def test_set_min_capacity_threshold(self, router):
        """Test updating minimum capacity threshold."""
        router.set_min_capacity_threshold(0.5)
        assert router.min_capacity_threshold == 0.5
    
    def test_clear_history(self, router, env):
        """Test clearing routing history."""
        router.route("agent_1", "msg1", mode=RouteMode.BROADCAST)
        
        router.clear_history()
        
        assert len(router.routing_history) == 0
        assert router.stats.total_messages_routed == 0
    
    def test_get_recent_routing_decisions(self, router, env):
        """Test getting recent routing decisions."""
        router.route("agent_1", "msg1", mode=RouteMode.BROADCAST)
        router.route("agent_1", "msg2", mode=RouteMode.TARGETED, target_ids=["agent_2"])
        
        recent = router.get_recent_routing_decisions(limit=1)
        
        assert len(recent) == 1
        assert recent[0].route_mode == RouteMode.TARGETED
    
    def test_routing_history_limit(self, env):
        """Test routing history trimming."""
        router = MessageRouter(env, routing_history_limit=5)
        
        for i in range(10):
            router.route("agent_1", f"msg_{i}", mode=RouteMode.BROADCAST)
        
        assert len(router.routing_history) == 5
    
    def test_on_routed_callback(self, env):
        """Test on_routed callback."""
        callback = Mock()
        router = MessageRouter(env)
        router.on_routed = callback
        
        router.route("agent_1", "msg", mode=RouteMode.BROADCAST)
        
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], RoutingDecision)


class TestCreateRouter:
    """Tests for create_router factory function."""
    
    def test_create_router_default(self):
        """Test creating router with defaults."""
        env = MultiAgentEnvironment()
        router = create_router(env)
        
        assert router.default_mode == RouteMode.CAPACITY_WEIGHTED
        assert router.temperature == 1.0
    
    def test_create_router_custom(self):
        """Test creating router with custom settings."""
        env = MultiAgentEnvironment()
        router = create_router(env, mode="broadcast", temperature=0.5)
        
        assert router.default_mode == RouteMode.BROADCAST
        assert router.temperature == 0.5


class TestSoftmaxWeights:
    """Tests for softmax weight computation."""
    
    def test_softmax_weights_sum_to_one(self):
        """Test that softmax weights sum to 1."""
        env = MultiAgentEnvironment()
        env.register_agent("a1", Mock(), capacity=0.9)
        env.register_agent("a2", Mock(), capacity=0.5)
        env.register_agent("a3", Mock(), capacity=0.1)
        
        router = MessageRouter(env)
        router.route("a1", "test", mode=RouteMode.CAPACITY_WEIGHTED)
        
        weights = list(router.routing_history[0].capacity_weights.values())
        assert abs(sum(weights) - 1.0) < 0.001
    
    def test_softmax_with_uniform_capacities(self):
        """Test softmax when all capacities are equal."""
        env = MultiAgentEnvironment()
        env.register_agent("a1", Mock(), capacity=0.5)
        env.register_agent("a2", Mock(), capacity=0.5)
        env.register_agent("a3", Mock(), capacity=0.5)
        
        router = MessageRouter(env)
        router.route("a1", "test", mode=RouteMode.CAPACITY_WEIGHTED)
        
        weights = list(router.routing_history[0].capacity_weights.values())
        
        # With equal capacities, weights should be uniform (a2 and a3 are recipients)
        # Each should have weight 0.5
        for w in weights:
            assert abs(w - 0.5) < 0.01
    
    def test_softmax_low_temperature(self):
        """Test softmax with low temperature (more peaked)."""
        env = MultiAgentEnvironment()
        env.register_agent("high", Mock(), capacity=0.9)
        env.register_agent("low", Mock(), capacity=0.2)
        
        router = MessageRouter(env, temperature=0.1)
        router.route("high", "test", mode=RouteMode.CAPACITY_WEIGHTED)
        
        weights = router.routing_history[0].capacity_weights
        
        # "high" is sender, "low" is the only recipient with weight 1.0
        assert "low" in weights
        assert abs(weights["low"] - 1.0) < 0.01
    
    def test_softmax_high_temperature(self):
        """Test softmax with high temperature (more uniform)."""
        env = MultiAgentEnvironment()
        env.register_agent("high", Mock(), capacity=0.9)
        env.register_agent("low", Mock(), capacity=0.2)
        env.register_agent("med", Mock(), capacity=0.5)
        
        router = MessageRouter(env, temperature=10.0)
        router.route("high", "test", mode=RouteMode.CAPACITY_WEIGHTED)
        
        weights = router.routing_history[0].capacity_weights
        
        # High temperature should make weights more uniform
        # Recipients are "low" (0.2) and "med" (0.5)
        # With high temperature, the difference between their weights should be small
        assert "low" in weights
        assert "med" in weights
        weight_diff = abs(weights["low"] - weights["med"])
        assert weight_diff < 0.1  # Should be nearly equal


class TestRouterIntegration:
    """Integration tests for MessageRouter with environment."""
    
    def test_full_routing_workflow(self):
        """Test a complete routing workflow."""
        env = create_collaboration_environment(num_workers=3, num_coordinators=1)
        router = MessageRouter(env)
        
        # Coordinator routes task to high-capacity workers
        messages = router.route_to_high_capacity(
            sender_id="coordinator_0",
            content="Analyze this problem",
            top_k=2,
        )
        
        assert len(messages) > 0
        
        # Workers receive messages
        for msg in messages:
            worker_msgs = env.get_messages_for_agent(msg.receiver_id)
            assert len(worker_msgs) > 0
    
    def test_routing_with_task_completion(self):
        """Test routing integrated with task completion."""
        env = MultiAgentEnvironment()
        env.register_agent("coord", Mock(), role=AgentRole.COORDINATOR, capacity=0.95)
        env.register_agent("worker_1", Mock(), role=AgentRole.WORKER, capacity=0.8)
        env.register_agent("worker_2", Mock(), role=AgentRole.WORKER, capacity=0.6)
        
        router = MessageRouter(env)
        
        # Route task to workers
        router.route_to_role(
            sender_id="coord",
            content="Complete this task",
            role=AgentRole.WORKER,
        )
        
        # Workers receive and process
        w1_msgs = env.get_messages_for_agent("worker_1")
        w2_msgs = env.get_messages_for_agent("worker_2")
        
        assert len(w1_msgs) == 1
        assert len(w2_msgs) == 1
    
    def test_capacity_based_prioritization(self):
        """Test that high-capacity agents get prioritized in routing."""
        env = MultiAgentEnvironment()
        env.register_agent("sender", Mock(), capacity=0.5)
        env.register_agent("high_cap", Mock(), capacity=0.95)
        env.register_agent("med_cap", Mock(), capacity=0.70)
        env.register_agent("low_cap", Mock(), capacity=0.30)
        
        router = MessageRouter(env)
        
        # Route to top 2 by capacity
        messages = router.route_to_high_capacity(
            sender_id="sender",
            content="Priority task",
            top_k=2,
        )
        
        # Should go to high_cap and med_cap
        receivers = [m.receiver_id for m in messages]
        assert "high_cap" in receivers
        assert "med_cap" in receivers
        assert "low_cap" not in receivers
    
    def test_routing_excludes_sender(self):
        """Test that routing excludes sender from recipients."""
        env = MultiAgentEnvironment()
        env.register_agent("a1", Mock(), capacity=0.9)
        env.register_agent("a2", Mock(), capacity=0.7)
        
        router = MessageRouter(env)
        
        messages = router.route(
            sender_id="a1",
            content="Test",
            mode=RouteMode.CAPACITY_WEIGHTED,
        )
        
        receivers = [m.receiver_id for m in messages]
        assert "a1" not in receivers
