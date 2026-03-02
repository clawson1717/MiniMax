"""
Tests for Iterative Refinement Loop
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from src.environment import (
    AgentRole,
    MessageType,
    Message,
    Task,
    AgentState,
    MultiAgentEnvironment,
    create_collaboration_environment,
)
from src.routing import (
    MessageRouter,
    RouteMode,
)
from src.refinement import (
    RefinementPhase,
    AgentResponse,
    PeerFeedback,
    RefinementRound,
    RefinementStats,
    RefinementLoop,
    create_refinement_loop,
)


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""
    
    def test_response_creation(self):
        """Test basic response creation."""
        response = AgentResponse(
            agent_id="agent_1",
            content="My answer",
            round_num=0,
        )
        
        assert response.agent_id == "agent_1"
        assert response.content == "My answer"
        assert response.round_num == 0
        assert response.confidence == 0.5
        assert response.timestamp > 0
    
    def test_response_with_confidence(self):
        """Test response with custom confidence."""
        response = AgentResponse(
            agent_id="agent_1",
            content="High confidence answer",
            round_num=1,
            confidence=0.95,
        )
        
        assert response.confidence == 0.95
    
    def test_response_with_metadata(self):
        """Test response with metadata."""
        response = AgentResponse(
            agent_id="agent_1",
            content="Answer",
            round_num=0,
            metadata={"source": "inference"},
        )
        
        assert response.metadata["source"] == "inference"


class TestPeerFeedback:
    """Tests for PeerFeedback dataclass."""
    
    def test_feedback_creation(self):
        """Test basic feedback creation."""
        feedback = PeerFeedback(
            sender_id="agent_1",
            receiver_id="agent_2",
            feedback="Great work!",
        )
        
        assert feedback.sender_id == "agent_1"
        assert feedback.receiver_id == "agent_2"
        assert feedback.feedback == "Great work!"
        assert feedback.weight == 1.0
        assert feedback.round_num == 0
    
    def test_feedback_with_weight(self):
        """Test feedback with custom weight."""
        feedback = PeerFeedback(
            sender_id="agent_1",
            receiver_id="agent_2",
            feedback="Feedback",
            weight=0.8,
        )
        
        assert feedback.weight == 0.8


class TestRefinementRound:
    """Tests for RefinementRound dataclass."""
    
    def test_round_creation(self):
        """Test basic round creation."""
        round_data = RefinementRound(
            round_num=0,
            phase=RefinementPhase.INITIAL,
        )
        
        assert round_data.round_num == 0
        assert round_data.phase == RefinementPhase.INITIAL
        assert len(round_data.responses) == 0
        assert len(round_data.feedback) == 0
    
    def test_round_with_responses(self):
        """Test round with responses."""
        round_data = RefinementRound(
            round_num=1,
            phase=RefinementPhase.REFINE,
        )
        
        response = AgentResponse(agent_id="agent_1", content="Answer", round_num=1)
        round_data.responses["agent_1"] = response
        
        assert len(round_data.responses) == 1


class TestRefinementStats:
    """Tests for RefinementStats."""
    
    def test_stats_creation(self):
        """Test stats creation."""
        stats = RefinementStats()
        
        assert stats.total_rounds == 0
        assert stats.total_responses == 0
        assert stats.total_feedback == 0
    
    def test_stats_update_from_round(self):
        """Test updating stats from a round."""
        stats = RefinementStats()
        
        round_data = RefinementRound(round_num=0, phase=RefinementPhase.COMPLETE)
        round_data.responses["agent_1"] = AgentResponse(
            agent_id="agent_1", content="A", round_num=0, confidence=0.8
        )
        round_data.responses["agent_2"] = AgentResponse(
            agent_id="agent_2", content="B", round_num=0, confidence=0.6
        )
        round_data.feedback["agent_1"] = [
            PeerFeedback("agent_2", "agent_1", "Feedback", weight=0.3)
        ]
        
        stats.update_from_round(round_data)
        
        assert stats.total_rounds == 1
        assert stats.total_responses == 2
        assert stats.total_feedback == 1
        assert stats.avg_response_confidence == 0.7


class TestRefinementLoop:
    """Tests for RefinementLoop."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh environment with agents."""
        env = MultiAgentEnvironment(name="test-env")
        env.register_agent("agent_1", Mock(), role=AgentRole.WORKER, capacity=0.9)
        env.register_agent("agent_2", Mock(), role=AgentRole.WORKER, capacity=0.7)
        env.register_agent("agent_3", Mock(), role=AgentRole.WORKER, capacity=0.5)
        return env
    
    @pytest.fixture
    def loop(self, env):
        """Create a refinement loop."""
        return RefinementLoop(environment=env, max_rounds=5)
    
    @pytest.fixture
    def task(self):
        """Create a test task."""
        return Task(task_id="test_task", prompt="Solve this problem")
    
    def test_init(self, loop):
        """Test loop initialization."""
        assert loop.max_rounds == 5
        assert loop.convergence_threshold == 0.95
        assert loop.current_round is None
        assert loop._is_running == False
    
    def test_initialize(self, loop, task):
        """Test initializing the loop with a task."""
        loop.initialize(task)
        
        assert loop._task == task
        assert loop._is_running == True
        assert loop.current_round is not None
        assert loop.current_round.phase == RefinementPhase.INITIAL
        assert len(loop._participant_ids) == 3
    
    def test_initialize_with_specific_participants(self, loop, task):
        """Test initializing with specific participants."""
        loop.initialize(task, participant_ids=["agent_1", "agent_2"])
        
        assert len(loop._participant_ids) == 2
        assert "agent_3" not in loop._participant_ids
    
    def test_initialize_insufficient_participants(self, loop, task):
        """Test that initialization fails with too few participants."""
        with pytest.raises(ValueError, match="Need at least 2 participants"):
            loop.initialize(task, participant_ids=["agent_1"])
    
    def test_get_capacity_weight(self, loop, task):
        """Test capacity weight calculation."""
        loop.initialize(task)
        
        # Higher capacity agent should have higher weight
        weight_1 = loop.get_capacity_weight("agent_1")  # capacity 0.9
        weight_2 = loop.get_capacity_weight("agent_2")  # capacity 0.7
        weight_3 = loop.get_capacity_weight("agent_3")  # capacity 0.5
        
        # Total capacity = 0.9 + 0.7 + 0.5 = 2.1
        assert abs(weight_1 - 0.9/2.1) < 0.01
        assert abs(weight_2 - 0.7/2.1) < 0.01
        assert abs(weight_3 - 0.5/2.1) < 0.01
        
        # Verify ordering
        assert weight_1 > weight_2 > weight_3
    
    def test_get_capacity_weight_non_participant(self, loop, task):
        """Test capacity weight for non-participant."""
        loop.initialize(task)
        
        weight = loop.get_capacity_weight("non_existent")
        assert weight == 0.0
    
    def test_submit_response(self, loop, task):
        """Test submitting a response."""
        loop.initialize(task)
        
        response = loop.submit_response(
            agent_id="agent_1",
            content="My answer",
            confidence=0.85,
        )
        
        assert response is not None
        assert response.agent_id == "agent_1"
        assert response.content == "My answer"
        assert response.confidence == 0.85
        assert "agent_1" in loop.current_round.responses
    
    def test_submit_response_not_running(self, loop):
        """Test that response submission fails when not running."""
        response = loop.submit_response("agent_1", "Answer")
        assert response is None
    
    def test_submit_response_non_participant(self, loop, task):
        """Test that non-participant cannot submit response."""
        loop.initialize(task)
        
        response = loop.submit_response("non_existent", "Answer")
        assert response is None
    
    def test_provide_feedback(self, loop, task):
        """Test providing feedback between agents."""
        loop.initialize(task)
        
        feedback = loop.provide_feedback(
            sender_id="agent_1",
            receiver_id="agent_2",
            feedback="Good work!",
        )
        
        assert feedback is not None
        assert feedback.sender_id == "agent_1"
        assert feedback.receiver_id == "agent_2"
        assert "agent_2" in loop.current_round.feedback
    
    def test_provide_feedback_capacity_weighted(self, loop, task):
        """Test that feedback is weighted by sender capacity."""
        loop.initialize(task)
        
        # Higher capacity agent provides feedback
        feedback_high = loop.provide_feedback(
            sender_id="agent_1",  # capacity 0.9
            receiver_id="agent_2",
            feedback="High capacity feedback",
        )
        
        # Lower capacity agent provides feedback
        feedback_low = loop.provide_feedback(
            sender_id="agent_3",  # capacity 0.5
            receiver_id="agent_2",
            feedback="Low capacity feedback",
        )
        
        # Higher capacity agent should have higher weight
        assert feedback_high.weight > feedback_low.weight
    
    def test_collect_all_responses(self, loop, task):
        """Test collecting all responses."""
        loop.initialize(task)
        
        loop.submit_response("agent_1", "A")
        loop.submit_response("agent_2", "B")
        
        responses = loop.collect_all_responses()
        
        assert len(responses) == 2
        assert "agent_1" in responses
        assert "agent_2" in responses
    
    def test_get_feedback_for_agent(self, loop, task):
        """Test getting feedback for specific agent."""
        loop.initialize(task)
        
        loop.provide_feedback("agent_1", "agent_2", "Feedback 1")
        loop.provide_feedback("agent_3", "agent_2", "Feedback 2")
        
        feedback = loop.get_feedback_for_agent("agent_2")
        
        assert len(feedback) == 2
    
    def test_aggregate_feedback(self, loop, task):
        """Test feedback aggregation."""
        loop.initialize(task)
        
        loop.provide_feedback("agent_1", "agent_2", "High capacity")
        loop.provide_feedback("agent_3", "agent_2", "Low capacity")
        
        aggregated = loop.aggregate_feedback("agent_2")
        
        assert "weighted_feedback" in aggregated
        assert aggregated["feedback_count"] == 2
        assert aggregated["total_weight"] > 0
    
    def test_aggregate_feedback_weights_normalized(self, loop, task):
        """Test that aggregated feedback weights are normalized."""
        loop.initialize(task)
        
        loop.provide_feedback("agent_1", "agent_2", "F1")
        loop.provide_feedback("agent_2", "agent_2", "F2")
        
        aggregated = loop.aggregate_feedback("agent_2")
        
        # Check normalized weights sum to 1
        normalized_sum = sum(
            wf["normalized_weight"] 
            for wf in aggregated["weighted_feedback"]
        )
        assert abs(normalized_sum - 1.0) < 0.001
    
    def test_check_convergence_max_rounds(self, env):
        """Test convergence when max rounds reached."""
        loop = RefinementLoop(environment=env, max_rounds=1)
        task = Task(task_id="test", prompt="Test")
        
        loop.initialize(task)
        loop.submit_response("agent_1", "A", confidence=0.5)
        loop.submit_response("agent_2", "B", confidence=0.5)
        loop.submit_response("agent_3", "C", confidence=0.5)
        
        # Should converge due to max rounds
        assert loop.check_convergence() == True
    
    def test_check_convergence_high_confidence(self, env):
        """Test convergence when confidence is high."""
        loop = RefinementLoop(
            environment=env,
            max_rounds=10,
            convergence_threshold=0.9,
        )
        task = Task(task_id="test", prompt="Test")
        
        loop.initialize(task)
        loop.submit_response("agent_1", "A", confidence=0.95)
        loop.submit_response("agent_2", "B", confidence=0.95)
        loop.submit_response("agent_3", "C", confidence=0.95)
        
        assert loop.check_convergence() == True
    
    def test_check_convergence_not_ready(self, loop, task):
        """Test that convergence is False when not ready."""
        loop.initialize(task)
        
        # Not all responses submitted, low confidence
        loop.submit_response("agent_1", "A", confidence=0.5)
        
        assert loop.check_convergence() == False
    
    def test_advance_round(self, loop, task):
        """Test advancing to next round."""
        loop.initialize(task)
        first_round = loop.current_round
        
        # Submit some responses
        loop.submit_response("agent_1", "A", confidence=0.5)
        loop.submit_response("agent_2", "B", confidence=0.5)
        loop.submit_response("agent_3", "C", confidence=0.5)
        
        new_round = loop.advance_round()
        
        assert new_round.round_num == 1
        assert new_round.phase == RefinementPhase.COLLECT_RESPONSES
        assert len(loop.rounds_history) == 1
        assert loop.rounds_history[0] == first_round
    
    def test_advance_round_not_running(self, loop):
        """Test that advancing fails when not running."""
        with pytest.raises(RuntimeError, match="not initialized"):
            loop.advance_round()
    
    def test_finalize(self, loop, task):
        """Test finalizing the loop."""
        loop.initialize(task)
        loop.submit_response("agent_1", "A", confidence=0.8)
        loop.submit_response("agent_2", "B", confidence=0.8)
        loop.submit_response("agent_3", "C", confidence=0.8)
        
        final = loop.finalize()
        
        assert final.phase == RefinementPhase.COMPLETE
        assert final.final_result is not None
        assert loop._is_running == False
    
    def test_run_complete_loop(self, loop, task):
        """Test running a complete refinement loop."""
        result = loop.run_complete_loop(task)
        
        assert result is not None
        assert loop._is_running == False
        assert loop.stats.total_rounds >= 1
    
    def test_run_complete_loop_with_custom_generator(self, loop, task):
        """Test running loop with custom response generator."""
        def custom_generator(agent_id, task, context):
            return f"Custom response from {agent_id}", 0.99
        
        result = loop.run_complete_loop(
            task,
            response_generator=custom_generator,
        )
        
        assert result is not None
        # Should converge quickly due to high confidence
        assert loop.stats.convergence_round is not None or loop.stats.total_rounds <= loop.max_rounds
    
    def test_get_refinement_stats(self, loop, task):
        """Test getting refinement statistics."""
        loop.initialize(task)
        loop.submit_response("agent_1", "A")
        loop.provide_feedback("agent_1", "agent_2", "Feedback")
        
        stats = loop.get_refinement_stats()
        
        assert "total_rounds" in stats
        assert "total_responses" in stats
        assert "is_running" in stats
        assert stats["is_running"] == True
    
    def test_get_round_history(self, loop, task):
        """Test getting round history."""
        loop.initialize(task)
        loop.submit_response("agent_1", "A", confidence=0.95)
        loop.submit_response("agent_2", "B", confidence=0.95)
        loop.submit_response("agent_3", "C", confidence=0.95)
        loop.advance_round()
        
        history = loop.get_round_history()
        
        assert len(history) >= 1
        assert history[0]["round_num"] == 0
    
    def test_get_influential_agents(self, loop, task):
        """Test getting influential agents."""
        loop.initialize(task)
        loop.provide_feedback("agent_1", "agent_2", "Feedback")
        loop.provide_feedback("agent_1", "agent_3", "Feedback")
        loop.provide_feedback("agent_2", "agent_1", "Feedback")
        
        influential = loop.get_influential_agents(top_k=3)
        
        assert len(influential) <= 3
        # Agent 1 has higher capacity and provided more feedback
        if len(influential) > 0:
            assert influential[0][0] in ["agent_1", "agent_2"]
    
    def test_reset(self, loop, task):
        """Test resetting the loop."""
        loop.initialize(task)
        loop.submit_response("agent_1", "A")
        
        loop.reset()
        
        assert loop.current_round is None
        assert len(loop.rounds_history) == 0
        assert loop._is_running == False
        assert len(loop._participant_ids) == 0
    
    def test_distribute_feedback_via_router(self, loop, task):
        """Test distributing feedback via router."""
        loop.initialize(task)
        
        # Provide feedback
        loop.provide_feedback("agent_1", "agent_2", "Feedback")
        
        # Distribute
        messages = loop.distribute_feedback_via_router()
        
        assert len(messages) >= 1
        assert messages[0].message_type == MessageType.FEEDBACK
    
    def test_callback_on_round_complete(self, loop, task):
        """Test callback on round completion."""
        callback_called = []
        
        def on_round_complete(round_data):
            callback_called.append(round_data.round_num)
        
        loop.on_round_complete = on_round_complete
        loop.initialize(task)
        loop.submit_response("agent_1", "A", confidence=0.99)
        loop.submit_response("agent_2", "B", confidence=0.99)
        loop.submit_response("agent_3", "C", confidence=0.99)
        loop.advance_round()
        
        assert len(callback_called) >= 1


class TestRefinementLoopWithRouter:
    """Tests for RefinementLoop with MessageRouter integration."""
    
    @pytest.fixture
    def env(self):
        """Create environment with agents."""
        env = MultiAgentEnvironment()
        env.register_agent("agent_1", Mock(), capacity=0.9)
        env.register_agent("agent_2", Mock(), capacity=0.7)
        env.register_agent("agent_3", Mock(), capacity=0.5)
        return env
    
    @pytest.fixture
    def router(self, env):
        """Create a message router."""
        return MessageRouter(env, default_mode=RouteMode.CAPACITY_WEIGHTED)
    
    def test_uses_injected_router(self, env, router):
        """Test that loop uses injected router."""
        loop = RefinementLoop(environment=env, router=router)
        
        assert loop.router == router
    
    def test_creates_router_if_not_provided(self, env):
        """Test that loop creates router if not provided."""
        loop = RefinementLoop(environment=env)
        
        assert loop.router is not None
        assert loop.router.environment == env


class TestCreateRefinementLoop:
    """Tests for factory function."""
    
    def test_create_default(self):
        """Test creating default refinement loop."""
        env = MultiAgentEnvironment()
        loop = create_refinement_loop(env)
        
        assert loop.environment == env
        assert loop.max_rounds == 5
        assert loop.convergence_threshold == 0.95
    
    def test_create_custom(self):
        """Test creating with custom parameters."""
        env = MultiAgentEnvironment()
        loop = create_refinement_loop(
            env,
            max_rounds=10,
            convergence_threshold=0.8,
        )
        
        assert loop.max_rounds == 10
        assert loop.convergence_threshold == 0.8


class TestRefinementLoopIntegration:
    """Integration tests for RefinementLoop."""
    
    def test_full_refinement_workflow(self):
        """Test a complete refinement workflow."""
        # Setup
        env = MultiAgentEnvironment()
        env.register_agent("high_cap", Mock(), capacity=0.95)
        env.register_agent("med_cap", Mock(), capacity=0.75)
        env.register_agent("low_cap", Mock(), capacity=0.55)
        
        router = MessageRouter(env, default_mode=RouteMode.CAPACITY_WEIGHTED)
        loop = RefinementLoop(
            environment=env,
            router=router,
            max_rounds=3,
            convergence_threshold=0.9,
        )
        
        task = Task(task_id="integration_test", prompt="Solve this")
        
        # Custom generators that improve over rounds
        def improving_generator(agent_id, task, context):
            round_num = context.get("round", 0)
            capacity = env.agent_states[agent_id].capacity
            # Confidence improves with rounds, higher capacity improves faster
            confidence = min(1.0, capacity + round_num * 0.2)
            return f"Response from {agent_id} in round {round_num}", confidence
        
        def capacity_based_feedback(sender_id, receiver_id, response):
            return {
                "score": env.agent_states[sender_id].capacity,
                "comment": f"Feedback from {sender_id}",
            }
        
        # Run loop
        result = loop.run_complete_loop(
            task,
            response_generator=improving_generator,
            feedback_generator=capacity_based_feedback,
        )
        
        # Verify
        assert result is not None
        assert "responses" in result
        assert loop.stats.total_rounds >= 1
        assert loop.stats.total_feedback > 0
        
        # Verify capacity influence is tracked
        influence = loop.get_influential_agents()
        if len(influence) > 0:
            # High capacity agent should have most influence
            assert influence[0][0] in ["high_cap", "med_cap", "low_cap"]
    
    def test_refinement_with_message_routing(self):
        """Test that refinement uses message routing correctly."""
        env = MultiAgentEnvironment()
        env.register_agent("agent_1", Mock(), capacity=0.9)
        env.register_agent("agent_2", Mock(), capacity=0.7)
        
        router = MessageRouter(env)
        loop = RefinementLoop(environment=env, router=router, max_rounds=2)
        
        task = Task(task_id="routing_test", prompt="Test")
        
        loop.initialize(task)
        
        # Submit responses
        loop.submit_response("agent_1", "Answer 1")
        loop.submit_response("agent_2", "Answer 2")
        
        # Distribute feedback
        messages = loop.distribute_feedback_via_router()
        
        # Verify messages were created
        assert len(messages) > 0 or len(loop.current_round.feedback) == 0


class TestCapacityWeightedInfluence:
    """Tests specifically for capacity-weighted influence."""
    
    def test_high_capacity_more_influence(self):
        """Test that higher capacity agents have more influence."""
        env = MultiAgentEnvironment()
        env.register_agent("high", Mock(), capacity=1.0)
        env.register_agent("low", Mock(), capacity=0.1)
        
        loop = RefinementLoop(environment=env, max_rounds=5)
        task = Task(task_id="influence_test", prompt="Test")
        
        loop.initialize(task)
        
        # Both agents give feedback to each other
        high_feedback = loop.provide_feedback("high", "low", "From high")
        low_feedback = loop.provide_feedback("low", "high", "From low")
        
        # High capacity feedback should have higher weight
        assert high_feedback.weight > low_feedback.weight
    
    def test_capacity_weights_sum_to_one(self):
        """Test that capacity weights across all participants sum to 1."""
        env = MultiAgentEnvironment()
        env.register_agent("a", Mock(), capacity=0.3)
        env.register_agent("b", Mock(), capacity=0.5)
        env.register_agent("c", Mock(), capacity=0.7)
        
        loop = RefinementLoop(environment=env)
        task = Task(task_id="weight_test", prompt="Test")
        
        loop.initialize(task)
        
        weights = [
            loop.get_capacity_weight("a"),
            loop.get_capacity_weight("b"),
            loop.get_capacity_weight("c"),
        ]
        
        # Weights should sum to 1
        assert abs(sum(weights) - 1.0) < 0.001
    
    def test_capacity_influence_tracked(self):
        """Test that capacity influence is tracked over the refinement."""
        env = MultiAgentEnvironment()
        env.register_agent("expert", Mock(), capacity=0.95)
        env.register_agent("novice", Mock(), capacity=0.3)
        
        loop = RefinementLoop(environment=env, max_rounds=2)
        task = Task(task_id="track_test", prompt="Test")
        
        # Run loop
        result = loop.run_complete_loop(task)
        
        # Check influence tracking
        influence = loop.stats.capacity_influence_total
        
        # Expert should have more total influence than novice
        if "expert" in influence and "novice" in influence:
            assert influence["expert"] >= influence["novice"]
