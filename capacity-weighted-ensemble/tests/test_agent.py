"""
Tests for Ensemble Agent.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.agent import (
    EnsembleAgent,
    EnsembleResponse,
    VoteResult,
    EnsembleCoordinator,
)
from src.capacity import CapacityEstimator, CapacityResult


class TestVoteResult:
    """Tests for VoteResult dataclass."""
    
    def test_valid_result_creation(self):
        """Test creating a valid VoteResult."""
        result = VoteResult(
            winner="response_a",
            votes={"response_a": 3, "response_b": 1},
            agreement_score=0.75
        )
        assert result.winner == "response_a"
        assert result.votes == {"response_a": 3, "response_b": 1}
        assert result.agreement_score == 0.75
    
    def test_invalid_agreement_score_out_of_range(self):
        """Test that agreement_score outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="agreement_score must be between"):
            VoteResult(winner="a", votes={"a": 1}, agreement_score=1.5)
        
        with pytest.raises(ValueError, match="agreement_score must be between"):
            VoteResult(winner="a", votes={"a": 1}, agreement_score=-0.1)
    
    def test_winner_not_in_votes(self):
        """Test that winner not in votes raises ValueError."""
        with pytest.raises(ValueError, match="winner must be in votes"):
            VoteResult(winner="missing", votes={"a": 1}, agreement_score=1.0)
    
    def test_empty_winner_with_votes(self):
        """Test that empty winner with votes raises ValueError."""
        with pytest.raises(ValueError, match="winner cannot be empty"):
            VoteResult(winner="", votes={"a": 1}, agreement_score=1.0)


class TestEnsembleResponse:
    """Tests for EnsembleResponse dataclass."""
    
    def test_valid_response_creation(self):
        """Test creating a valid EnsembleResponse."""
        response = EnsembleResponse(
            final_response="final answer",
            agent_responses={"agent_1": "answer 1", "agent_2": "answer 2"},
            capacities={"agent_1": 10.0, "agent_2": 5.0},
            vote_result=VoteResult(winner="answer 1", votes={"answer 1": 1, "answer 2": 1}, agreement_score=0.5),
            metadata={"time": 1.5}
        )
        assert response.final_response == "final answer"
        assert len(response.agent_responses) == 2
        assert response.vote_result is not None
    
    def test_response_without_vote_result(self):
        """Test EnsembleResponse without vote_result."""
        response = EnsembleResponse(
            final_response="answer",
            agent_responses={"agent_1": "answer"},
            capacities={"agent_1": 10.0}
        )
        assert response.vote_result is None
        assert response.metadata == {}
    
    def test_invalid_agent_responses_type(self):
        """Test that invalid agent_responses type raises ValueError."""
        with pytest.raises(ValueError, match="agent_responses must be a dictionary"):
            EnsembleResponse(
                final_response="answer",
                agent_responses=["not", "a", "dict"],
                capacities={}
            )
    
    def test_invalid_capacities_type(self):
        """Test that invalid capacities type raises ValueError."""
        with pytest.raises(ValueError, match="capacities must be a dictionary"):
            EnsembleResponse(
                final_response="answer",
                agent_responses={},
                capacities="not a dict"
            )
    
    def test_invalid_metadata_type(self):
        """Test that invalid metadata type raises ValueError."""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            EnsembleResponse(
                final_response="answer",
                agent_responses={},
                capacities={},
                metadata="not a dict"
            )


class TestEnsembleCoordinator:
    """Tests for EnsembleCoordinator."""
    
    def test_majority_strategy(self):
        """Test majority coordination strategy."""
        coordinator = EnsembleCoordinator(strategy="majority")
        result = coordinator.coordinate(
            responses={"a1": "response_a", "a2": "response_a", "a3": "response_b"},
            capacities={"a1": 1.0, "a2": 1.0, "a3": 1.0}
        )
        assert result == "response_a"
    
    def test_weighted_strategy(self):
        """Test weighted coordination strategy."""
        coordinator = EnsembleCoordinator(strategy="weighted")
        # response_b has higher capacity backing
        result = coordinator.coordinate(
            responses={"a1": "response_a", "a2": "response_b"},
            capacities={"a1": 1.0, "a2": 10.0}
        )
        assert result == "response_b"
    
    def test_consensus_strategy(self):
        """Test consensus coordination strategy."""
        coordinator = EnsembleCoordinator(strategy="consensus")
        result = coordinator.coordinate(
            responses={"a1": "response_a", "a2": "response_a"},
            capacities={"a1": 1.0, "a2": 1.0}
        )
        assert result == "response_a"
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            EnsembleCoordinator(strategy="invalid")
    
    def test_empty_responses_raises_error(self):
        """Test that empty responses raises ValueError."""
        coordinator = EnsembleCoordinator()
        with pytest.raises(ValueError, match="No responses to coordinate"):
            coordinator.coordinate(responses={}, capacities={})


class TestEnsembleAgentInit:
    """Tests for EnsembleAgent initialization."""
    
    def test_init_with_single_callable(self):
        """Test initialization with a single callable agent."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        assert ensemble.num_agents == 1
        assert "agent_0" in ensemble.agent_ids
    
    def test_init_with_multiple_callables(self):
        """Test initialization with multiple callable agents."""
        def agent1(prompt):
            return "response 1"
        
        def agent2(prompt):
            return "response 2"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent1, agent2], estimator)
        
        assert ensemble.num_agents == 2
    
    def test_init_with_generate_method(self):
        """Test initialization with agent that has generate method."""
        agent = Mock()
        agent.generate = Mock(return_value="generated response")
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        assert ensemble.num_agents == 1
    
    def test_init_with_respond_method(self):
        """Test initialization with agent that has respond method."""
        agent = Mock()
        agent.respond = Mock(return_value="responded")
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        assert ensemble.num_agents == 1
    
    def test_init_with_named_agent(self):
        """Test that agent with name attribute uses that as ID."""
        agent = Mock(spec=['name', '__call__'])
        agent.name = "custom_agent"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        assert "custom_agent" in ensemble.agent_ids
    
    def test_init_with_agent_id_attribute(self):
        """Test that agent with agent_id attribute uses that as ID."""
        agent = Mock(spec=['agent_id', '__call__'])
        agent.agent_id = "special_agent"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        assert "special_agent" in ensemble.agent_ids
    
    def test_init_with_coordinator(self):
        """Test initialization with a coordinator."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        coordinator = EnsembleCoordinator(strategy="weighted")
        ensemble = EnsembleAgent([agent], estimator, coordinator)
        
        assert ensemble.coordinator is coordinator
    
    def test_init_empty_agents_raises_error(self):
        """Test that empty agents list raises ValueError."""
        estimator = CapacityEstimator()
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            EnsembleAgent([], estimator)


class TestEnsembleAgentGeneration:
    """Tests for EnsembleAgent generate method."""
    
    def test_single_agent_generation(self):
        """Test generation with a single agent."""
        def agent(prompt):
            return f"Answer: {prompt}"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.generate("What is 2+2?")
        
        assert result.final_response == "Answer: What is 2+2?"
        assert len(result.agent_responses) == 1
        assert result.vote_result is not None
    
    def test_multi_agent_generation(self):
        """Test generation with multiple agents."""
        def agent1(prompt):
            return "response a"
        
        def agent2(prompt):
            return "response b"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent1, agent2], estimator)
        
        result = ensemble.generate("test task")
        
        assert len(result.agent_responses) == 2
        assert "response a" in result.agent_responses.values()
        assert "response b" in result.agent_responses.values()
    
    def test_generation_with_unanimous_agreement(self):
        """Test generation when all agents agree."""
        def agent1(prompt):
            return "same answer"
        
        def agent2(prompt):
            return "same answer"
        
        def agent3(prompt):
            return "same answer"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent1, agent2, agent3], estimator)
        
        result = ensemble.generate("test")
        
        assert result.final_response == "same answer"
        assert result.vote_result.agreement_score == 1.0
    
    def test_generation_with_majority(self):
        """Test generation with majority agreement."""
        def agent1(prompt):
            return "majority answer"
        
        def agent2(prompt):
            return "majority answer"
        
        def agent3(prompt):
            return "minority answer"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent1, agent2, agent3], estimator)
        
        result = ensemble.generate("test")
        
        assert result.final_response == "majority answer"
        assert result.vote_result.agreement_score == pytest.approx(2/3, rel=0.1)
    
    def test_generation_metadata(self):
        """Test that generation includes metadata."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.generate("test")
        
        assert "num_agents" in result.metadata
        assert "elapsed_time_seconds" in result.metadata
        assert result.metadata["num_agents"] == 1


class TestEnsembleAgentVoting:
    """Tests for EnsembleAgent vote method."""
    
    def test_vote_clear_winner(self):
        """Test voting with a clear winner."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.vote(["a", "a", "a", "b"])
        
        assert result.winner == "a"
        assert result.votes["a"] == 3
        assert result.votes["b"] == 1
        assert result.agreement_score == 0.75
    
    def test_vote_with_tie(self):
        """Test voting with a tie (first encountered wins)."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.vote(["a", "a", "b", "b"])
        
        # In case of tie, max() returns the first one encountered
        assert result.winner in ("a", "b")
        assert result.agreement_score == 0.5
    
    def test_vote_unanimous(self):
        """Test unanimous voting."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.vote(["same", "same", "same"])
        
        assert result.winner == "same"
        assert result.agreement_score == 1.0
    
    def test_vote_empty_responses_raises_error(self):
        """Test that empty responses raises ValueError."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        with pytest.raises(ValueError, match="responses cannot be empty"):
            ensemble.vote([])
    
    def test_vote_strips_whitespace(self):
        """Test that voting normalizes whitespace."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.vote(["  a  ", "a", "\ta\t"])
        
        assert result.winner == "a"
        assert result.votes["a"] == 3


class TestEnsembleAgentConsensus:
    """Tests for EnsembleAgent consensus method."""
    
    def test_consensus_basic(self):
        """Test basic consensus mechanism."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.consensus(["a", "a", "b"], {"agent_0": 1.0})
        
        assert result == "a"
    
    def test_consensus_empty_raises_error(self):
        """Test that empty responses raises ValueError."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        with pytest.raises(ValueError, match="responses cannot be empty"):
            ensemble.consensus([], {})


class TestEnsembleAgentManagement:
    """Tests for agent management methods."""
    
    def test_add_agent(self):
        """Test adding an agent to the ensemble."""
        def agent1(prompt):
            return "response 1"
        
        def agent2(prompt):
            return "response 2"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent1], estimator)
        
        assert ensemble.num_agents == 1
        
        agent_id = ensemble.add_agent(agent2, "custom_id")
        assert agent_id == "custom_id"
        assert ensemble.num_agents == 2
        assert "custom_id" in ensemble.agent_ids
    
    def test_add_agent_auto_id(self):
        """Test adding an agent with auto-generated ID."""
        def agent1(prompt):
            return "response"
        
        def agent2(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent1], estimator)
        
        agent_id = ensemble.add_agent(agent2)
        assert agent_id is not None
        assert ensemble.num_agents == 2
    
    def test_remove_agent(self):
        """Test removing an agent from the ensemble."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        agent_id = ensemble.agent_ids[0]
        result = ensemble.remove_agent(agent_id)
        
        assert result is True
        assert ensemble.num_agents == 0
    
    def test_remove_nonexistent_agent(self):
        """Test removing an agent that doesn't exist."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.remove_agent("nonexistent")
        assert result is False
    
    def test_get_agent(self):
        """Test getting an agent by ID."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        agent_id = ensemble.agent_ids[0]
        retrieved = ensemble.get_agent(agent_id)
        
        assert retrieved is agent
    
    def test_get_nonexistent_agent(self):
        """Test getting an agent that doesn't exist."""
        def agent(prompt):
            return "response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.get_agent("nonexistent")
        assert result is None


class TestAgentInterfaces:
    """Tests for different agent interface types."""
    
    def test_callable_agent(self):
        """Test callable agent (function)."""
        def my_agent(prompt):
            return f"Processed: {prompt}"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([my_agent], estimator)
        
        result = ensemble.generate("test")
        assert "Processed: test" in result.final_response
    
    def test_generate_method_agent(self):
        """Test agent with generate method."""
        class MyAgent:
            def generate(self, prompt):
                return f"Generated: {prompt}"
        
        agent = MyAgent()
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.generate("test")
        assert "Generated: test" in result.final_response
    
    def test_respond_method_agent(self):
        """Test agent with respond method."""
        class MyAgent:
            def respond(self, prompt):
                return f"Responded: {prompt}"
        
        agent = MyAgent()
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([agent], estimator)
        
        result = ensemble.generate("test")
        assert "Responded: test" in result.final_response
    
    def test_mixed_agent_types(self):
        """Test ensemble with mixed agent types."""
        def func_agent(prompt):
            return "func"
        
        class MethodAgent:
            def generate(self, prompt):
                return "method"
        
        method_agent = MethodAgent()
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([func_agent, method_agent], estimator)
        
        result = ensemble.generate("test")
        assert len(result.agent_responses) == 2


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_failed_agent_call(self):
        """Test handling of failed agent calls."""
        def bad_agent(prompt):
            raise RuntimeError("Agent failed")
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([bad_agent], estimator)
        
        with pytest.raises(RuntimeError, match="All agents failed"):
            ensemble.generate("test")
    
    def test_partial_agent_failure(self):
        """Test handling when some agents fail."""
        def bad_agent(prompt):
            raise RuntimeError("Agent failed")
        
        def good_agent(prompt):
            return "good response"
        
        estimator = CapacityEstimator()
        ensemble = EnsembleAgent([bad_agent, good_agent], estimator)
        
        result = ensemble.generate("test")
        
        # Should still have one response from good agent
        assert len(result.agent_responses) == 1
        assert "good response" in result.agent_responses.values()
        assert result.metadata["errors"] is not None
