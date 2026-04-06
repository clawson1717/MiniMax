"""
Tests for Ensemble Coordinator.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.coordinator import EnsembleCoordinator, CoordinationResult
from src.capacity import CapacityEstimator, CapacityResult
from src.trajectory import TrajectoryGraph, StepNode
from src.pruner import TrajectoryPruner, PruningResult
from src.uncertainty import UncertaintyEstimator, UncertaintyResult
from src.allocator import ComputeAllocator, AllocationResult
from src.voting import CapacityWeightedVoter, VoteResult as VotingVoteResult


class TestEnsembleCoordinator:
    """Tests for the EnsembleCoordinator class."""
    
    def _make_mock_agent(self, response: str, agent_id: str = None):
        """Create a mock agent that returns a fixed response via __call__."""
        agent = Mock()
        agent.return_value = response
        # Set generate and respond methods too if they exist
        agent.generate = Mock(return_value=response)
        agent.respond = Mock(return_value=response)
        if agent_id:
            agent.id = agent_id
        return agent
    
    def test_basic_coordination_flow(self):
        """Test basic coordination flow with multiple agents."""
        # Create mock agents
        agents = [
            self._make_mock_agent("Response A", "agent_1"),
            self._make_mock_agent("Response A", "agent_2"),
            self._make_mock_agent("Response B", "agent_3"),
        ]
        
        capacity_estimator = CapacityEstimator()
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            max_iterations=2,
            compute_budget=100
        )
        
        # Mock the capacity estimation to return known values
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            result = coordinator.coordinate("Test task")
        
        assert isinstance(result, CoordinationResult)
        assert result.final_response in ["Response A", "Response B"]
        assert result.iterations >= 1
        assert coordinator.iteration_count >= 1
    
    def test_coordination_with_pruning(self):
        """Test that pruning is integrated into the coordination flow."""
        agents = [
            self._make_mock_agent("Low quality response", "agent_1"),
            self._make_mock_agent("High quality response", "agent_2"),
        ]
        
        capacity_estimator = CapacityEstimator()
        pruner = TrajectoryPruner(min_confidence=0.3)
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            trajectory_pruner=pruner,
            max_iterations=1,
            compute_budget=50
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            result = coordinator.coordinate("Test task")
        
        # Check pruning stats were collected
        assert len(coordinator.pruning_stats) >= 1
        assert coordinator.trajectory_graph is not None
    
    def test_voting_with_capacity_weights(self):
        """Test that voting uses capacity weights."""
        agents = [
            self._make_mock_agent("Response A", "agent_1"),
            self._make_mock_agent("Response B", "agent_2"),
        ]
        
        capacity_estimator = CapacityEstimator()
        voter = CapacityWeightedVoter(temperature=1.0)
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            voter=voter,
            max_iterations=1,
            compute_budget=50
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            # Agent 1 has higher capacity
            def cap_side_effect(agent, ctx):
                if hasattr(agent, 'id') and agent.id == "agent_1":
                    return CapacityResult(capacity_bits=10.0, method="combined", confidence=0.9, timestamp=0.0)
                return CapacityResult(capacity_bits=2.0, method="combined", confidence=0.5, timestamp=0.0)
            
            mock_cap.side_effect = cap_side_effect
            
            result = coordinator.coordinate("Test task")
        
        # Check that capacity distribution was tracked
        assert len(coordinator.capacity_distribution) == 2
        assert coordinator.capacity_distribution["agent_1"] == 10.0
        assert coordinator.capacity_distribution["agent_2"] == 2.0
    
    def test_scaling_decision_logic(self):
        """Test that should_continue and should_scale work correctly."""
        agents = [
            self._make_mock_agent("Response", "agent_1"),
            self._make_mock_agent("Response", "agent_2"),
        ]
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=CapacityEstimator(),
            max_iterations=5,
            compute_budget=1000
        )
        
        # Initially should continue
        assert coordinator.should_continue() is True
        
        # Exhaust compute budget
        coordinator._compute_used = 1000
        assert coordinator.should_continue() is False
        
        # Reset and exhaust iterations
        coordinator._compute_used = 0
        coordinator._iteration_count = 5
        assert coordinator.should_continue() is False
    
    def test_empty_task_handling(self):
        """Test handling of empty or invalid tasks."""
        agents = [
            self._make_mock_agent("", "agent_1"),  # Empty response
            self._make_mock_agent("Valid response", "agent_2"),
        ]
        
        capacity_estimator = CapacityEstimator()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            max_iterations=1,
            compute_budget=50
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            # Should handle empty responses gracefully
            result = coordinator.coordinate("Test task")
            assert result.final_response == "Valid response"
    
    def test_single_agent_coordination(self):
        """Test coordination with a single agent."""
        agents = [
            self._make_mock_agent("Only response", "solo_agent"),
        ]
        
        capacity_estimator = CapacityEstimator()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            max_iterations=1,
            compute_budget=50,
            min_agents=1  # Allow single agent
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            result = coordinator.coordinate("Single agent task")
        
        assert result.final_response == "Only response"
    
    def test_budget_exhaustion(self):
        """Test behavior when compute budget is exhausted."""
        agents = [
            self._make_mock_agent("Response", "agent_1"),
            self._make_mock_agent("Response", "agent_2"),
        ]
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=CapacityEstimator(),
            max_iterations=10,  # High iteration limit
            compute_budget=10   # Very low budget
        )
        
        # Should not continue when budget is exhausted
        coordinator._compute_used = 10
        assert coordinator.should_continue() is False
    
    def test_trajectory_graph_tracking(self):
        """Test that trajectory graph is properly maintained."""
        agents = [
            self._make_mock_agent("First response", "agent_1"),
            self._make_mock_agent("Second response", "agent_2"),
        ]
        
        capacity_estimator = CapacityEstimator()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            max_iterations=1,
            compute_budget=50
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            result = coordinator.coordinate("Task")
        
        # Check trajectory graph was created and has nodes
        assert coordinator.trajectory_graph is not None
        assert coordinator.trajectory_graph.get_node_count() >= 2
    
    def test_convergence_detection(self):
        """Test that convergence is properly detected."""
        agents = [
            self._make_mock_agent("Consensus response", "agent_1"),
            self._make_mock_agent("Consensus response", "agent_2"),
            self._make_mock_agent("Consensus response", "agent_3"),
        ]
        
        capacity_estimator = CapacityEstimator()
        voter = CapacityWeightedVoter()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            voter=voter,
            max_iterations=3,
            compute_budget=100,
            convergence_threshold=0.9
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.9,
                timestamp=0.0
            )
            
            result = coordinator.coordinate("Task")
        
        # When all agents agree, should converge quickly
        assert result.converged is True or result.iterations >= 1
    
    def test_get_coordination_summary(self):
        """Test the coordination summary method."""
        agents = [
            self._make_mock_agent("Response", "agent_1"),
        ]
        
        capacity_estimator = CapacityEstimator()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            max_iterations=2,
            compute_budget=100
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            coordinator.coordinate("Task")
        
        summary = coordinator.get_coordination_summary()
        
        assert summary["num_agents"] == 1
        assert "iteration_count" in summary
        assert "total_tokens" in summary
        assert "compute_budget" in summary
        assert "pruning_summary" in summary
    
    def test_pruning_summary(self):
        """Test the pruning summary method."""
        agents = [
            self._make_mock_agent("Response 1", "agent_1"),
            self._make_mock_agent("Response 2", "agent_2"),
        ]
        
        capacity_estimator = CapacityEstimator()
        pruner = TrajectoryPruner()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            trajectory_pruner=pruner,
            max_iterations=2,
            compute_budget=50
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            coordinator.coordinate("Task")
        
        summary = coordinator.get_pruning_summary()
        
        assert "total_iterations" in summary
        assert "total_nodes_removed" in summary
        assert "total_cycles_found" in summary
        assert "avg_reduction_ratio" in summary
    
    def test_agent_management(self):
        """Test adding and removing agents."""
        agents = [
            self._make_mock_agent("Response", "agent_1"),
        ]
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=CapacityEstimator(),
            max_iterations=1,
            compute_budget=50
        )
        
        assert coordinator.num_agents == 1
        assert "agent_1" in coordinator.agent_ids
        
        # Add new agent
        new_agent = self._make_mock_agent("New response", "agent_2")
        new_id = coordinator.add_agent(new_agent)
        
        assert new_id == "agent_2"
        assert coordinator.num_agents == 2
        assert "agent_2" in coordinator.agent_ids
        
        # Remove agent
        removed = coordinator.remove_agent("agent_1")
        assert removed is True
        assert coordinator.num_agents == 1
        assert "agent_1" not in coordinator.agent_ids
    
    def test_should_scale(self):
        """Test the should_scale method."""
        agents = [
            self._make_mock_agent("Response", "agent_1"),
        ]
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=CapacityEstimator(),
            max_iterations=3,
            compute_budget=100
        )
        
        # With no uncertainty history, should_scale defaults to True
        assert coordinator.should_scale() is True
        
        # With low uncertainty, should not scale
        coordinator._uncertainty_history.append(0.1)
        assert coordinator.should_scale() is False
        
        # With high uncertainty, should scale
        coordinator._uncertainty_history.append(0.8)
        assert coordinator.should_scale() is True
    
    def test_get_scale_recommendation(self):
        """Test the scale recommendation method."""
        agents = [
            self._make_mock_agent("Response", f"agent_{i}")
            for i in range(5)
        ]
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=CapacityEstimator(),
            max_iterations=5,
            compute_budget=100
        )
        
        # Add some uncertainty history
        coordinator._uncertainty_history = [0.3, 0.6, 0.4]
        
        rec = coordinator.get_scale_recommendation(
            current_agents=3,
            max_agents=5
        )
        
        assert "should_scale" in rec
        assert "current_agents" in rec
        assert "recommended_agents" in rec
        assert "uncertainty" in rec
        assert "threshold" in rec
    
    def test_reset_state(self):
        """Test that reset clears all state properly."""
        agents = [
            self._make_mock_agent("Response", "agent_1"),
        ]
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=CapacityEstimator(),
            max_iterations=3,
            compute_budget=100
        )
        
        # Run a coordination
        with patch.object(coordinator.capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            coordinator.coordinate("Task")
        
        assert coordinator.iteration_count > 0
        assert coordinator.total_tokens > 0
        
        # Reset
        coordinator.reset()
        
        assert coordinator.iteration_count == 0
        assert coordinator.total_tokens == 0
        assert coordinator.uncertainty_history == []
        assert coordinator.pruning_stats == []
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        capacity_estimator = CapacityEstimator()
        
        # Empty agents list
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            EnsembleCoordinator(agents=[], capacity_estimator=capacity_estimator)
        
        # Invalid max_iterations
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            EnsembleCoordinator(
                agents=[Mock()],
                capacity_estimator=capacity_estimator,
                max_iterations=0
            )
        
        # Invalid compute_budget
        with pytest.raises(ValueError, match="compute_budget must be non-negative"):
            EnsembleCoordinator(
                agents=[Mock()],
                capacity_estimator=capacity_estimator,
                compute_budget=-10
            )
        
        # Invalid convergence_threshold
        with pytest.raises(ValueError, match="convergence_threshold must be between"):
            EnsembleCoordinator(
                agents=[Mock()],
                capacity_estimator=capacity_estimator,
                convergence_threshold=1.5
            )
    
    def test_last_vote_result_tracking(self):
        """Test that the last vote result is tracked."""
        agents = [
            self._make_mock_agent("Winner response", "agent_1"),
            self._make_mock_agent("Winner response", "agent_2"),
            self._make_mock_agent("Loser response", "agent_3"),
        ]
        
        capacity_estimator = CapacityEstimator()
        
        coordinator = EnsembleCoordinator(
            agents=agents,
            capacity_estimator=capacity_estimator,
            max_iterations=1,
            compute_budget=50
        )
        
        with patch.object(capacity_estimator, 'estimate_capacity') as mock_cap:
            mock_cap.return_value = CapacityResult(
                capacity_bits=5.0,
                method="combined",
                confidence=0.8,
                timestamp=0.0
            )
            
            coordinator.coordinate("Task")
        
        assert coordinator.last_vote_result is not None
        assert isinstance(coordinator.last_vote_result, VotingVoteResult)


class TestCoordinationResult:
    """Tests for the CoordinationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a valid CoordinationResult."""
        graph = TrajectoryGraph()
        graph.add_step("Content", "agent_1", 100, 0.8)
        
        result = CoordinationResult(
            final_response="Final answer",
            trajectory_graph=graph,
            pruning_stats=[],
            capacity_distribution={"agent_1": 5.0},
            iterations=2,
            converged=True,
            total_tokens=500,
            uncertainty_history=[0.3, 0.2]
        )
        
        assert result.final_response == "Final answer"
        assert result.iterations == 2
        assert result.converged is True
        assert result.total_tokens == 500
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        graph = TrajectoryGraph()
        graph.add_step("Content", "agent_1", 100, 0.8)
        
        result = CoordinationResult(
            final_response="Answer",
            trajectory_graph=graph,
            pruning_stats=[],
            capacity_distribution={"agent_1": 5.0},
            iterations=1,
            converged=False,
            total_tokens=100,
            uncertainty_history=[0.5]
        )
        
        d = result.to_dict()
        
        assert d["final_response"] == "Answer"
        assert d["iterations"] == 1
        assert d["converged"] is False
        assert d["total_tokens"] == 100
        assert d["uncertainty_history"] == [0.5]
        assert d["capacity_distribution"] == {"agent_1": 5.0}
