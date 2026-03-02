"""
Tests for Diffusion-Based Coordinator
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.coordinator import (
    DenoisingStage,
    DiffusionState,
    CoordinatorConfig,
    DiffusionCoordinator,
    create_coordinator,
)
from src.environment import (
    MultiAgentEnvironment,
    AgentRole,
    MessageType,
    create_collaboration_environment,
)
from src.capacity import InformationCapacityEstimator
from src.agent import ReasoningAgent, create_agent


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig dataclass."""
    
    def test_init_defaults(self):
        """Test default configuration."""
        config = CoordinatorConfig()
        
        assert config.num_steps == 10
        assert config.initial_temperature == 1.0
        assert config.final_temperature == 0.1
        assert config.entropy_coef == 0.1
        assert config.min_capacity_weight == 0.01
        assert config.vector_dim == 64
        assert config.convergence_threshold == 0.001
        assert config.max_refinement_iterations == 5
    
    def test_init_custom(self):
        """Test custom configuration."""
        config = CoordinatorConfig(
            num_steps=20,
            initial_temperature=2.0,
            final_temperature=0.01,
            entropy_coef=0.5,
        )
        
        assert config.num_steps == 20
        assert config.initial_temperature == 2.0
        assert config.final_temperature == 0.01
        assert config.entropy_coef == 0.5


class TestDiffusionState:
    """Tests for DiffusionState dataclass."""
    
    def test_init(self):
        """Test state initialization."""
        vector = np.zeros(64)
        state = DiffusionState(
            response_vector=vector,
            current_step=0,
            total_steps=10,
            temperature=1.0,
            entropy=0.5,
        )
        
        assert np.array_equal(state.response_vector, vector)
        assert state.current_step == 0
        assert state.total_steps == 10
        assert state.temperature == 1.0
        assert state.entropy == 0.5
        assert state.stage == DenoisingStage.INIT
        assert state.agent_contributions == {}
    
    def test_with_contributions(self):
        """Test state with agent contributions."""
        vector = np.ones(64)
        contributions = {
            "agent_1": np.zeros(64),
            "agent_2": np.ones(64),
        }
        
        state = DiffusionState(
            response_vector=vector,
            current_step=5,
            total_steps=10,
            temperature=0.5,
            entropy=0.3,
            agent_contributions=contributions,
        )
        
        assert len(state.agent_contributions) == 2
        assert "agent_1" in state.agent_contributions


class TestDiffusionCoordinator:
    """Tests for DiffusionCoordinator class."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh environment with agents."""
        env = MultiAgentEnvironment(name="test-env")
        
        # Register some mock agents
        env.register_agent("agent_1", Mock(), role=AgentRole.WORKER, capacity=0.8)
        env.register_agent("agent_2", Mock(), role=AgentRole.WORKER, capacity=0.6)
        env.register_agent("coord", Mock(), role=AgentRole.COORDINATOR, capacity=0.9)
        
        return env
    
    @pytest.fixture
    def coordinator(self, env):
        """Create a coordinator with the test environment."""
        config = CoordinatorConfig(num_steps=5)
        return DiffusionCoordinator(environment=env, config=config)
    
    def test_init(self, env):
        """Test coordinator initialization."""
        config = CoordinatorConfig(num_steps=5)
        coord = DiffusionCoordinator(environment=env, config=config)
        
        assert coord.environment == env
        assert coord.config.num_steps == 5
        assert len(coord.active_states) == 0
        assert isinstance(coord.capacity_estimator, InformationCapacityEstimator)
    
    def test_init_with_custom_estimator(self, env):
        """Test coordinator with custom capacity estimator."""
        estimator = InformationCapacityEstimator(method="entropy")
        coord = DiffusionCoordinator(environment=env, capacity_estimator=estimator)
        
        assert coord.capacity_estimator.method == "entropy"
    
    def test_update_capacity_weights(self, coordinator, env):
        """Test capacity weight updates."""
        coordinator._update_capacity_weights()
        
        # Weights should sum to 1
        total = sum(coordinator._capacity_weights.values())
        assert abs(total - 1.0) < 0.001
        
        # High capacity agent should have higher weight
        assert coordinator._capacity_weights["coord"] > coordinator._capacity_weights["agent_2"]
    
    def test_get_capacity_weight(self, coordinator):
        """Test getting capacity weight for an agent."""
        coordinator._update_capacity_weights()
        
        weight = coordinator.get_capacity_weight("coord")
        assert 0 <= weight <= 1
    
    def test_get_capacity_weight_nonexistent(self, coordinator):
        """Test getting capacity weight for nonexistent agent."""
        weight = coordinator.get_capacity_weight("nonexistent")
        assert weight == 0.0
    
    def test_temperature_schedule(self, coordinator):
        """Test temperature schedule generation."""
        schedule = coordinator.get_temperature_schedule()
        
        assert len(schedule) == coordinator.config.num_steps
        assert schedule[0] == coordinator.config.initial_temperature
        assert schedule[-1] == coordinator.config.final_temperature
        # Should be monotonically decreasing
        assert all(schedule[i] >= schedule[i+1] for i in range(len(schedule)-1))
    
    def test_initialize_diffusion(self, coordinator):
        """Test diffusion initialization."""
        state = coordinator.initialize_diffusion("test_task", "Test prompt")
        
        assert state.current_step == 0
        assert state.total_steps == coordinator.config.num_steps
        assert state.stage == DenoisingStage.INIT
        assert state.temperature == coordinator.config.initial_temperature
        assert len(state.response_vector) == coordinator.config.vector_dim
        assert "test_task" in coordinator.active_states
    
    def test_compute_vector_entropy(self, coordinator):
        """Test entropy computation for vectors."""
        # Zero vector has zero entropy
        zero_vec = np.zeros(64)
        zero_entropy = coordinator._compute_vector_entropy(zero_vec)
        assert zero_entropy == 0.0
        
        # Uniform vector should have higher entropy
        uniform_vec = np.ones(64)
        uniform_entropy = coordinator._compute_vector_entropy(uniform_vec)
        assert uniform_entropy > 0
        
        # Maximum entropy is log2(dim)
        max_entropy = np.log2(64)
        assert uniform_entropy <= max_entropy
    
    def test_inject_noise(self, coordinator):
        """Test noise injection."""
        vector = np.zeros(64)
        
        # High temperature = more noise
        noisy_high = coordinator._inject_noise(vector.copy(), temperature=1.0)
        noisy_low = coordinator._inject_noise(vector.copy(), temperature=0.1)
        
        high_variance = np.var(noisy_high)
        low_variance = np.var(noisy_low)
        
        assert high_variance > low_variance
    
    def test_entropy_augmentation(self, coordinator):
        """Test entropy augmentation."""
        vector = np.zeros(64)
        
        augmented = coordinator._compute_entropy_augmentation(
            vector,
            entropy_coef=0.5,
        )
        
        # Augmentation should add some noise
        assert not np.array_equal(augmented, vector)
        assert len(augmented) == len(vector)
    
    def test_collect_agent_contributions(self, coordinator):
        """Test collecting contributions from agents."""
        state = coordinator.initialize_diffusion("test_task")
        
        contributions = coordinator.collect_agent_contributions(state)
        
        # Should have contributions from all agents
        assert len(contributions) == 3
        assert "agent_1" in contributions
        assert "agent_2" in contributions
        assert "coord" in contributions
    
    def test_collect_agent_contributions_with_none_agents(self):
        """Test contributions with None agent references."""
        env = MultiAgentEnvironment()
        env.register_agent("agent_1", None, capacity=0.5)
        
        coordinator = DiffusionCoordinator(environment=env)
        state = coordinator.initialize_diffusion("test")
        
        contributions = coordinator.collect_agent_contributions(state)
        
        # Should handle None agents gracefully
        assert len(contributions) == 0
    
    def test_aggregate_contributions(self, coordinator):
        """Test aggregating agent contributions."""
        state = coordinator.initialize_diffusion("test_task")
        contributions = coordinator.collect_agent_contributions(state)
        
        aggregated = coordinator.aggregate_contributions(contributions, temperature=0.5)
        
        assert len(aggregated) == coordinator.config.vector_dim
        assert isinstance(aggregated, np.ndarray)
    
    def test_aggregate_contributions_empty(self, coordinator):
        """Test aggregating empty contributions."""
        aggregated = coordinator.aggregate_contributions({}, temperature=0.5)
        
        assert np.all(aggregated == 0)
    
    def test_diffusion_step(self, coordinator):
        """Test single diffusion step."""
        state = coordinator.initialize_diffusion("test_task")
        
        new_state = coordinator.diffusion_step(state, temperature=0.5)
        
        assert new_state.current_step == state.current_step + 1
        assert new_state.temperature == 0.5
        assert len(new_state.agent_contributions) > 0
        assert new_state.stage == DenoisingStage.DIFFUSION
    
    def test_diffusion_step_updates_active_state(self, coordinator):
        """Test that diffusion step updates the active state."""
        state = coordinator.initialize_diffusion("test_task")
        new_state = coordinator.diffusion_step(state, temperature=0.5)
        
        # Active state should be updated
        assert coordinator.active_states["test_task"] == new_state
    
    def test_refine(self, coordinator):
        """Test refinement phase."""
        state = coordinator.initialize_diffusion("test_task")
        
        # Do some diffusion steps
        for temp in coordinator.get_temperature_schedule():
            state = coordinator.diffusion_step(state, temp)
        
        # Refine
        refined = coordinator.refine(state)
        
        assert refined.stage == DenoisingStage.COMPLETE
        assert refined.temperature == coordinator.config.final_temperature * 0.5
    
    def test_refine_convergence(self, coordinator):
        """Test refinement with convergence."""
        config = CoordinatorConfig(
            num_steps=3,
            convergence_threshold=0.1,  # High threshold for quick convergence
        )
        coord = DiffusionCoordinator(environment=coordinator.environment, config=config)
        
        state = coord.initialize_diffusion("test_task")
        
        # Do diffusion steps
        for temp in coord.get_temperature_schedule():
            state = coord.diffusion_step(state, temp)
        
        refined = coord.refine(state)
        
        assert refined.stage == DenoisingStage.COMPLETE
    
    def test_run_diffusion(self, coordinator):
        """Test running complete diffusion process."""
        state = coordinator.run_diffusion("complete_task", "Test prompt")
        
        assert state.current_step == coordinator.config.num_steps
        assert state.stage == DenoisingStage.COMPLETE
        assert state.temperature < coordinator.config.initial_temperature
    
    def test_run_parallel_diffusion(self, coordinator):
        """Test running parallel diffusion processes."""
        task_ids = ["task_1", "task_2", "task_3"]
        
        results = coordinator.run_parallel_diffusion(task_ids)
        
        assert len(results) == 3
        for task_id in task_ids:
            assert task_id in results
            assert results[task_id].stage == DenoisingStage.COMPLETE
    
    def test_get_state(self, coordinator):
        """Test getting state by task ID."""
        coordinator.initialize_diffusion("test_task")
        
        state = coordinator.get_state("test_task")
        assert state is not None
        assert state.metadata.get("task_id") == "test_task"
    
    def test_get_state_nonexistent(self, coordinator):
        """Test getting state for nonexistent task."""
        state = coordinator.get_state("nonexistent")
        assert state is None
    
    def test_clear_state(self, coordinator):
        """Test clearing a diffusion state."""
        coordinator.initialize_diffusion("test_task")
        
        result = coordinator.clear_state("test_task")
        
        assert result == True
        assert "test_task" not in coordinator.active_states
    
    def test_clear_state_nonexistent(self, coordinator):
        """Test clearing nonexistent state."""
        result = coordinator.clear_state("nonexistent")
        assert result == False
    
    def test_vector_to_response(self, coordinator):
        """Test converting state to response."""
        state = coordinator.run_diffusion("test_task")
        
        response = coordinator.vector_to_response(state)
        
        assert "task_id" in response
        assert "status" in response
        assert "final_entropy" in response
        assert "agent_influences" in response
        assert response["status"] == "complete"
    
    def test_get_coordinator_stats(self, coordinator):
        """Test getting coordinator statistics."""
        coordinator.initialize_diffusion("task_1")
        coordinator.initialize_diffusion("task_2")
        
        stats = coordinator.get_coordinator_stats()
        
        assert stats["num_active_diffusions"] == 2
        assert stats["num_agents"] == 3
        assert "capacity_weights" in stats
        assert "config" in stats


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_environment(self):
        """Test coordinator with empty environment."""
        env = MultiAgentEnvironment()
        coordinator = DiffusionCoordinator(environment=env)
        
        # Should handle gracefully
        state = coordinator.run_diffusion("test_task")
        
        assert state is not None
        assert state.stage == DenoisingStage.COMPLETE
    
    def test_single_agent(self):
        """Test coordinator with single agent."""
        env = MultiAgentEnvironment()
        env.register_agent("only_agent", Mock(), capacity=0.5)
        
        coordinator = DiffusionCoordinator(environment=env)
        state = coordinator.run_diffusion("test_task")
        
        assert state.stage == DenoisingStage.COMPLETE
        # Single agent should have all weight
        assert coordinator.get_capacity_weight("only_agent") == 1.0
    
    def test_zero_capacity_agents(self):
        """Test coordinator with zero capacity agents."""
        env = MultiAgentEnvironment()
        env.register_agent("agent_1", Mock(), capacity=0.0)
        env.register_agent("agent_2", Mock(), capacity=0.0)
        
        coordinator = DiffusionCoordinator(environment=env)
        
        # Should use min_capacity_weight
        coordinator._update_capacity_weights()
        
        total_weight = sum(coordinator._capacity_weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def test_very_high_capacity_agent(self):
        """Test coordinator with very high capacity agent."""
        env = MultiAgentEnvironment()
        env.register_agent("expert", Mock(), capacity=100.0)
        env.register_agent("novice", Mock(), capacity=0.1)
        
        coordinator = DiffusionCoordinator(environment=env)
        coordinator._update_capacity_weights()
        
        # Expert should have much higher weight
        expert_weight = coordinator.get_capacity_weight("expert")
        novice_weight = coordinator.get_capacity_weight("novice")
        
        assert expert_weight > novice_weight
    
    def test_custom_vector_dim(self):
        """Test coordinator with custom vector dimension."""
        env = MultiAgentEnvironment()
        env.register_agent("agent", Mock(), capacity=0.5)
        
        config = CoordinatorConfig(vector_dim=128)
        coordinator = DiffusionCoordinator(environment=env, config=config)
        
        state = coordinator.initialize_diffusion("test")
        
        assert len(state.response_vector) == 128
    
    def test_zero_entropy_coef(self):
        """Test coordinator with zero entropy coefficient."""
        env = MultiAgentEnvironment()
        env.register_agent("agent", Mock(), capacity=0.5)
        
        config = CoordinatorConfig(entropy_coef=0.0)
        coordinator = DiffusionCoordinator(environment=env, config=config)
        
        state = coordinator.run_diffusion("test")
        
        assert state.stage == DenoisingStage.COMPLETE
    
    def test_single_step_diffusion(self):
        """Test diffusion with only one step."""
        env = MultiAgentEnvironment()
        env.register_agent("agent", Mock(), capacity=0.5)
        
        config = CoordinatorConfig(num_steps=1)
        coordinator = DiffusionCoordinator(environment=env, config=config)
        
        state = coordinator.run_diffusion("test")
        
        assert state.current_step == 1
        assert state.stage == DenoisingStage.COMPLETE


class TestIntegration:
    """Integration tests for coordinator with environment."""
    
    def test_with_reasoning_agents(self):
        """Test coordinator with actual ReasoningAgent instances."""
        env = MultiAgentEnvironment()
        
        # Create real agents
        agent1 = ReasoningAgent()
        agent2 = ReasoningAgent()
        
        env.register_agent("agent_1", agent1, capacity=0.8)
        env.register_agent("agent_2", agent2, capacity=0.6)
        
        coordinator = DiffusionCoordinator(environment=env)
        
        state = coordinator.run_diffusion("test_task", "Analyze this problem")
        
        assert state.stage == DenoisingStage.COMPLETE
    
    def test_capacity_weighted_influence(self):
        """Test that higher capacity agents have more influence."""
        env = MultiAgentEnvironment()
        
        env.register_agent("high_cap", Mock(), capacity=0.95)
        env.register_agent("low_cap", Mock(), capacity=0.1)
        
        coordinator = DiffusionCoordinator(environment=env)
        state = coordinator.initialize_diffusion("test")
        
        contributions = coordinator.collect_agent_contributions(state)
        
        # High capacity agent should have larger contribution magnitude
        high_contrib = np.linalg.norm(contributions["high_cap"])
        low_contrib = np.linalg.norm(contributions["low_cap"])
        
        # Due to capacity weighting
        assert high_contrib >= low_contrib
    
    def test_multiple_sequential_diffusions(self):
        """Test running multiple diffusion processes sequentially."""
        env = create_collaboration_environment(num_workers=2)
        coordinator = DiffusionCoordinator(environment=env)
        
        # Run multiple diffusions
        for i in range(5):
            state = coordinator.run_diffusion(f"task_{i}")
            assert state.stage == DenoisingStage.COMPLETE
            
            # Clear after each
            coordinator.clear_state(f"task_{i}")
        
        assert len(coordinator.active_states) == 0
    
    def test_diffusion_with_temperature_schedule(self):
        """Test that temperature decreases during diffusion."""
        env = MultiAgentEnvironment()
        env.register_agent("agent", Mock(), capacity=0.5)
        
        config = CoordinatorConfig(
            num_steps=10,
            initial_temperature=1.0,
            final_temperature=0.1,
        )
        coordinator = DiffusionCoordinator(environment=env, config=config)
        
        temperatures = []
        state = coordinator.initialize_diffusion("test")
        
        for temp in coordinator.get_temperature_schedule():
            state = coordinator.diffusion_step(state, temp)
            temperatures.append(state.temperature)
        
        # Temperatures should generally decrease
        assert temperatures[0] >= temperatures[-1]
    
    def test_parallel_diffusion_independence(self):
        """Test that parallel diffusions are independent."""
        env = MultiAgentEnvironment()
        env.register_agent("agent", Mock(), capacity=0.5)
        
        coordinator = DiffusionCoordinator(environment=env)
        
        results = coordinator.run_parallel_diffusion(["task_1", "task_2"])
        
        # Each should have different final states (due to random noise)
        vec1 = results["task_1"].response_vector
        vec2 = results["task_2"].response_vector
        
        # Should be different due to random initialization
        assert not np.allclose(vec1, vec2)


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_coordinator_default(self):
        """Test creating coordinator with defaults."""
        coordinator = create_coordinator()
        
        assert coordinator is not None
        assert coordinator.config.num_steps == 10
        assert coordinator.environment is not None
    
    def test_create_coordinator_custom(self):
        """Test creating coordinator with custom settings."""
        env = MultiAgentEnvironment()
        coordinator = create_coordinator(
            environment=env,
            num_steps=20,
            entropy_coef=0.5,
        )
        
        assert coordinator.environment == env
        assert coordinator.config.num_steps == 20
        assert coordinator.config.entropy_coef == 0.5
    
    def test_create_coordinator_creates_env(self):
        """Test that factory creates environment if not provided."""
        coordinator = create_coordinator()
        
        assert coordinator.environment is not None
        assert len(coordinator.environment.agents) > 0
