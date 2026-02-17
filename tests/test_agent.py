"""Tests for agent module."""

import pytest
from src.agent import AdaptiveWebAgent, AgentConfig, Step


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.max_steps == 50
        assert config.max_retries == 3
        assert config.uncertainty_threshold == 0.7
        assert config.trajectory_pruning_threshold == 0.5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            max_steps=100,
            max_retries=5,
            debug=True,
        )
        assert config.max_steps == 100
        assert config.max_retries == 5
        assert config.debug is True


class TestStep:
    """Tests for Step dataclass."""
    
    def test_step_creation(self):
        """Test creating a Step."""
        step = Step(index=0, action="navigate", observation="page loaded")
        assert step.index == 0
        assert step.action == "navigate"
        assert step.observation == "page loaded"
        assert step.is_pruned is False
    
    def test_step_to_dict(self):
        """Test converting Step to dictionary."""
        step = Step(index=1, action="click", thought="click button")
        step_dict = step.to_dict()
        assert step_dict["index"] == 1
        assert step_dict["action"] == "click"
        assert step_dict["is_pruned"] is False


class TestAdaptiveWebAgent:
    """Tests for AdaptiveWebAgent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = AdaptiveWebAgent()
        assert agent.current_step == 0
        assert agent.is_running is False
        assert len(agent.trajectory) == 0
    
    def test_agent_with_config(self):
        """Test agent with custom config."""
        config = AgentConfig(max_steps=10)
        agent = AdaptiveWebAgent(config)
        assert agent.config.max_steps == 10
    
    def test_step_execution(self):
        """Test executing a single step."""
        agent = AdaptiveWebAgent()
        step = agent.step("navigate")
        assert step.index == 0
        assert step.action == "navigate"
        assert len(agent.trajectory) == 1
    
    def test_get_trajectory(self):
        """Test getting trajectory."""
        agent = AdaptiveWebAgent()
        agent.step("action1")
        agent.step("action2")
        trajectory = agent.get_trajectory()
        assert len(trajectory) == 2
    
    def test_prune_trajectory(self):
        """Test trajectory pruning."""
        config = AgentConfig(trajectory_pruning_threshold=0.6)
        agent = AdaptiveWebAgent(config)
        
        step1 = agent.step("action1")
        step1.uncertainty = 0.3  # Below threshold - should prune
        step2 = agent.step("action2")
        step2.uncertainty = 0.8  # Above threshold - keep
        
        pruned = agent.prune_trajectory()
        assert pruned == 1
        assert step1.is_pruned is True
        assert step2.is_pruned is False
    
    def test_reset(self):
        """Test agent reset."""
        agent = AdaptiveWebAgent()
        agent.step("action1")
        agent.step("action2")
        agent.reset()
        assert len(agent.trajectory) == 0
        assert agent.current_step == 0
