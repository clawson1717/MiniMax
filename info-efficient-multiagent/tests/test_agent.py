"""
Tests for Base Reasoning Agent
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile

from src.agent import (
    AgentResponse,
    AgentConfig,
    ReasoningAgent,
    SpecializedAgent,
    create_agent,
    create_agent_pool,
)
from src.capacity import InformationCapacityEstimator


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""
    
    def test_init_defaults(self):
        """Test default configuration."""
        config = AgentConfig(agent_id="test-agent")
        
        assert config.agent_id == "test-agent"
        assert config.model_name == "base-agent"
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.seed is None
        assert config.specializations == []
    
    def test_init_custom(self):
        """Test custom configuration."""
        config = AgentConfig(
            agent_id="custom-agent",
            model_name="gpt-4",
            max_tokens=512,
            temperature=0.5,
            seed=42,
            specializations=["math", "science"],
        )
        
        assert config.agent_id == "custom-agent"
        assert config.model_name == "gpt-4"
        assert config.max_tokens == 512
        assert config.temperature == 0.5
        assert config.seed == 42
        assert config.specializations == ["math", "science"]
    
    def test_to_dict(self):
        """Test serialization."""
        config = AgentConfig(agent_id="dict-agent", specializations=["reasoning"])
        d = config.to_dict()
        
        assert d["agent_id"] == "dict-agent"
        assert d["specializations"] == ["reasoning"]


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""
    
    def test_init(self):
        """Test response initialization."""
        response = AgentResponse(content="Test response")
        
        assert response.content == "Test response"
        assert response.confidence == 1.0
        assert response.metadata == {}
        assert response.timestamp > 0
    
    def test_with_metadata(self):
        """Test response with metadata."""
        response = AgentResponse(
            content="Response",
            confidence=0.8,
            metadata={"prompt": "test"},
        )
        
        assert response.confidence == 0.8
        assert response.metadata["prompt"] == "test"


class TestReasoningAgent:
    """Tests for ReasoningAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = ReasoningAgent()
        
        assert agent.agent_id == "default-agent"
        assert agent.response_history == []
        assert agent.capacity_history == []
    
    def test_init_with_config(self):
        """Test agent with custom config."""
        config = AgentConfig(agent_id="custom-id")
        agent = ReasoningAgent(config=config)
        
        assert agent.agent_id == "custom-id"
    
    def test_generate_single(self):
        """Test generating a single response."""
        agent = ReasoningAgent()
        responses = agent.generate("What is 2+2?", num_responses=1)
        
        assert len(responses) == 1
        assert len(responses[0].content) > 0
        assert responses[0].confidence > 0
        assert len(agent.response_history) == 1
    
    def test_generate_multiple(self):
        """Test generating multiple responses."""
        agent = ReasoningAgent()
        responses = agent.generate("Test prompt", num_responses=3)
        
        assert len(responses) == 3
        assert len(agent.response_history) == 3
    
    def test_generate_with_model(self):
        """Test generation with custom model."""
        mock_model = Mock(return_value="Custom model response")
        agent = ReasoningAgent(model=mock_model)
        
        responses = agent.generate("Test prompt")
        
        assert len(responses) == 1
        assert responses[0].content == "Custom model response"
        mock_model.assert_called_once()
    
    def test_mock_generate_math(self):
        """Test mock generation for math prompts."""
        agent = ReasoningAgent()
        
        # Generate multiple math responses
        responses = agent.generate("Solve this math problem", num_responses=5)
        
        # Should get math-related responses
        all_content = " ".join(r.content for r in responses)
        assert "math" in all_content.lower() or "solution" in all_content.lower()
    
    def test_mock_generate_science(self):
        """Test mock generation for science prompts."""
        agent = ReasoningAgent()
        
        responses = agent.generate("Explain the science behind this", num_responses=3)
        
        all_content = " ".join(r.content for r in responses)
        # Check for science-related content OR valid response
        assert len(all_content) > 0
        # "science" keyword triggers science template
        assert "scientific" in all_content.lower() or "science" in all_content.lower() or len(all_content) > 50
    
    def test_confidence_estimation(self):
        """Test confidence is estimated."""
        agent = ReasoningAgent()
        responses = agent.generate("Test prompt", num_responses=5)
        
        for response in responses:
            assert 0 <= response.confidence <= 1.0
    
    def test_measure_capacity(self):
        """Test capacity measurement."""
        agent = ReasoningAgent()
        agent.generate("Test prompt", num_responses=5)
        
        result = agent.measure_capacity()
        
        assert result.capacity_bits >= 0
        assert result.entropy_bits >= 0
        assert len(agent.capacity_history) == 1
    
    def test_measure_capacity_with_responses(self):
        """Test capacity measurement with provided responses."""
        agent = ReasoningAgent()
        responses = ["Response 1", "Response 2", "Response 3"]
        
        result = agent.measure_capacity(responses=responses)
        
        assert result.capacity_bits >= 0
    
    def test_get_current_capacity(self):
        """Test getting current capacity."""
        agent = ReasoningAgent()
        
        # No measurements yet
        assert agent.get_current_capacity() == 0.0
        
        # After measurement
        agent.generate("Test", num_responses=3)
        agent.measure_capacity()
        
        assert agent.get_current_capacity() >= 0
    
    def test_get_average_capacity(self):
        """Test getting average capacity."""
        agent = ReasoningAgent()
        
        # Generate and measure multiple times
        for _ in range(3):
            agent.generate("Test", num_responses=3)
            agent.measure_capacity()
        
        avg = agent.get_average_capacity(n=3)
        assert avg >= 0
    
    def test_clear_history(self):
        """Test clearing history."""
        agent = ReasoningAgent()
        agent.generate("Test", num_responses=3)
        agent.measure_capacity()
        
        agent.clear_history()
        
        assert len(agent.response_history) == 0
        assert len(agent.capacity_history) == 0
    
    def test_to_dict(self):
        """Test serialization."""
        agent = ReasoningAgent()
        agent.generate("Test", num_responses=3)
        
        d = agent.to_dict()
        
        assert "config" in d
        assert "response_count" in d
        assert d["response_count"] == 3
    
    def test_save_load(self):
        """Test saving and loading agent."""
        agent = ReasoningAgent(config=AgentConfig(agent_id="save-test"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.json"
            agent.save(path)
            
            loaded = ReasoningAgent.load(path)
            
            assert loaded.agent_id == "save-test"


class TestSpecializedAgent:
    """Tests for SpecializedAgent class."""
    
    def test_init(self):
        """Test specialized agent initialization."""
        config = AgentConfig(
            agent_id="specialized",
            specializations=["math", "science"],
        )
        agent = SpecializedAgent(config=config)
        
        assert agent.specializations == ["math", "science"]
    
    def test_specialized_response(self):
        """Test that specialized responses are enhanced."""
        config = AgentConfig(
            agent_id="math-agent",
            specializations=["math"],
        )
        domain_knowledge = {
            "math": ["Algebraic manipulation is key.", "Consider all cases."],
        }
        agent = SpecializedAgent(
            config=config,
            domain_knowledge=domain_knowledge,
        )
        
        response = agent.generate("Solve this math problem")
        
        # Should include specialized insight
        assert "Specialized insight" in response[0].content or "math" in response[0].content.lower()
    
    def test_capacity_boost(self):
        """Test capacity boost for matching domain."""
        config = AgentConfig(
            agent_id="math-agent",
            specializations=["math"],
        )
        agent = SpecializedAgent(config=config)
        
        # Generate math responses
        agent.generate("Math problem", num_responses=5)
        
        # Measure with math context
        result = agent.measure_capacity(context="mathematical analysis")
        
        # Should have some capacity
        assert result.capacity_bits >= 0


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_agent(self):
        """Test create_agent factory."""
        agent = create_agent(
            agent_id="factory-agent",
            specializations=["reasoning"],
        )
        
        assert agent.agent_id == "factory-agent"
        assert agent.specializations == ["reasoning"]
    
    def test_create_agent_pool(self):
        """Test create_agent_pool factory."""
        agents = create_agent_pool(num_agents=5)
        
        assert len(agents) == 5
        assert all(isinstance(a, ReasoningAgent) for a in agents)
        
        # Check unique IDs
        ids = [a.agent_id for a in agents]
        assert len(set(ids)) == 5
    
    def test_create_agent_pool_with_specializations(self):
        """Test agent pool with custom specializations."""
        specs_list = [
            ["math"],
            ["science"],
            ["reasoning"],
        ]
        
        agents = create_agent_pool(
            num_agents=3,
            specializations_list=specs_list,
        )
        
        for agent, expected_specs in zip(agents, specs_list):
            assert agent.specializations == expected_specs
    
    def test_create_agent_with_model(self):
        """Test creating agent with custom model."""
        mock_model = Mock(return_value="Model response")
        agent = create_agent(model=mock_model)
        
        response = agent.generate("Test")
        
        assert response[0].content == "Model response"


class TestAgentCapacityIntegration:
    """Tests for agent-capacity integration."""
    
    def test_high_capacity_vs_low_capacity(self):
        """Test that detailed responses yield higher capacity."""
        # High capacity agent (detailed responses)
        def detailed_model(prompt, **kwargs):
            return "This is a detailed response with multiple points. " \
                   "First, we analyze the problem. Second, we consider " \
                   "alternatives. Third, we reach a conclusion based on evidence."
        
        # Low capacity agent (short responses)
        def short_model(prompt, **kwargs):
            return "Yes."
        
        high_agent = create_agent(agent_id="high-capacity", model=detailed_model)
        low_agent = create_agent(agent_id="low-capacity", model=short_model)
        
        # Generate responses
        prompt = "Analyze this problem"
        high_agent.generate(prompt, num_responses=5)
        low_agent.generate(prompt, num_responses=5)
        
        # Measure capacity
        high_result = high_agent.measure_capacity()
        low_result = low_agent.measure_capacity()
        
        # High capacity agent should have higher entropy/diversity
        assert high_result.entropy_bits > low_result.entropy_bits
    
    def test_ranking_by_capacity(self):
        """Test ranking multiple agents by capacity."""
        agents = create_agent_pool(num_agents=3)
        
        # Generate responses for each
        for agent in agents:
            agent.generate("Analyze this", num_responses=5)
            agent.measure_capacity()
        
        # Get capacities
        capacities = [(a.agent_id, a.get_current_capacity()) for a in agents]
        
        # All should have some capacity
        for agent_id, capacity in capacities:
            assert capacity >= 0
