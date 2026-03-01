"""
Tests for the Information Capacity Estimator.

Tests cover:
- Capacity estimation with mock agents
- Agent ranking
- Different estimation methods
- Edge cases and error handling
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capacity import CapacityEstimator, CapacityResult


# =============================================================================
# Mock Agents for Testing
# =============================================================================

class MockAgent:
    """Simple mock agent for testing."""
    
    def __init__(self, responses: list, name: str = "mock"):
        self.responses = responses
        self.name = name
        self.call_count = 0
    
    def generate(self, context):
        """Return next response from list, cycling if needed."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def __repr__(self):
        return f"MockAgent({self.name})"


class CallableMockAgent:
    """Mock agent that is callable."""
    
    def __init__(self, response: str = "default response"):
        self.response = response
    
    def __call__(self, context):
        return self.response


class RespondMethodAgent:
    """Mock agent with respond() method."""
    
    def __init__(self, response: str = "responded"):
        self.response = response
    
    def respond(self, context):
        return self.response


class FailingAgent:
    """Mock agent that raises errors."""
    
    def generate(self, context):
        raise RuntimeError("Agent failed")


class StringOnlyAgent:
    """Mock agent that only has string representation."""
    pass


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_task():
    """Simple task context for testing."""
    return {
        "query": "What is machine learning?",
        "keywords": ["machine", "learning", "AI", "algorithm"],
        "domain": "technology"
    }


@pytest.fixture
def diverse_agent():
    """Agent with diverse responses."""
    return MockAgent([
        "Machine learning is a subset of artificial intelligence.",
        "ML algorithms learn patterns from data.",
        "AI and machine learning are transforming technology.",
        "Deep learning is a type of machine learning algorithm."
    ], name="diverse")


@pytest.fixture
def repetitive_agent():
    """Agent with repetitive responses."""
    return MockAgent([
        "Yes.",
        "Yes.",
        "Yes.",
        "Yes."
    ], name="repetitive")


@pytest.fixture
def estimator():
    """Default CapacityEstimator instance."""
    return CapacityEstimator(num_samples=4)


# =============================================================================
# CapacityResult Tests
# =============================================================================

def test_capacity_result_creation():
    """Test basic CapacityResult creation."""
    result = CapacityResult(
        capacity_bits=5.5,
        method="entropy",
        confidence=0.8,
        timestamp=time.time()
    )
    assert result.capacity_bits == 5.5
    assert result.method == "entropy"
    assert result.confidence == 0.8


def test_capacity_result_invalid_bits():
    """Test that negative capacity_bits raises ValueError."""
    with pytest.raises(ValueError, match="capacity_bits must be non-negative"):
        CapacityResult(
            capacity_bits=-1.0,
            method="entropy",
            confidence=0.5,
            timestamp=time.time()
        )


def test_capacity_result_invalid_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Invalid method"):
        CapacityResult(
            capacity_bits=1.0,
            method="invalid_method",
            confidence=0.5,
            timestamp=time.time()
        )


def test_capacity_result_invalid_confidence():
    """Test that confidence outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="confidence must be between"):
        CapacityResult(
            capacity_bits=1.0,
            method="entropy",
            confidence=1.5,
            timestamp=time.time()
        )
    
    with pytest.raises(ValueError, match="confidence must be between"):
        CapacityResult(
            capacity_bits=1.0,
            method="entropy",
            confidence=-0.1,
            timestamp=time.time()
        )


# =============================================================================
# CapacityEstimator Initialization Tests
# =============================================================================

def test_estimator_default_init():
    """Test default estimator initialization."""
    estimator = CapacityEstimator()
    assert estimator.default_method == "combined"
    assert estimator.num_samples == 10
    assert estimator.random_seed is None


def test_estimator_custom_init():
    """Test custom estimator initialization."""
    estimator = CapacityEstimator(
        default_method="entropy",
        num_samples=5,
        random_seed=42
    )
    assert estimator.default_method == "entropy"
    assert estimator.num_samples == 5
    assert estimator.random_seed == 42


def test_estimator_invalid_method():
    """Test that invalid default method raises ValueError."""
    with pytest.raises(ValueError, match="Invalid method"):
        CapacityEstimator(default_method="invalid")


# =============================================================================
# Entropy-based Estimation Tests
# =============================================================================

def test_entropy_estimation_diverse_agent(estimator, simple_task, diverse_agent):
    """Test entropy estimation with diverse agent."""
    result = estimator.estimate_capacity(diverse_agent, simple_task, method="entropy")
    assert isinstance(result, CapacityResult)
    assert result.method == "entropy"
    assert result.capacity_bits > 0
    assert 0 <= result.confidence <= 1


def test_entropy_estimation_repetitive_agent(estimator, simple_task, repetitive_agent):
    """Test entropy estimation with repetitive agent."""
    result = estimator.estimate_capacity(repetitive_agent, simple_task, method="entropy")
    # Repetitive agent should have low entropy
    assert result.capacity_bits >= 0
    assert result.method == "entropy"


def test_entropy_estimation_empty_responses(estimator, simple_task):
    """Test entropy estimation with agent returning empty strings."""
    empty_agent = MockAgent(["", "", "", ""])
    result = estimator.estimate_capacity(empty_agent, simple_task, method="entropy")
    assert result.capacity_bits == 0.0


def test_entropy_estimation_failing_agent(estimator, simple_task):
    """Test entropy estimation with failing agent."""
    failing_agent = FailingAgent()
    result = estimator.estimate_capacity(failing_agent, simple_task, method="entropy")
    # Failing agent should return zero capacity
    assert result.capacity_bits == 0.0
    assert result.confidence == 0.0


# =============================================================================
# Mutual Information Estimation Tests
# =============================================================================

def test_mutual_info_estimation(estimator, simple_task, diverse_agent):
    """Test mutual information estimation."""
    result = estimator.estimate_capacity(diverse_agent, simple_task, method="mutual_info")
    assert isinstance(result, CapacityResult)
    assert result.method == "mutual_info"
    assert result.capacity_bits >= 0
    assert 0 <= result.confidence <= 1


def test_mutual_info_with_keywords(estimator):
    """Test MI estimation with task keywords."""
    task = {
        "query": "test query",
        "keywords": ["machine", "learning", "AI"]
    }
    agent = MockAgent([
        "machine learning is great",
        "AI and machine learning work together",
        "learning algorithms are useful"
    ])
    result = estimator.estimate_capacity(agent, task, method="mutual_info")
    assert result.capacity_bits >= 0


def test_mutual_info_no_keywords(estimator):
    """Test MI estimation without keywords in task."""
    task = {"query": "simple question"}
    agent = MockAgent(["response one", "response two"])
    result = estimator.estimate_capacity(agent, task, method="mutual_info")
    assert result.capacity_bits >= 0


# =============================================================================
# Combined Estimation Tests
# =============================================================================

def test_combined_estimation(estimator, simple_task, diverse_agent):
    """Test combined estimation method."""
    result = estimator.estimate_capacity(diverse_agent, simple_task, method="combined")
    assert result.method == "combined"
    assert result.capacity_bits >= 0


def test_default_method_combined(simple_task, diverse_agent):
    """Test that default method is combined."""
    estimator = CapacityEstimator(num_samples=4)
    result = estimator.estimate_capacity(diverse_agent, simple_task)
    assert result.method == "combined"


# =============================================================================
# Agent Interface Tests
# =============================================================================

def test_agent_with_generate_method(simple_task):
    """Test agent with generate() method."""
    agent = MockAgent(["response"])
    estimator = CapacityEstimator(num_samples=1)
    result = estimator.estimate_capacity(agent, simple_task, method="entropy")
    assert result.capacity_bits >= 0


def test_agent_callable(simple_task):
    """Test callable agent."""
    agent = CallableMockAgent("callable response")
    estimator = CapacityEstimator(num_samples=1)
    result = estimator.estimate_capacity(agent, simple_task, method="entropy")
    assert result.capacity_bits >= 0


def test_agent_with_respond_method(simple_task):
    """Test agent with respond() method."""
    agent = RespondMethodAgent("responded content")
    estimator = CapacityEstimator(num_samples=1)
    result = estimator.estimate_capacity(agent, simple_task, method="entropy")
    assert result.capacity_bits >= 0


def test_agent_string_only(simple_task):
    """Test agent with no methods, just string representation."""
    agent = StringOnlyAgent()
    estimator = CapacityEstimator(num_samples=1)
    result = estimator.estimate_capacity(agent, simple_task, method="entropy")
    assert result.capacity_bits >= 0


# =============================================================================
# Agent Ranking Tests
# =============================================================================

def test_rank_agents_basic(estimator, simple_task):
    """Test basic agent ranking."""
    agents = [
        ("diverse", MockAgent([
            "Machine learning uses algorithms.",
            "AI systems learn from data.",
            "Neural networks process information.",
            "Deep learning models are powerful."
        ])),
        ("simple", MockAgent(["Yes.", "No.", "Maybe.", "OK."]))
    ]
    
    ranked = estimator.rank_agents(agents, simple_task)
    
    assert len(ranked) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked)
    assert all(isinstance(r[1], float) for r in ranked)
    
    # Check descending order
    assert ranked[0][1] >= ranked[1][1]


def test_rank_agents_empty_list(estimator, simple_task):
    """Test ranking with empty agent list."""
    ranked = estimator.rank_agents([], simple_task)
    assert ranked == []


def test_rank_agents_single(estimator, simple_task):
    """Test ranking with single agent."""
    agents = [("only", MockAgent(["Only response"]))]
    ranked = estimator.rank_agents(agents, simple_task)
    assert len(ranked) == 1
    assert ranked[0][0] == "only"


def test_rank_agents_with_results(estimator, simple_task):
    """Test ranking with full results."""
    agents = [
        ("agent_a", MockAgent(["response A"])),
        ("agent_b", MockAgent(["response B"]))
    ]
    
    ranked = estimator.rank_agents_with_results(agents, simple_task)
    
    assert len(ranked) == 2
    assert all(isinstance(r[1], CapacityResult) for r in ranked)
    
    # Check descending order by capacity
    assert ranked[0][1].capacity_bits >= ranked[1][1].capacity_bits


def test_rank_agents_custom_method(estimator, simple_task):
    """Test ranking with custom estimation method."""
    agents = [
        ("agent_a", MockAgent(["A"])),
        ("agent_b", MockAgent(["B"]))
    ]
    
    ranked = estimator.rank_agents(agents, simple_task, method="entropy")
    assert len(ranked) == 2


# =============================================================================
# Edge Cases and Special Scenarios
# =============================================================================

def test_estimate_capacity_with_none_context(estimator):
    """Test estimation with minimal context."""
    agent = MockAgent(["response"])
    # Should handle empty context gracefully
    result = estimator.estimate_capacity(agent, {}, method="entropy")
    assert result.capacity_bits >= 0


def test_timestamp_is_recent(estimator, simple_task, diverse_agent):
    """Test that timestamp is recent."""
    before = time.time()
    result = estimator.estimate_capacity(diverse_agent, simple_task)
    after = time.time()
    
    assert before <= result.timestamp <= after


def test_different_methods_produce_different_results(estimator, simple_task, diverse_agent):
    """Test that different methods can produce different results."""
    entropy_result = estimator.estimate_capacity(diverse_agent, simple_task, method="entropy")
    mi_result = estimator.estimate_capacity(diverse_agent, simple_task, method="mutual_info")
    combined_result = estimator.estimate_capacity(diverse_agent, simple_task, method="combined")
    
    # All should have valid results
    assert entropy_result.method == "entropy"
    assert mi_result.method == "mutual_info"
    assert combined_result.method == "combined"
    
    # Combined should be average of entropy and MI
    # (approximately, due to floating point)
    expected_combined = 0.5 * entropy_result.capacity_bits + 0.5 * mi_result.capacity_bits
    assert abs(combined_result.capacity_bits - expected_combined) < 1e-9


def test_num_samples_affects_confidence(simple_task):
    """Test that more samples can affect confidence."""
    low_sample_estimator = CapacityEstimator(num_samples=1)
    high_sample_estimator = CapacityEstimator(num_samples=20)
    
    agent = MockAgent([
        "Machine learning is fascinating.",
        "AI transforms industries.",
        "Algorithms process data.",
        "Neural networks learn patterns."
    ] * 5)  # Repeat to have enough responses
    
    low_result = low_sample_estimator.estimate_capacity(agent, simple_task)
    high_result = high_sample_estimator.estimate_capacity(agent, simple_task)
    
    # Both should have valid results
    assert low_result.confidence >= 0
    assert high_result.confidence >= 0


def test_imports_from_init():
    """Test that imports work from package __init__."""
    from src import CapacityEstimator, CapacityResult
    assert CapacityEstimator is not None
    assert CapacityResult is not None
