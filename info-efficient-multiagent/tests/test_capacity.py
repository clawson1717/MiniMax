"""
Tests for Information Capacity Estimator.
"""

import pytest
import math
from src.capacity import (
    InformationCapacityEstimator,
    CapacityResult,
    create_estimator,
)


class TestCapacityResult:
    """Tests for CapacityResult dataclass."""
    
    def test_creation(self):
        """Test creating a capacity result."""
        result = CapacityResult(
            capacity_bits=5.5,
            entropy_bits=4.2,
            mutual_info_bits=1.3,
            response_diversity=0.8,
            confidence=0.9,
        )
        
        assert result.capacity_bits == 5.5
        assert result.entropy_bits == 4.2
        assert result.mutual_info_bits == 1.3
        assert result.response_diversity == 0.8
        assert result.confidence == 0.9
    
    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        result = CapacityResult(
            capacity_bits=1.0,
            entropy_bits=1.0,
            mutual_info_bits=0.0,
            response_diversity=1.0,
            confidence=1.0,
        )
        
        assert result.metadata == {}


class TestInformationCapacityEstimator:
    """Tests for InformationCapacityEstimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = InformationCapacityEstimator()
        
        assert estimator.method == "combined"
        assert estimator.num_samples == 10
        assert estimator.base == 2.0
    
    def test_custom_initialization(self):
        """Test estimator with custom parameters."""
        estimator = InformationCapacityEstimator(
            method="entropy",
            num_samples=20,
            base=math.e,
        )
        
        assert estimator.method == "entropy"
        assert estimator.num_samples == 20
        assert estimator.base == math.e
    
    def test_empty_responses(self):
        """Test with empty response list."""
        estimator = InformationCapacityEstimator()
        result = estimator.estimate_capacity([])
        
        assert result.capacity_bits == 0.0
        assert result.entropy_bits == 0.0
        assert result.confidence == 0.0
    
    def test_single_response(self):
        """Test with a single response."""
        estimator = InformationCapacityEstimator()
        result = estimator.estimate_capacity(["This is a test response"])
        
        assert result.capacity_bits > 0
        assert result.entropy_bits >= 0
        assert result.confidence == 0.1  # 1/10 samples
    
    def test_multiple_responses(self):
        """Test with multiple responses."""
        estimator = InformationCapacityEstimator()
        responses = [
            "The answer is 42",
            "The answer is forty-two",
            "The result is 42",
            "42 is the answer",
        ]
        result = estimator.estimate_capacity(responses)
        
        assert result.capacity_bits > 0
        assert result.entropy_bits > 0
        assert result.response_diversity > 0
    
    def test_entropy_computation(self):
        """Test entropy computation."""
        estimator = InformationCapacityEstimator()
        
        # High entropy: diverse vocabulary
        high_entropy = estimator._compute_entropy([
            "completely different words here",
            "another set of unique terms",
            "yet more distinct vocabulary",
        ])
        
        # Low entropy: repeated words
        low_entropy = estimator._compute_entropy([
            "test test test test test",
            "test test test test test",
            "test test test test test",
        ])
        
        assert high_entropy > low_entropy
    
    def test_diversity_computation(self):
        """Test diversity computation."""
        estimator = InformationCapacityEstimator()
        
        # All unique
        high_diversity = estimator._compute_diversity([
            "response one",
            "response two",
            "response three",
        ])
        
        # All same
        low_diversity = estimator._compute_diversity([
            "same response",
            "same response",
            "same response",
        ])
        
        assert high_diversity == 1.0
        assert low_diversity < 1.0
    
    def test_mutual_information(self):
        """Test mutual information computation."""
        estimator = InformationCapacityEstimator()
        
        # High overlap with reference
        high_mi = estimator._compute_mutual_information(
            responses=["The capital of France is Paris"],
            context="Geography question",
            reference_responses=["Paris is the capital of France"],
        )
        
        # Low overlap with reference
        low_mi = estimator._compute_mutual_information(
            responses=["I like apples and oranges"],
            context="Geography question",
            reference_responses=["Paris is the capital of France"],
        )
        
        assert high_mi > low_mi
    
    def test_method_entropy(self):
        """Test entropy-only method."""
        estimator = InformationCapacityEstimator(method="entropy")
        result = estimator.estimate_capacity([
            "response one",
            "response two",
            "response three",
        ])
        
        # Capacity should equal entropy for this method
        assert result.capacity_bits == result.entropy_bits
    
    def test_method_mutual_info(self):
        """Test mutual information method."""
        estimator = InformationCapacityEstimator(method="mutual_info")
        result = estimator.estimate_capacity(
            responses=["Paris is the capital"],
            context="What is the capital of France?",
            reference_responses=["Paris"],
        )
        
        # Capacity should be based on mutual info
        assert result.capacity_bits == result.mutual_info_bits
    
    def test_compare_agents(self):
        """Test comparing multiple agents."""
        estimator = InformationCapacityEstimator()
        
        agent_responses = {
            "agent_a": ["Detailed response with lots of information"],
            "agent_b": ["Short"],
            "agent_c": ["Another detailed response with many words"],
        }
        
        results = estimator.compare_agents(agent_responses)
        
        assert len(results) == 3
        assert "agent_a" in results
        assert "agent_b" in results
        assert "agent_c" in results
        assert all(isinstance(r, CapacityResult) for r in results.values())
    
    def test_rank_agents_by_capacity(self):
        """Test ranking agents by capacity."""
        estimator = InformationCapacityEstimator()
        
        agent_responses = {
            "low_capacity": ["short"],
            "high_capacity": ["This is a very detailed and comprehensive response with many words"],
            "medium_capacity": ["A medium length response"],
        }
        
        ranked = estimator.rank_agents_by_capacity(agent_responses)
        
        assert len(ranked) == 3
        # Should be sorted descending
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]


class TestCreateEstimator:
    """Tests for factory function."""
    
    def test_default(self):
        """Test creating default estimator."""
        estimator = create_estimator()
        
        assert isinstance(estimator, InformationCapacityEstimator)
        assert estimator.method == "combined"
    
    def test_custom_method(self):
        """Test creating estimator with custom method."""
        estimator = create_estimator(method="entropy")
        
        assert estimator.method == "entropy"


class TestCapacityMetrics:
    """Tests for capacity metric properties."""
    
    def test_capacity_non_negative(self):
        """Test that capacity is always non-negative."""
        estimator = InformationCapacityEstimator()
        
        test_cases = [
            [],
            ["single"],
            ["one", "two", "three"],
            ["same", "same", "same"],
        ]
        
        for responses in test_cases:
            result = estimator.estimate_capacity(responses)
            assert result.capacity_bits >= 0
    
    def test_confidence_scales_with_samples(self):
        """Test that confidence increases with more samples."""
        estimator = InformationCapacityEstimator(num_samples=10)
        
        result_1 = estimator.estimate_capacity(["one"])
        result_5 = estimator.estimate_capacity(["one", "two", "three", "four", "five"])
        result_10 = estimator.estimate_capacity([str(i) for i in range(10)])
        
        assert result_1.confidence < result_5.confidence
        assert result_5.confidence < result_10.confidence
        assert result_10.confidence == 1.0
    
    def test_context_improves_mutual_info(self):
        """Test that providing context and reference improves mutual info."""
        estimator = InformationCapacityEstimator(method="combined")
        
        result_no_context = estimator.estimate_capacity(
            responses=["Paris is beautiful"]
        )
        
        result_with_context = estimator.estimate_capacity(
            responses=["Paris is beautiful"],
            context="What is the capital of France?",
            reference_responses=["Paris"],
        )
        
        # With context, mutual info should be computed
        assert result_with_context.mutual_info_bits >= result_no_context.mutual_info_bits
