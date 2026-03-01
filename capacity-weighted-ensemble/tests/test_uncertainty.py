"""
Tests for the Uncertainty Estimator for test-time compute scaling.
"""

import pytest
import math
from src.uncertainty import UncertaintyEstimator, UncertaintyResult


class TestUncertaintyResult:
    """Tests for the UncertaintyResult dataclass."""
    
    def test_valid_result_creation(self):
        """Test creating a valid UncertaintyResult."""
        result = UncertaintyResult(
            uncertainty_score=0.5,
            entropy=1.0,
            unique_responses=2,
            vote_distribution={"A": 0.5, "B": 0.5},
            method="entropy"
        )
        assert result.uncertainty_score == 0.5
        assert result.entropy == 1.0
        assert result.unique_responses == 2
        assert result.method == "entropy"
    
    def test_invalid_uncertainty_score(self):
        """Test that invalid uncertainty score raises ValueError."""
        with pytest.raises(ValueError, match="uncertainty_score"):
            UncertaintyResult(
                uncertainty_score=1.5,
                entropy=1.0,
                unique_responses=2,
                vote_distribution={"A": 0.5, "B": 0.5},
                method="entropy"
            )
    
    def test_invalid_entropy(self):
        """Test that negative entropy raises ValueError."""
        with pytest.raises(ValueError, match="entropy"):
            UncertaintyResult(
                uncertainty_score=0.5,
                entropy=-1.0,
                unique_responses=2,
                vote_distribution={"A": 0.5, "B": 0.5},
                method="entropy"
            )
    
    def test_invalid_vote_distribution(self):
        """Test that vote distribution not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="vote_distribution"):
            UncertaintyResult(
                uncertainty_score=0.5,
                entropy=1.0,
                unique_responses=2,
                vote_distribution={"A": 0.3, "B": 0.3},
                method="entropy"
            )


class TestUncertaintyEstimatorInit:
    """Tests for UncertaintyEstimator initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        estimator = UncertaintyEstimator()
        assert estimator.scale_threshold == 0.5
        assert estimator.min_samples == 1
        assert estimator.normalize is True
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        estimator = UncertaintyEstimator(scale_threshold=0.7, min_samples=3, normalize=False)
        assert estimator.scale_threshold == 0.7
        assert estimator.min_samples == 3
        assert estimator.normalize is False
    
    def test_invalid_scale_threshold(self):
        """Test that invalid scale threshold raises ValueError."""
        with pytest.raises(ValueError, match="scale_threshold"):
            UncertaintyEstimator(scale_threshold=1.5)
    
    def test_invalid_min_samples(self):
        """Test that invalid min_samples raises ValueError."""
        with pytest.raises(ValueError, match="min_samples"):
            UncertaintyEstimator(min_samples=0)


class TestUncertaintyEstimation:
    """Tests for uncertainty estimation."""
    
    def test_identical_responses_low_uncertainty(self):
        """Test that identical responses yield low uncertainty."""
        estimator = UncertaintyEstimator()
        responses = ["answer"] * 5
        result = estimator.estimate_uncertainty(responses)
        
        assert result.uncertainty_score == 0.0
        assert result.entropy == 0.0
        assert result.unique_responses == 1
        assert result.vote_distribution == {"answer": 1.0}
    
    def test_diverse_responses_high_uncertainty(self):
        """Test that diverse responses yield high uncertainty."""
        estimator = UncertaintyEstimator()
        responses = ["A", "B", "C", "D"]
        result = estimator.estimate_uncertainty(responses)
        
        assert result.uncertainty_score > 0.9  # Should be close to 1.0
        assert result.unique_responses == 4
    
    def test_two_way_split_moderate_uncertainty(self):
        """Test that 50/50 split yields moderate uncertainty."""
        estimator = UncertaintyEstimator()
        responses = ["A", "A", "B", "B"]
        result = estimator.estimate_uncertainty(responses)
        
        # For 2 options with equal probability, entropy = 1 bit
        assert result.entropy == pytest.approx(1.0)
        assert result.vote_distribution["A"] == 0.5
        assert result.vote_distribution["B"] == 0.5
    
    def test_single_response(self):
        """Test uncertainty with single response."""
        estimator = UncertaintyEstimator()
        result = estimator.estimate_uncertainty(["only answer"])
        
        assert result.uncertainty_score == 0.0
        assert result.unique_responses == 1
    
    def test_empty_responses_raises_error(self):
        """Test that empty responses raises ValueError."""
        estimator = UncertaintyEstimator()
        with pytest.raises(ValueError):
            estimator.estimate_uncertainty([])


class TestShouldScale:
    """Tests for should_scale method."""
    
    def test_should_scale_above_threshold(self):
        """Test that high uncertainty triggers scaling."""
        estimator = UncertaintyEstimator(scale_threshold=0.5)
        assert estimator.should_scale(0.7) is True
        assert estimator.should_scale(0.51) is True  # Above threshold
        assert estimator.should_scale(0.5) is False  # At threshold (strictly greater)
    
    def test_should_not_scale_below_threshold(self):
        """Test that low uncertainty does not trigger scaling."""
        estimator = UncertaintyEstimator(scale_threshold=0.5)
        assert estimator.should_scale(0.3) is False
        assert estimator.should_scale(0.49) is False
    
    def test_should_scale_with_zero_uncertainty(self):
        """Test that zero uncertainty never triggers scaling."""
        estimator = UncertaintyEstimator(scale_threshold=0.5)
        assert estimator.should_scale(0.0) is False


class TestVoteDistribution:
    """Tests for vote distribution calculation."""
    
    def test_vote_distribution_calculation(self):
        """Test correct vote distribution calculation."""
        estimator = UncertaintyEstimator()
        responses = ["A", "A", "A", "B", "C"]
        distribution = estimator.get_vote_distribution(responses)
        
        assert distribution["A"] == pytest.approx(0.6)
        assert distribution["B"] == pytest.approx(0.2)
        assert distribution["C"] == pytest.approx(0.2)
    
    def test_vote_distribution_single_response(self):
        """Test vote distribution with single response."""
        estimator = UncertaintyEstimator()
        distribution = estimator.get_vote_distribution(["X"])
        
        assert distribution == {"X": 1.0}
    
    def test_vote_distribution_sums_to_one(self):
        """Test that vote distribution always sums to 1.0."""
        estimator = UncertaintyEstimator()
        distribution = estimator.get_vote_distribution(["A", "B", "C", "D", "E"])
        
        total = sum(distribution.values())
        assert total == pytest.approx(1.0)


class TestEntropyCalculation:
    """Tests for entropy calculation."""
    
    def test_entropy_perfect_agreement(self):
        """Test entropy is zero for perfect agreement."""
        estimator = UncertaintyEstimator()
        responses = ["same"] * 10
        result = estimator.estimate_uncertainty(responses)
        
        assert result.entropy == 0.0
    
    def test_entropy_uniform_distribution(self):
        """Test entropy for uniform distribution."""
        estimator = UncertaintyEstimator()
        responses = ["A", "B", "C", "D"]  # 4 options, each 25%
        result = estimator.estimate_uncertainty(responses)
        
        # Entropy should be log2(4) = 2 bits
        assert result.entropy == pytest.approx(2.0, rel=0.01)
    
    def test_normalized_entropy(self):
        """Test that normalized entropy is between 0 and 1."""
        estimator = UncertaintyEstimator(normalize=True)
        responses = ["A", "B", "C"]
        result = estimator.estimate_uncertainty(responses)
        
        assert 0.0 <= result.uncertainty_score <= 1.0
    
    def test_unnormalized_entropy(self):
        """Test unnormalized entropy gives raw bits."""
        estimator = UncertaintyEstimator(normalize=False)
        responses = ["A", "B"]  # 2 options, entropy = 1 bit
        result = estimator.estimate_uncertainty(responses)
        
        assert result.entropy == pytest.approx(1.0)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_large_number_of_responses(self):
        """Test with large number of responses."""
        estimator = UncertaintyEstimator()
        responses = [f"response_{i % 10}" for i in range(1000)]
        result = estimator.estimate_uncertainty(responses)
        
        assert result.unique_responses == 10
        assert result.uncertainty_score > 0
    
    def test_unicode_responses(self):
        """Test with unicode responses."""
        estimator = UncertaintyEstimator()
        responses = ["答案", "答案", "回答"]
        result = estimator.estimate_uncertainty(responses)
        
        assert result.unique_responses == 2
        assert "答案" in result.vote_distribution
    
    def test_whitespace_responses(self):
        """Test with whitespace differences."""
        estimator = UncertaintyEstimator()
        responses = ["answer", "answer ", "answer  "]
        result = estimator.estimate_uncertainty(responses)
        
        # Implementation strips whitespace, so all should be treated as same
        assert result.unique_responses == 1
        assert result.vote_distribution == {"answer": 1.0}
    
    def test_integration_with_should_scale(self):
        """Test integration between estimation and should_scale."""
        estimator = UncertaintyEstimator(scale_threshold=0.3)
        
        # High disagreement should trigger scaling
        diverse = ["A", "B", "C", "D", "E"]
        result = estimator.estimate_uncertainty(diverse)
        assert estimator.should_scale(result.uncertainty_score)
        
        # Low disagreement should not trigger scaling (threshold 0.3)
        # With 10 A's and 1 B, entropy is low
        similar = ["A"] * 10 + ["B"]
        result = estimator.estimate_uncertainty(similar)
        # uncertainty should be around 0.44, threshold is 0.3, so it will scale
        # Let's use an even more skewed distribution
        similar2 = ["A"] * 20 + ["B"]
        result2 = estimator.estimate_uncertainty(similar2)
        assert not estimator.should_scale(result2.uncertainty_score)
