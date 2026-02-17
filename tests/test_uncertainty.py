"""Tests for uncertainty module - CATTS Uncertainty Statistics."""

import pytest
import math
from src.uncertainty import UncertaintyEstimator, VoteDistribution


class TestVoteDistribution:
    """Tests for VoteDistribution dataclass."""
    
    def test_initialization(self):
        """Test VoteDistribution initialization."""
        votes = VoteDistribution()
        assert votes.candidates == {}
        assert votes.total_votes == 0
        assert votes.observation == ""
    
    def test_add_vote(self):
        """Test adding votes."""
        votes = VoteDistribution()
        votes.add_vote("action_a")
        assert votes.candidates["action_a"] == 1
        assert votes.total_votes == 1
        
        votes.add_vote("action_a")
        assert votes.candidates["action_a"] == 2
        assert votes.total_votes == 2
        
        votes.add_vote("action_b")
        assert votes.candidates["action_b"] == 1
        assert votes.total_votes == 3
    
    def test_get_probabilities(self):
        """Test probability calculation."""
        votes = VoteDistribution()
        votes.add_vote("action_a")
        votes.add_vote("action_a")
        votes.add_vote("action_b")
        
        probs = votes.get_probabilities()
        assert probs["action_a"] == 2/3
        assert probs["action_b"] == 1/3
    
    def test_get_probabilities_empty(self):
        """Test probability calculation with no votes."""
        votes = VoteDistribution()
        assert votes.get_probabilities() == {}
    
    def test_get_most_common(self):
        """Test getting most common action."""
        votes = VoteDistribution()
        votes.add_vote("action_a")
        votes.add_vote("action_a")
        votes.add_vote("action_b")
        
        action, count = votes.get_most_common()
        assert action == "action_a"
        assert count == 2
    
    def test_get_most_common_empty(self):
        """Test getting most common with no votes."""
        votes = VoteDistribution()
        action, count = votes.get_most_common()
        assert action == ""
        assert count == 0
    
    def test_get_confidence(self):
        """Test confidence calculation."""
        votes = VoteDistribution()
        votes.add_vote("action_a")
        votes.add_vote("action_a")
        votes.add_vote("action_b")
        
        # Confidence = top votes / total votes = 2/3
        assert votes.get_confidence() == pytest.approx(2/3)
    
    def test_get_confidence_empty(self):
        """Test confidence with no votes."""
        votes = VoteDistribution()
        assert votes.get_confidence() == 0.0


class TestUncertaintyEstimatorInitialization:
    """Tests for UncertaintyEstimator initialization."""
    
    def test_default_initialization(self):
        """Test estimator with default parameters."""
        estimator = UncertaintyEstimator()
        assert estimator.method == "ensemble"
        assert estimator.num_samples == 5
    
    def test_custom_initialization(self):
        """Test estimator with custom parameters."""
        estimator = UncertaintyEstimator(method="entropy", num_samples=10, random_seed=42)
        assert estimator.method == "entropy"
        assert estimator.num_samples == 10
    
    def test_reset(self):
        """Test resetting history."""
        estimator = UncertaintyEstimator()
        estimator._history.append({"test": "data"})
        estimator.reset()
        assert len(estimator._history) == 0


class TestGenerateVotes:
    """Tests for generate_votes method."""
    
    def test_generate_votes_basic(self):
        """Test vote generation."""
        estimator = UncertaintyEstimator(random_seed=42)
        votes = estimator.generate_votes("Click the submit button", n_samples=10)
        
        assert isinstance(votes, VoteDistribution)
        assert votes.total_votes == 10
        assert len(votes.candidates) > 0
    
    def test_generate_votes_with_candidates(self):
        """Test vote generation with predefined candidates."""
        estimator = UncertaintyEstimator(random_seed=42)
        candidates = ["click_submit", "click_cancel", "fill_form"]
        votes = estimator.generate_votes(
            "Form page", 
            n_samples=20,
            candidate_actions=candidates
        )
        
        assert votes.total_votes == 20
        # All votes should be from the candidate list
        for action in votes.candidates.keys():
            assert action in candidates
    
    def test_generate_votes_metadata(self):
        """Test vote generation includes metadata."""
        estimator = UncertaintyEstimator()
        votes = estimator.generate_votes("Test observation", n_samples=5)
        
        assert votes.metadata["n_samples"] == 5
        assert votes.metadata["method"] == "ensemble"


class TestComputeEntropy:
    """Tests for compute_entropy method."""
    
    def test_entropy_uniform_distribution(self):
        """Test entropy with uniform distribution (max uncertainty)."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # 4 actions with equal votes
        for action in ["a", "b", "c", "d"]:
            for _ in range(5):
                votes.add_vote(action)
        
        entropy = estimator.compute_entropy(votes)
        # Max entropy for 4 equally likely outcomes = log2(4) = 2
        assert entropy == pytest.approx(2.0, abs=0.01)
    
    def test_entropy_certain_distribution(self):
        """Test entropy with certain distribution (no uncertainty)."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # All votes for one action
        for _ in range(10):
            votes.add_vote("action_a")
        
        entropy = estimator.compute_entropy(votes)
        assert entropy == pytest.approx(0.0, abs=0.01)
    
    def test_entropy_empty(self):
        """Test entropy with empty votes."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        assert estimator.compute_entropy(votes) == 0.0
    
    def test_entropy_binary(self):
        """Test entropy with binary distribution."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # 75/25 split
        for _ in range(6):
            votes.add_vote("action_a")
        for _ in range(2):
            votes.add_vote("action_b")
        
        entropy = estimator.compute_entropy(votes)
        # H = -0.75*log2(0.75) - 0.25*log2(0.25) ≈ 0.811
        expected = -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
        assert entropy == pytest.approx(expected, abs=0.01)


class TestComputeVariance:
    """Tests for compute_variance method."""
    
    def test_variance_high(self):
        """Test variance with concentrated votes."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # Highly concentrated: 9, 1
        for _ in range(9):
            votes.add_vote("action_a")
        votes.add_vote("action_b")
        
        variance = estimator.compute_variance(votes)
        # Mean = 5, variance = ((9-5)^2 + (1-5)^2) / 2 = 16
        assert variance == pytest.approx(16.0, abs=0.01)
    
    def test_variance_low(self):
        """Test variance with evenly distributed votes."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # Even distribution: 5, 5
        for _ in range(5):
            votes.add_vote("action_a")
            votes.add_vote("action_b")
        
        variance = estimator.compute_variance(votes)
        # Mean = 5, variance = 0
        assert variance == pytest.approx(0.0, abs=0.01)
    
    def test_variance_empty(self):
        """Test variance with empty votes."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        assert estimator.compute_variance(votes) == 0.0
    
    def test_variance_single_candidate(self):
        """Test variance with single candidate."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        votes.add_vote("action_a")
        assert estimator.compute_variance(votes) == 0.0


class TestComputePairwiseDisagreement:
    """Tests for compute_pairwise_disagreement method."""
    
    def test_disagreement_complete(self):
        """Test with maximum disagreement."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # Perfect split: 50% a, 50% b
        for _ in range(5):
            votes.add_vote("action_a")
            votes.add_vote("action_b")
        
        disagreement = estimator.compute_pairwise_disagreement(votes)
        # All pairs disagree
        assert disagreement == pytest.approx(0.555, abs=0.01)
    
    def test_disagreement_none(self):
        """Test with no disagreement."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # All same
        for _ in range(10):
            votes.add_vote("action_a")
        
        disagreement = estimator.compute_pairwise_disagreement(votes)
        assert disagreement == 0.0
    
    def test_disagreement_empty(self):
        """Test with empty votes."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        assert estimator.compute_pairwise_disagreement(votes) == 0.0
    
    def test_disagreement_single_vote(self):
        """Test with single vote."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        votes.add_vote("action_a")
        assert estimator.compute_pairwise_disagreement(votes) == 0.0


class TestGetUncertaintyStats:
    """Tests for get_uncertainty_stats method."""
    
    def test_stats_structure(self):
        """Test that stats contains all expected keys."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        votes.add_vote("action_a")
        votes.add_vote("action_b")
        
        stats = estimator.get_uncertainty_stats(votes)
        
        expected_keys = [
            "entropy", "normalized_entropy", "variance",
            "pairwise_disagreement", "confidence", "uncertainty",
            "num_candidates", "total_votes"
        ]
        for key in expected_keys:
            assert key in stats
    
    def test_stats_uncertain_distribution(self):
        """Test stats for uncertain distribution."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # Uncertain: equal votes
        for _ in range(5):
            votes.add_vote("action_a")
            votes.add_vote("action_b")
        
        stats = estimator.get_uncertainty_stats(votes)
        
        assert stats["normalized_entropy"] > 0.9  # High uncertainty
        assert stats["variance"] < 0.1  # Low variance
        assert stats["confidence"] == 0.5  # 50/50 split
        assert stats["uncertainty"] == 0.5
    
    def test_stats_certain_distribution(self):
        """Test stats for certain distribution."""
        estimator = UncertaintyEstimator()
        votes = VoteDistribution()
        
        # Certain: all same
        for _ in range(10):
            votes.add_vote("action_a")
        
        stats = estimator.get_uncertainty_stats(votes)
        
        assert stats["entropy"] == 0.0
        assert stats["normalized_entropy"] == 0.0
        assert stats["confidence"] == 1.0
        assert stats["uncertainty"] == 0.0


class TestShouldScaleUp:
    """Tests for should_scale_up method."""
    
    def test_should_scale_up_high_uncertainty(self):
        """Test scaling up when uncertain."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.8, "pairwise_disagreement": 0.7}
        
        assert estimator.should_scale_up(stats, threshold=0.5) is True
    
    def test_should_scale_up_low_uncertainty(self):
        """Test not scaling up when certain."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.2, "pairwise_disagreement": 0.1}
        
        assert estimator.should_scale_up(stats, threshold=0.5) is False
    
    def test_should_scale_up_by_disagreement(self):
        """Test scaling up due to high pairwise disagreement."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.3, "pairwise_disagreement": 0.8}
        
        assert estimator.should_scale_up(stats, threshold=0.5) is True
    
    def test_should_scale_up_threshold_edge(self):
        """Test at threshold boundary."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.5, "pairwise_disagreement": 0.0}
        
        assert estimator.should_scale_up(stats, threshold=0.5) is True


class TestGetComputeBudget:
    """Tests for get_compute_budget method."""
    
    def test_budget_max_uncertainty(self):
        """Test budget with maximum uncertainty."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 1.0}
        
        budget = estimator.get_compute_budget(stats, min_samples=5, max_samples=20)
        assert budget == 20
    
    def test_budget_min_uncertainty(self):
        """Test budget with minimum uncertainty."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.0}
        
        budget = estimator.get_compute_budget(stats, min_samples=5, max_samples=20)
        assert budget == 5
    
    def test_budget_mid_uncertainty(self):
        """Test budget with medium uncertainty."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.5}
        
        budget = estimator.get_compute_budget(stats, min_samples=5, max_samples=15)
        assert budget == 10
    
    def test_budget_default_params(self):
        """Test budget with default parameters."""
        estimator = UncertaintyEstimator()
        stats = {"normalized_entropy": 0.5}
        
        budget = estimator.get_compute_budget(stats)
        assert budget >= 3
        assert budget <= 20


class TestLegacyMethods:
    """Tests for backward compatibility methods."""
    
    def test_estimate(self):
        """Test legacy estimate method."""
        estimator = UncertaintyEstimator(random_seed=42)
        context = {"observation": "Click the button"}
        uncertainty = estimator.estimate("click", context)
        
        assert 0 <= uncertainty <= 1
    
    def test_estimate_action_sequence(self):
        """Test legacy estimate_action_sequence method."""
        estimator = UncertaintyEstimator(random_seed=42)
        actions = ["click1", "click2", "click3"]
        uncertainties = estimator.estimate_action_sequence(actions, {})
        
        assert len(uncertainties) == len(actions)
        assert all(0 <= u <= 1 for u in uncertainties)
    
    def test_should_pause(self):
        """Test pause decision."""
        estimator = UncertaintyEstimator()
        assert estimator.should_pause(0.8, threshold=0.7) is True
        assert estimator.should_pause(0.5, threshold=0.7) is False
        assert estimator.should_pause(0.7, threshold=0.7) is True
    
    def test_get_confidence(self):
        """Test confidence conversion."""
        estimator = UncertaintyEstimator()
        assert estimator.get_confidence(0.3) == pytest.approx(0.7)
        assert estimator.get_confidence(0.0) == pytest.approx(1.0)
        assert estimator.get_confidence(1.0) == pytest.approx(0.0)


class TestIntegration:
    """Integration tests for the full CATTS workflow."""
    
    def test_full_workflow_uncertain_state(self):
        """Test full workflow with uncertain state."""
        estimator = UncertaintyEstimator(random_seed=42)
        
        # Generate initial votes
        observation = "Complex page with multiple possible actions"
        votes = estimator.generate_votes(observation, n_samples=5)
        
        # Get uncertainty stats
        stats = estimator.get_uncertainty_stats(votes)
        
        # Check if we need more samples
        if estimator.should_scale_up(stats, threshold=0.5):
            additional_samples = estimator.get_compute_budget(stats, min_samples=5, max_samples=15)
            # Generate more votes
            more_votes = estimator.generate_votes(observation, n_samples=additional_samples)
            # Combine distributions (simplified)
            for action, count in more_votes.candidates.items():
                for _ in range(count):
                    votes.add_vote(action)
            # Recompute stats
            stats = estimator.get_uncertainty_stats(votes)
        
        # Assertions
        assert "entropy" in stats
        assert votes.total_votes >= 5
    
    def test_full_workflow_certain_state(self):
        """Test full workflow with certain state."""
        estimator = UncertaintyEstimator(random_seed=42)
        
        observation = "Simple page with clear next action"
        # Provide limited candidates to simulate certainty
        candidates = ["click_submit"]
        votes = estimator.generate_votes(
            observation, 
            n_samples=10,
            candidate_actions=candidates
        )
        
        stats = estimator.get_uncertainty_stats(votes)
        
        # With only one candidate, certainty is high
        assert stats["normalized_entropy"] == 0.0
        assert stats["confidence"] == 1.0
        assert estimator.should_scale_up(stats, threshold=0.5) is False
        assert estimator.get_compute_budget(stats, min_samples=3, max_samples=10) == 3
