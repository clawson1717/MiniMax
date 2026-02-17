"""Tests for uncertainty module."""

import pytest
from src.uncertainty import UncertaintyEstimator


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = UncertaintyEstimator(method="ensemble", num_samples=5)
        assert estimator.method == "ensemble"
        assert estimator.num_samples == 5
    
    def test_estimate_uncertainty(self):
        """Test uncertainty estimation."""
        estimator = UncertaintyEstimator()
        uncertainty = estimator.estimate("click button", {"page": "home"})
        assert 0 <= uncertainty <= 1
    
    def test_estimate_action_sequence(self):
        """Test sequence uncertainty estimation."""
        estimator = UncertaintyEstimator()
        actions = ["click1", "click2", "click3"]
        uncertainties = estimator.estimate_action_sequence(actions, {})
        assert len(uncertainties) == len(actions)
        assert all(0 <= u <= 1 for u in uncertainties)
    
    def test_should_pause(self):
        """Test pause decision."""
        estimator = UncertaintyEstimator()
        assert estimator.should_pause(0.8, threshold=0.7) is True
        assert estimator.should_pause(0.5, threshold=0.7) is False
    
    def test_get_confidence(self):
        """Test confidence conversion."""
        estimator = UncertaintyEstimator()
        assert estimator.get_confidence(0.3) == 0.7
        assert estimator.get_confidence(0.8) == pytest.approx(0.2)
    
    def test_reset(self):
        """Test resetting history."""
        estimator = UncertaintyEstimator()
        estimator.estimate("action", {})
        estimator.reset()
        assert len(estimator._history) == 0
