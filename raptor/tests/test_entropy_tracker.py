"""Tests for entropy_tracker.py — Step 2."""

import numpy as np
import pytest

from raptor.config import EntropyConfig
from raptor.entropy_tracker import EntropyTracker


class TestEntropyComputation:
    """Tests for the pure-math _compute_entropy helper."""

    def test_uniform_distribution_has_max_entropy(self):
        """Uniform distribution over vocab should have high entropy."""
        tracker = EntropyTracker(EntropyConfig())
        uniform_log_probs = np.log(np.ones(100) / 100)
        H = tracker._compute_entropy(uniform_log_probs)
        # Uniform distribution has max entropy = log(100) ~ 4.605
        assert 4.0 < H < 5.0

    def test_sharp_distribution_has_low_entropy(self):
        """A single peak should have near-zero entropy."""
        tracker = EntropyTracker(EntropyConfig())
        sharp_log_probs = np.zeros(100)
        sharp_log_probs[0] = 0.0
        sharp_log_probs[1:] = -100.0
        H = tracker._compute_entropy(sharp_log_probs)
        assert H < 0.5

    def test_numerical_stability_with_extreme_values(self):
        """Should not overflow/underflow with very negative log probs."""
        tracker = EntropyTracker(EntropyConfig())
        log_probs = np.array([-1000.0, -1001.0])
        H = tracker._compute_entropy(log_probs)
        assert 0.0 <= H < 1.0


class TestMonotonicityCheck:
    """Tests for _check_monotonicity helper."""

    def test_empty_trajectory_is_monotone(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._check_monotonicity([]) is True

    def test_single_element_is_monotone(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._check_monotonicity([1.0]) is True

    def test_strictly_decreasing_is_monotone(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._check_monotonicity([4.0, 3.0, 2.0, 1.0]) is True

    def test_increasing_is_not_monotone(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._check_monotonicity([1.0, 2.0, 3.0]) is False

    def test_flat_then_decrease_not_monotone(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._check_monotonicity([3.0, 3.0, 2.0]) is False


class TestSlopeComputation:
    """Tests for _compute_slope helper."""

    def test_zero_length_returns_zero(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._compute_slope([]) == 0.0

    def test_single_element_returns_zero(self):
        tracker = EntropyTracker(EntropyConfig())
        assert tracker._compute_slope([1.0]) == 0.0

    def test_negative_slope_for_decreasing_trajectory(self):
        tracker = EntropyTracker(EntropyConfig())
        slope = tracker._compute_slope([4.0, 3.0, 2.0, 1.0])
        assert slope < 0

    def test_positive_slope_for_increasing_trajectory(self):
        tracker = EntropyTracker(EntropyConfig())
        slope = tracker._compute_slope([1.0, 2.0, 3.0, 4.0])
        assert slope > 0

    def test_zero_slope_for_flat_trajectory(self):
        tracker = EntropyTracker(EntropyConfig())
        slope = tracker._compute_slope([2.0, 2.0, 2.0, 2.0])
        assert abs(slope) < 1e-10


class TestEntropyTrackerIntegration:
    """Integration tests for the full EntropyTracker."""

    def _make_log_probs(self, entropy_level: str = "medium") -> np.ndarray:
        """Create log_probs with roughly controlled entropy."""
        if entropy_level == "high":
            return np.log(np.ones(100) / 100)
        elif entropy_level == "low":
            probs = np.full(100, 1e-10)
            probs[0] = 1.0
            probs = probs / probs.sum()
            return np.log(probs)
        else:
            probs = np.random.dirichlet(np.ones(100) * 0.5)
            return np.log(probs + 1e-30)

    def test_update_adds_to_trajectory(self):
        tracker = EntropyTracker(EntropyConfig())
        assert len(tracker.entropies) == 0
        tracker.update(self._make_log_probs("high"))
        assert len(tracker.entropies) == 1
        tracker.update(self._make_log_probs("low"))
        assert len(tracker.entropies) == 2

    def test_compute_signal_returns_valid(self):
        tracker = EntropyTracker(EntropyConfig())
        tracker.update(self._make_log_probs("high"))
        tracker.update(self._make_log_probs("low"))
        signal = tracker.compute_signal()
        assert signal.n_steps == 2
        assert isinstance(signal.monotonicity, bool)
        assert isinstance(signal.entropy_slope, float)
        assert isinstance(signal.final_entropy, float)
        assert len(signal.entropies) == 2

    def test_reset_clears_state(self):
        tracker = EntropyTracker(EntropyConfig())
        tracker.update(self._make_log_probs("high"))
        tracker.update(self._make_log_probs("low"))
        assert len(tracker.entropies) == 2
        tracker.reset()
        assert len(tracker.entropies) == 0
        assert tracker.monotonicity is True

    def test_update_from_step_works(self):
        tracker = EntropyTracker(EntropyConfig())
        tracker.update_from_step("token_a", self._make_log_probs("high"))
        assert len(tracker.entropies) == 1

    def test_full_monotone_flow(self):
        """Decreasing entropy across steps should yield monotonicity=True and negative slope."""
        tracker = EntropyTracker(EntropyConfig())
        # Create strictly decreasing entropy: high → medium-high → medium → low
        for scale in [1.0, 2.0, 5.0, 100.0]:
            probs = np.ones(100)
            probs[0] = scale
            probs = probs / probs.sum()
            tracker.update(np.log(probs))

        signal = tracker.compute_signal()
        assert signal.monotonicity is True
        assert signal.entropy_slope < 0
        assert signal.n_steps == 4

    def test_confidence_score_bounded(self):
        tracker = EntropyTracker(EntropyConfig())
        tracker.update(self._make_log_probs("high"))
        tracker.update(self._make_log_probs("low"))
        signal = tracker.compute_signal()
        assert 0.0 <= signal.confidence_score <= 1.0

    def test_confidence_higher_for_monotone(self):
        """Monotone decreasing trajectory should have higher confidence than increasing."""
        # Monotone decreasing
        tracker_dec = EntropyTracker(EntropyConfig())
        for scale in [1.0, 2.0, 5.0, 100.0]:
            probs = np.ones(100)
            probs[0] = scale
            probs = probs / probs.sum()
            tracker_dec.update(np.log(probs))
        signal_dec = tracker_dec.compute_signal()

        # Increasing
        tracker_inc = EntropyTracker(EntropyConfig())
        for scale in [100.0, 5.0, 2.0, 1.0]:
            probs = np.ones(100)
            probs[0] = scale
            probs = probs / probs.sum()
            tracker_inc.update(np.log(probs))
        signal_inc = tracker_inc.compute_signal()

        assert signal_dec.confidence_score > signal_inc.confidence_score

    def test_empty_signal(self):
        tracker = EntropyTracker(EntropyConfig())
        signal = tracker.compute_signal()
        assert signal.n_steps == 0
        assert signal.final_entropy == 0.0
        assert signal.entropies == []
        assert signal.monotonicity is True
