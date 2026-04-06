"""Entropy Trajectory Tracker — monitors per-step entropy shape during chain-of-thought reasoning.

Implements: arXiv:2603.18940 — Entropy Trajectory Shape Predicts LLM Reasoning Reliability

Key insight: A chain is "monotone" if per-step answer-distribution entropy decreases
at every step. Monotone chains: 68.8% accuracy vs non-monotone: 46.8% (+21.9pp).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrajectorySignal:
    """Signal emitted by the Entropy Trajectory Tracker at each reasoning step."""

    n_steps: int
    monotonicity: bool  # True if entropy decreased at every step so far
    entropy_slope: float  # Linear slope of entropy trajectory (negative = good)
    final_entropy: float
    entropies: list[float]  # Full trajectory for visualization
    confidence_score: float  # 0-1, derived from monotonicity + slope


class EntropyTracker:
    """Monitors entropy trajectory shape during chain-of-thought reasoning.

    Usage:
        tracker = EntropyTracker(config)
        tracker.update(token_logprobs)  # Called per token
        signal = tracker.compute_signal()  # Called per step
    """

    def __init__(self, config: "EntropyConfig") -> None:
        self.config = config
        self.entropies: list[float] = []
        self.monotonicity: bool = True

    def update(self, log_probs: np.ndarray) -> None:
        """Update trajectory with new token's log-probability distribution.

        Args:
            log_probs: Array of log probabilities over vocabulary (shape: [vocab_size])
        """
        entropy = self._compute_entropy(log_probs)
        self.entropies.append(entropy)
        self.monotonicity = self._check_monotonicity(self.entropies)

    def update_from_step(self, step_token: str, step_log_probs: np.ndarray) -> None:
        """Update trajectory from a reasoning step's output distribution.

        Args:
            step_token: The token produced at this step
            step_log_probs: Log probability distribution over possible next tokens
        """
        self.update(step_log_probs)

    def compute_signal(self) -> TrajectorySignal:
        """Compute the TrajectorySignal from the current trajectory.

        Returns:
            TrajectorySignal with monotonicity flag, slope, and confidence score
        """
        slope = self._compute_slope(self.entropies)
        final_entropy = self.entropies[-1] if self.entropies else 0.0

        # Confidence: 0.7 base if monotone, 0.3 otherwise; slope contribution adds up to 0.3
        base = 0.7 if self.monotonicity else 0.3
        slope_contrib = max(0.0, min(0.3, -slope * 0.1))
        confidence = min(1.0, max(0.0, base + slope_contrib))

        return TrajectorySignal(
            n_steps=len(self.entropies),
            monotonicity=self.monotonicity,
            entropy_slope=slope,
            final_entropy=final_entropy,
            entropies=list(self.entropies),
            confidence_score=confidence,
        )

    def reset(self) -> None:
        """Reset the trajectory for a new reasoning chain."""
        self.entropies = []
        self.monotonicity = True

    # -------------------------------------------------------------------------
    # Helper methods (implement in Step 2)
    # -------------------------------------------------------------------------

    def _compute_entropy(self, log_probs: np.ndarray) -> float:
        """Compute Shannon entropy from log probability distribution.

        H = -sum(p * log(p)) where p = softmax(log_probs)
        """
        # DONE: pure math helper (no external dependencies)
        probs = np.exp(log_probs - log_probs.max())  # numerically stable softmax
        probs = probs / probs.sum()
        probs = probs[probs > 0]  # avoid log(0)
        return float(-np.sum(probs * np.log(probs)))

    def _check_monotonicity(self, entropies: list[float]) -> bool:
        """Check if entropy decreased at every consecutive step."""
        if len(entropies) < 2:
            return True
        return all(entropies[i] > entropies[i + 1] for i in range(len(entropies) - 1))

    def _compute_slope(self, entropies: list[float]) -> float:
        """Compute linear slope of entropy trajectory via OLS."""
        if len(entropies) < 2:
            return 0.0
        x = np.arange(len(entropies))
        y = np.array(entropies)
        # slope = cov(x,y) / var(x)
        return float(np.cov(x, y)[0, 1] / np.var(x))
