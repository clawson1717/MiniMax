"""Utility Score Engine — computes utility-guided action selection.

Implements: arXiv:2603.19896 — Utility-Guided Agent Orchestration for Efficient LLM Tool Use

Action space: respond, reroll, verify, escalate, retrieve, stop

Utility function:
    U(a | S) = Σ_k w_k · φ_k(a, S) - cost(a)

where φ_k are feature functions: gain, confidence, cost, redundancy, severity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from raptor.config import Config, OrchestrationAction, UtilityConfig
from raptor.entropy_tracker import TrajectorySignal
from raptor.disagreement import DisagreementSignal


@dataclass
class ActionScore:
    """Utility score for a single action."""

    action: OrchestrationAction
    utility: float
    breakdown: dict[str, float]  # φ_k contributions


class UtilityEngine:
    """Computes utility scores for all actions given the fused RAPTOR signal vector.

    Supports two weight modes:
      - ``"fixed"`` — weights from :class:`UtilityConfig` (hand-tuned, default)
      - ``"learned"`` — online SGD updates via :meth:`update_weights`

    Usage::

        engine = UtilityEngine(config)
        scores = engine.score_all(
            traj_signal=TrajectorySignal(...),
            disa_signal=DisagreementSignal(...),
            step_context={"has_retrieved": False, "n_rerolls": 0, ...}
        )
        best_action = engine.select_best(scores)
    """

    def __init__(self, config: Config, mode: str = "fixed") -> None:
        self.config = config.utility
        self._mode = mode
        # Copy weights so learned mode can mutate without touching config
        self._weights: dict[str, float] = dict(self.config.weights)
        self._validate_weights()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> dict[str, float]:
        """Current weights (may differ from config in learned mode)."""
        return dict(self._weights)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all(
        self,
        traj_signal: TrajectorySignal,
        disa_signal: DisagreementSignal,
        step_context: Optional[dict] = None,
    ) -> list[ActionScore]:
        """Compute utility scores for all 6 actions.

        Args:
            traj_signal: Signal from Entropy Trajectory Tracker.
            disa_signal: Signal from Disagreement Monitor.
            step_context: Runtime context — keys:
                ``n_rerolls``, ``has_retrieved``, ``step_num``,
                ``max_rerolls``, ``previous_action``.

        Returns:
            List of :class:`ActionScore` sorted by utility **descending**.
        """
        ctx = step_context or {}
        scores: list[ActionScore] = []

        for action in OrchestrationAction:
            features = {
                "gain": self._phi_gain(action, traj_signal, disa_signal),
                "confidence": self._phi_confidence(action, traj_signal, disa_signal),
                "cost_penalty": self._phi_cost(action, ctx),
                "redundancy_penalty": self._phi_redundancy(action, ctx),
                "severity": self._phi_severity(action, traj_signal, disa_signal),
            }

            # U(a|S) = Σ_k w_k · φ_k(a, S)
            # Note: cost(a) is folded into φ_cost via action_costs + step scaling;
            # the cost_penalty weight controls how much cost matters overall.
            utility = sum(
                self._weights[k] * features[k] for k in self._weights
            )

            scores.append(
                ActionScore(action=action, utility=utility, breakdown=features)
            )

        scores.sort(key=lambda s: s.utility, reverse=True)
        return scores

    def select_best(
        self,
        scores: list[ActionScore],
        previous_action: Optional[OrchestrationAction] = None,
    ) -> OrchestrationAction:
        """Select the highest-utility action, with optional hysteresis.

        If *previous_action* is supplied and the best-scoring action differs,
        a switch only happens when the margin exceeds
        ``UtilityConfig.switch_threshold``.
        """
        if not scores:
            return OrchestrationAction.RESPOND

        best = scores[0]

        if previous_action is None or best.action == previous_action:
            return best.action

        # Look up the previous action's utility in the scored list
        prev_utility: Optional[float] = None
        for s in scores:
            if s.action == previous_action:
                prev_utility = s.utility
                break

        if prev_utility is not None:
            margin = best.utility - prev_utility
            if margin <= self.config.switch_threshold:
                return previous_action

        return best.action

    # ------------------------------------------------------------------
    # Learned-weight mode — online SGD
    # ------------------------------------------------------------------

    def update_weights(
        self,
        features: dict[str, float],
        reward: float,
        predicted: float,
        learning_rate: float = 0.01,
    ) -> None:
        """Update weights via online SGD (learned-weight mode only).

        Gradient step::

            w_k ← w_k + lr · (reward − predicted) · φ_k

        Args:
            features: Feature dict from the last scored action's ``breakdown``.
            reward: Observed reward (e.g. 1.0 for correct, 0.0 otherwise).
            predicted: Predicted utility from :meth:`score_all`.
            learning_rate: SGD step size.
        """
        if self._mode != "learned":
            return

        error = reward - predicted
        for k in self._weights:
            if k in features:
                self._weights[k] += learning_rate * error * features[k]

    # ------------------------------------------------------------------
    # Feature functions  (φ_k)
    # ------------------------------------------------------------------

    def _phi_gain(
        self,
        action: OrchestrationAction,
        traj: TrajectorySignal,
        disa: DisagreementSignal,
    ) -> float:
        """φ_gain — Expected accuracy improvement for *action* given signals.

        * **RESPOND** gains most when monotone + low-disagreement.
        * **REROLL** gains most when non-monotone + weak-disagreement.
        * **VERIFY** gains most when non-monotone + medium-disagreement.
        """
        is_mono = traj.monotonicity
        tier = disa.disagreement_tier
        combined_conf = 0.5 * traj.confidence_score + 0.5 * disa.confidence_score

        if action == OrchestrationAction.RESPOND:
            if is_mono and tier == "low":
                return 0.9
            if is_mono:
                return 0.6
            if tier == "low":
                return 0.3
            return 0.1

        if action == OrchestrationAction.REROLL:
            if not is_mono and tier == "weak":
                return 0.9
            if not is_mono:
                return 0.5
            if tier == "weak":
                return 0.4
            return 0.15

        if action == OrchestrationAction.VERIFY:
            if not is_mono and tier == "medium":
                return 0.9
            if not is_mono:
                return 0.5
            if tier == "medium":
                return 0.4
            return 0.15

        if action == OrchestrationAction.ESCALATE:
            if not is_mono and tier == "weak":
                return 0.7
            if not is_mono:
                return 0.4
            if tier == "weak":
                return 0.3
            return 0.1

        if action == OrchestrationAction.RETRIEVE:
            if combined_conf < 0.4:
                return 0.6
            if combined_conf < 0.7:
                return 0.3
            return 0.1

        if action == OrchestrationAction.STOP:
            if combined_conf < 0.2:
                return 0.3
            return 0.05

        return 0.0  # pragma: no cover

    def _phi_confidence(
        self,
        action: OrchestrationAction,
        traj: TrajectorySignal,
        disa: DisagreementSignal,
    ) -> float:
        """φ_confidence — Calibrated confidence from combined signals."""
        return float(traj.confidence_score * 0.5 + disa.confidence_score * 0.5)

    def _phi_cost(
        self,
        action: OrchestrationAction,
        step_context: dict,
    ) -> float:
        """φ_cost — Step cost (token usage, latency).

        Base cost from ``action_costs``, scaled upward as step_num increases
        (urgency to finish).  Capped at 1.0.
        """
        base_cost = self.config.action_costs.get(action, 0.0)
        step_num = step_context.get("step_num", 0)
        step_factor = 1.0 + step_num / 20.0
        return min(1.0, base_cost * step_factor)

    def _phi_redundancy(
        self,
        action: OrchestrationAction,
        step_context: dict,
    ) -> float:
        """φ_redundancy — Penalty for already-taken actions.

        * RETRIEVE after already retrieving → 1.0
        * REROLL beyond budget → 1.0; within budget → proportional
        """
        if action == OrchestrationAction.RETRIEVE:
            if step_context.get("has_retrieved", False):
                return 1.0

        if action == OrchestrationAction.REROLL:
            n_rerolls = step_context.get("n_rerolls", 0)
            max_rerolls = step_context.get("max_rerolls", 3)
            if max_rerolls <= 0:
                return 1.0 if n_rerolls > 0 else 0.0
            if n_rerolls >= max_rerolls:
                return 1.0
            return n_rerolls / max_rerolls

        return 0.0

    def _phi_severity(
        self,
        action: OrchestrationAction,
        traj: TrajectorySignal,
        disa: DisagreementSignal,
    ) -> float:
        """φ_severity — How bad is getting it wrong?

        Base severity is highest when non-monotone + weak disagreement.

        * RESPOND / STOP commit to an answer → severity *inverted* (prefer
          responding only when severity is low).
        * VERIFY / ESCALATE address uncertainty → severity passed through.
        * REROLL / RETRIEVE get partial credit.
        """
        is_mono = traj.monotonicity
        tier = disa.disagreement_tier

        # Base severity from signal state
        if not is_mono and tier == "weak":
            base = 0.9
        elif not is_mono and tier == "medium":
            base = 0.6
        elif not is_mono and tier == "low":
            base = 0.3
        elif is_mono and tier == "weak":
            base = 0.5
        elif is_mono and tier == "medium":
            base = 0.3
        else:  # monotone + low
            base = 0.1

        # Per-action modulation
        if action in (OrchestrationAction.RESPOND, OrchestrationAction.STOP):
            return 1.0 - base
        if action in (OrchestrationAction.VERIFY, OrchestrationAction.ESCALATE):
            return base
        if action == OrchestrationAction.REROLL:
            return base * 0.7
        if action == OrchestrationAction.RETRIEVE:
            return base * 0.5
        return 0.0  # pragma: no cover

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        """Ensure weight dict has all required keys."""
        required = {"gain", "confidence", "cost_penalty", "redundancy_penalty", "severity"}
        missing = required - set(self._weights.keys())
        if missing:
            raise ValueError(f"Missing utility weights: {missing}")


# ----------------------------------------------------------------------
# Stub (kept for backward compatibility)
# ----------------------------------------------------------------------


class _StubUtilityEngine:
    """Stub used during Step 1 (scaffold only). Retained for tests."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def score_all(
        self,
        traj_signal: TrajectorySignal,
        disa_signal: DisagreementSignal,
        step_context: Optional[dict] = None,
    ) -> list[ActionScore]:
        return [
            ActionScore(
                action=OrchestrationAction.RESPOND,
                utility=0.7,
                breakdown={},
            ),
        ]

    def select_best(self, scores: list[ActionScore]) -> OrchestrationAction:
        return OrchestrationAction.RESPOND
