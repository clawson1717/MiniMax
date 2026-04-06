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

from raptor.config import Config, OrchestrationAction
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

    Usage:
        engine = UtilityEngine(config)
        scores = engine.score_all(
            traj_signal=TrajectorySignal(...),
            disa_signal=DisagreementSignal(...),
            step_context={"has_retrieved": False, "n_rerolls": 0, ...}
        )
        best_action = engine.select_best(scores)
    """

    def __init__(self, config: Config) -> None:
        self.config = config.utility
        self._validate_weights()

    def score_all(
        self,
        traj_signal: TrajectorySignal,
        disa_signal: DisagreementSignal,
        step_context: Optional[dict] = None,
    ) -> list[ActionScore]:
        """Compute utility scores for all 6 actions.

        Args:
            traj_signal: Signal from Entropy Trajectory Tracker
            disa_signal: Signal from Disagreement Monitor
            step_context: Runtime context (n_rerolls, has_retrieved, step_num, ...)

        Returns:
            List of ActionScore for each action, sorted by utility descending
        """
        # TODO: implement Step 4
        # - Build signal dict from traj_signal + disa_signal
        # - For each action a:
        #     φ_gain(a, S)     = expected accuracy improvement if action succeeds
        #     φ_conf(a, S)     = current calibrated confidence
        #     φ_cost(a, S)     = step cost (token usage, latency)
        #     φ_redund(a, S)   = redundancy penalty (already retrieved?)
        #     φ_severity(a, S) = severity of getting it wrong
        # - Compute U(a|S) = Σ w_k · φ_k - cost(a)
        # - Return sorted list of ActionScore
        raise NotImplementedError("Step 4 — Utility Score Engine not yet implemented")

    def select_best(self, scores: list[ActionScore]) -> OrchestrationAction:
        """Select the action with highest utility score.

        Applies hysteresis: if current action is same as previous,
        require utility margin > switch_threshold to switch.
        """
        # TODO: implement Step 4
        raise NotImplementedError("Step 4 — Utility Score Engine not yet implemented")

    # -------------------------------------------------------------------------
    # Feature functions (implement in Step 4)
    # -------------------------------------------------------------------------

    def _phi_gain(
        self, action: OrchestrationAction, traj: TrajectorySignal, disa: DisagreementSignal
    ) -> float:
        """Expected gain: how much accuracy improvement does this action provide?

        REROLL gains most when: non-monotone AND weak-disagreement tier
        VERIFY gains most when: non-monotone AND medium-disagreement
        RESPOND gains most when: monotone AND low-disagreement
        """
        # TODO: Step 4
        raise NotImplementedError("Step 4")

    def _phi_confidence(
        self, action: OrchestrationAction, traj: TrajectorySignal, disa: DisagreementSignal
    ) -> float:
        """Current calibrated confidence from combined signals."""
        # DONE: straightforward combination
        return float(traj.confidence_score * 0.5 + disa.confidence_score * 0.5)

    def _phi_cost(
        self, action: OrchestrationAction, step_context: dict
    ) -> float:
        """Cost penalty: token usage and latency for this action."""
        # TODO: Step 4
        raise NotImplementedError("Step 4")

    def _phi_redundancy(
        self, action: OrchestrationAction, step_context: dict
    ) -> float:
        """Redundancy penalty: has this action already been taken?"""
        # TODO: Step 4
        raise NotImplementedError("Step 4")

    def _phi_severity(
        self, action: OrchestrationAction, traj: TrajectorySignal, disa: DisagreementSignal
    ) -> float:
        """Severity: how bad is getting this wrong?

        Higher severity when: non-monotone + weak disagreement (high stakes)
        """
        # TODO: Step 4
        raise NotImplementedError("Step 4")

    def _validate_weights(self) -> None:
        """Ensure weight dict has all required keys."""
        required = {"gain", "confidence", "cost_penalty", "redundancy_penalty", "severity"}
        missing = required - set(self.config.weights.keys())
        if missing:
            raise ValueError(f"Missing utility weights: {missing}")


# ----------------------------------------------------------------------
# Stub
# ----------------------------------------------------------------------
class _StubUtilityEngine:
    """Stub used during Step 1 (scaffold only). Remove after Step 4."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def score_all(
        self,
        traj_signal: TrajectorySignal,
        disa_signal: DisagreementSignal,
        step_context: Optional[dict] = None,
    ) -> list[ActionScore]:
        return [
            ActionScore(action=OrchestrationAction.RESPOND, utility=0.7, breakdown={}),
        ]

    def select_best(self, scores: list[ActionScore]) -> OrchestrationAction:
        return OrchestrationAction.RESPOND


UtilityEngine = _StubUtilityEngine  # type: ignore[assignment]
