"""RAPTOR Orchestrator — the main controller that fuses signals and selects actions.

This is the core of RAPTOR: it combines entropy trajectory signals and
disagreement signals into a unified signal vector, then uses the utility
engine to select the next orchestration action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from raptor.config import Config, OrchestrationAction
from raptor.entropy_tracker import EntropyTracker, TrajectorySignal
from raptor.disagreement import DisagreementMonitor, DisagreementSignal, AgentResponse
from raptor.utility import UtilityEngine, ActionScore


@dataclass
class OrchestratorState:
    """Mutable state maintained across reasoning steps."""

    step_num: int
    n_rerolls: int
    has_retrieved: bool
    current_action: OrchestrationAction
    signal_history: list[dict]  # Signal vector at each step (for replay)


@dataclass
class OrchestrationDecision:
    """Final output of the RAPTOR orchestrator for one step."""

    action: OrchestrationAction
    utility_score: float
    action_breakdown: dict[str, float]
    traj_signal: TrajectorySignal
    disa_signal: DisagreementSignal
    reason: str  # Human-readable explanation


class RAPTOROrchestrator:
    """Main RAPTOR controller.

    Combines:
      - Entropy Trajectory Tracker (monotonicity + slope)
      - Disagreement Monitor (DiscoUQ-style features)
      - Utility Engine (action selection)

    Usage:
        orchestrator = RAPTOROrchestrator(config)
        decision = orchestrator.step(
            prompt="Solve: 2x + 3 = 7",
            agent_responses=[...],  # from poll_agents()
        )
        if decision.action == OrchestrationAction.REROLL:
            ...
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.entropy_tracker = EntropyTracker(config.entropy)
        self.disagreement_monitor = DisagreementMonitor(config.disagreement)
        self.utility_engine = UtilityEngine(config)
        self._state = OrchestratorState(
            step_num=0,
            n_rerolls=0,
            has_retrieved=False,
            current_action=OrchestrationAction.RESPOND,
            signal_history=[],
        )
        self._log_dir = Path(config.log_dir)
        if config.log_signal_history:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def step(
        self,
        prompt: str,
        agent_responses: list[AgentResponse],
        step_context: Optional[dict] = None,
    ) -> OrchestrationDecision:
        """Run one orchestration step.

        Args:
            prompt: The reasoning prompt
            agent_responses: Multi-agent responses from poll_agents()
            step_context: Additional context (n_rerolls, has_retrieved, ...)

        Returns:
            OrchestrationDecision with selected action and signal history
        """
        # TODO: implement Step 5
        # 1. Update entropy tracker with agent responses
        # 2. Compute TrajectorySignal
        # 3. Compute DisagreementSignal
        # 4. Build step_context (merge with _state)
        # 5. Score all actions via utility_engine
        # 6. Select best action
        # 7. Update _state
        # 8. Log signal history
        # 9. Return OrchestrationDecision
        raise NotImplementedError("Step 5 — RAPTOR Orchestrator not yet implemented")

    def reroll_with_selection(
        self,
        agent_responses: list[AgentResponse],
        n_candidates: int = 3,
    ) -> AgentResponse:
        """Reroll and select the best candidate via disagreement monitoring.

        Args:
            agent_responses: Current agent responses
            n_candidates: Number of new candidates to generate

        Returns:
            The selected best candidate AgentResponse
        """
        # TODO: implement in Step 5 or Step 6
        raise NotImplementedError("Step 5 — reroll_with_selection not yet implemented")

    def reset(self) -> None:
        """Reset orchestrator state for a new reasoning session."""
        self.entropy_tracker.reset()
        self._state = OrchestratorState(
            step_num=0,
            n_rerolls=0,
            has_retrieved=False,
            current_action=OrchestrationAction.RESPOND,
            signal_history=[],
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_signals(
        self,
        decision: OrchestrationDecision,
        extra_context: Optional[dict] = None,
    ) -> None:
        """Write signal history to JSON log file."""
        if not self.config.log_signal_history:
            return
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": self._state.step_num,
            "action": decision.action.value,
            "utility": decision.utility_score,
            "traj_signal": {
                "monotonicity": decision.traj_signal.monotonicity,
                "entropy_slope": decision.traj_signal.entropy_slope,
                "confidence_score": decision.traj_signal.confidence_score,
            },
            "disa_signal": {
                "evidence_overlap": decision.disa_signal.evidence_overlap,
                "argument_strength": decision.disa_signal.argument_strength,
                "divergence_depth": decision.disa_signal.divergence_depth,
                "dispersion": decision.disa_signal.dispersion,
                "cohesion": decision.disa_signal.cohesion,
                "confidence_score": decision.disa_signal.confidence_score,
                "disagreement_tier": decision.disa_signal.disagreement_tier,
            },
            "breakdown": decision.action_breakdown,
            "reason": decision.reason,
        }
        if extra_context:
            record["context"] = extra_context

        log_file = self._log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
