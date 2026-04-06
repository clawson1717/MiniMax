"""RAPTOR Orchestrator — the main controller that fuses signals and selects actions.

This is the core of RAPTOR: it combines entropy trajectory signals and
disagreement signals into a unified signal vector S(t), then uses the
utility engine to select the next orchestration action.

Signal Fusion:
    S(t) = [monotonicity_flag, entropy_slope, disagreement_score,
            dispersion_score, cohesion_score, divergence_depth]

The orchestrator maintains step history for hysteresis (avoiding action
flickering) and emits structured JSONL logs for replay and analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from raptor.config import Config, OrchestrationAction
from raptor.entropy_tracker import EntropyTracker, TrajectorySignal
from raptor.disagreement import DisagreementMonitor, DisagreementSignal, AgentResponse
from raptor.utility import UtilityEngine, ActionScore


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

_SYNTH_VOCAB_SIZE = 256  # Hash bucket count for synthetic log-probability generation


@dataclass
class SignalVector:
    """Fused RAPTOR signal vector S(t) — entropy + disagreement combined.

    Fields map directly to the paper's signal vector definition::

        S(t) = [monotonicity_flag, entropy_slope, disagreement_score,
                dispersion_score, cohesion_score, divergence_depth]
    """

    monotonicity_flag: bool
    entropy_slope: float
    disagreement_score: float  # 1 − confidence_score (higher → more disagreement)
    dispersion_score: float
    cohesion_score: float
    divergence_depth: int

    def to_dict(self) -> dict:
        """Serialize for logging / replay."""
        return {
            "monotonicity_flag": self.monotonicity_flag,
            "entropy_slope": self.entropy_slope,
            "disagreement_score": self.disagreement_score,
            "dispersion_score": self.dispersion_score,
            "cohesion_score": self.cohesion_score,
            "divergence_depth": self.divergence_depth,
        }


@dataclass
class OrchestratorState:
    """Mutable state maintained across reasoning steps."""

    step_num: int
    n_rerolls: int
    has_retrieved: bool
    current_action: OrchestrationAction
    signal_history: list[dict]  # Signal vector + scores at each step (for replay)
    # Step-count hysteresis tracking
    pending_action: Optional[OrchestrationAction] = None
    pending_count: int = 0


@dataclass
class OrchestrationDecision:
    """Final output of the RAPTOR orchestrator for one step."""

    action: OrchestrationAction
    utility_score: float
    action_breakdown: dict[str, float]
    traj_signal: TrajectorySignal
    disa_signal: DisagreementSignal
    signal_vector: SignalVector
    all_scores: list[ActionScore]
    reason: str  # Human-readable explanation


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


class RAPTOROrchestrator:
    """Main RAPTOR controller.

    Combines:
      - Entropy Trajectory Tracker (monotonicity + slope)
      - Disagreement Monitor (DiscoUQ-style features)
      - Utility Engine (action selection)

    Usage::

        orchestrator = RAPTOROrchestrator(config)
        decision = orchestrator.step(
            prompt="Solve: 2x + 3 = 7",
            agent_responses=[...],  # from poll_agents()
        )
        if decision.action == OrchestrationAction.REROLL:
            # generate new reasoning chains
            ...

    The orchestrator maintains internal state across ``step()`` calls within
    a session.  Call ``reset()`` to start a new reasoning session.
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
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_dir = Path(config.log_dir)
        if config.log_signal_history:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> OrchestratorState:
        """Read-only access to orchestrator state."""
        return self._state

    @property
    def session_id(self) -> str:
        """Unique identifier for this orchestrator session."""
        return self._session_id

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(
        self,
        prompt: str,
        agent_responses: list[AgentResponse],
        step_context: Optional[dict] = None,
    ) -> OrchestrationDecision:
        """Run one orchestration step.

        Pipeline:
          1. Compute entropy trajectory signal (per-agent, then aggregate)
          2. Compute disagreement signal across agents
          3. Fuse into signal vector S(t)
          4. Build step context from internal state + user overrides
          5. Score all 6 actions via utility engine
          6. Select best action (threshold hysteresis from utility engine)
          7. Apply step-count hysteresis (orchestrator level)
          8. Update state, log, return decision

        Args:
            prompt: The reasoning prompt being processed.
            agent_responses: Multi-agent responses from ``poll_agents()``.
            step_context: Optional context overrides (keys: ``n_rerolls``,
                ``has_retrieved``, ``step_num``, ``max_rerolls``).

        Returns:
            :class:`OrchestrationDecision` with the selected action, utility
            scores, fused signals, and a human-readable reason string.

        Raises:
            ValueError: If *agent_responses* is empty.
        """
        if not agent_responses:
            raise ValueError("agent_responses must be non-empty")

        # 1. Compute entropy trajectory signal
        traj_signal = self._compute_entropy_signal(agent_responses)

        # 2. Compute disagreement signal
        disa_signal = self.disagreement_monitor.compute_signal(agent_responses)

        # 3. Fuse into signal vector S(t)
        signal_vector = self._fuse_signals(traj_signal, disa_signal)

        # 4. Build step context
        ctx = self._build_step_context(step_context)

        # 5. Score all actions via utility engine
        scores = self.utility_engine.score_all(traj_signal, disa_signal, ctx)

        # 6. Select best action (utility engine applies threshold hysteresis)
        proposed = self.utility_engine.select_best(
            scores, previous_action=self._state.current_action
        )

        # 7. Apply step-count hysteresis at orchestrator level
        action = self._apply_hysteresis(proposed)

        # 8. Build human-readable reason
        reason = self._build_reason(action, traj_signal, disa_signal, signal_vector)

        # 9. Get the score object for the chosen action
        chosen_score = next(
            (s for s in scores if s.action == action), scores[0]
        )

        # 10. Construct decision
        decision = OrchestrationDecision(
            action=action,
            utility_score=chosen_score.utility,
            action_breakdown=chosen_score.breakdown,
            traj_signal=traj_signal,
            disa_signal=disa_signal,
            signal_vector=signal_vector,
            all_scores=scores,
            reason=reason,
        )

        # 11. Update internal state
        self._update_state(action, signal_vector, scores)

        # 12. Structured logging
        self._log_signals(decision, extra_context={"prompt": prompt})

        # 13. Emit loguru event for observability
        logger.info(
            "RAPTOR step {step} -> {action} (U={utility:.3f}) | "
            "mono={mono} slope={slope:.3f} | "
            "tier={tier} conf={conf:.3f}",
            step=self._state.step_num,
            action=action.value,
            utility=chosen_score.utility,
            mono=traj_signal.monotonicity,
            slope=traj_signal.entropy_slope,
            tier=disa_signal.disagreement_tier,
            conf=disa_signal.confidence_score,
        )

        return decision

    def reroll_with_selection(
        self,
        agent_responses: list[AgentResponse],
        n_candidates: int = 3,
    ) -> AgentResponse:
        """Select the best candidate from agent responses via entropy signal.

        Evaluates each response's reasoning chain for entropy trajectory
        quality (monotonicity + confidence) and returns the one with the
        highest confidence score.

        In a full deployment this would first generate *n_candidates* new
        responses from the LLM (see Step 6 — Agent Protocol).  Here it
        operates purely as a selection mechanism over provided candidates.

        Args:
            agent_responses: Candidate agent responses to evaluate.
            n_candidates: Hint for number of candidates (all provided
                responses are evaluated regardless).

        Returns:
            The :class:`AgentResponse` with the highest entropy confidence.

        Raises:
            ValueError: If *agent_responses* is empty.
        """
        if not agent_responses:
            raise ValueError("No agent responses to select from")

        best_response: Optional[AgentResponse] = None
        best_confidence = -1.0

        for resp in agent_responses:
            tracker = EntropyTracker(self.config.entropy)
            for step_text in resp.reasoning_steps:
                log_probs = self._text_to_log_probs(step_text)
                tracker.update(log_probs)
            signal = tracker.compute_signal()

            if signal.confidence_score > best_confidence:
                best_confidence = signal.confidence_score
                best_response = resp

        logger.debug(
            "Reroll selection: chose agent {agent} (confidence={conf:.3f}) "
            "from {n} candidates",
            agent=best_response.agent_id,  # type: ignore[union-attr]
            conf=best_confidence,
            n=len(agent_responses),
        )

        return best_response  # type: ignore[return-value]

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
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # ------------------------------------------------------------------
    # Entropy signal computation
    # ------------------------------------------------------------------

    def _compute_entropy_signal(
        self, responses: list[AgentResponse]
    ) -> TrajectorySignal:
        """Compute aggregated entropy trajectory signal across all agents.

        For each agent:
          1. Create a fresh :class:`EntropyTracker`
          2. Convert each reasoning step to synthetic log-probabilities
          3. Feed through the tracker
          4. Compute per-agent :class:`TrajectorySignal`

        Aggregation rules (conservative):
          - ``monotonicity`` = ALL agents monotone
          - ``entropy_slope`` = mean across agents
          - ``confidence_score`` = mean across agents
          - ``final_entropy`` = mean across agents
          - ``entropies`` = element-wise mean (shorter trajectories padded
            with their last value)
        """
        if not responses:
            return TrajectorySignal(
                n_steps=0,
                monotonicity=True,
                entropy_slope=0.0,
                final_entropy=0.0,
                entropies=[],
                confidence_score=0.5,
            )

        agent_signals: list[TrajectorySignal] = []

        for resp in responses:
            tracker = EntropyTracker(self.config.entropy)
            for step_text in resp.reasoning_steps:
                log_probs = self._text_to_log_probs(step_text)
                tracker.update(log_probs)
            # Include final answer as the last step of the trajectory
            if resp.final_answer:
                log_probs = self._text_to_log_probs(resp.final_answer)
                tracker.update(log_probs)
            agent_signals.append(tracker.compute_signal())

        # Aggregate across agents
        all_mono = all(s.monotonicity for s in agent_signals)
        avg_slope = float(np.mean([s.entropy_slope for s in agent_signals]))
        avg_confidence = float(np.mean([s.confidence_score for s in agent_signals]))
        avg_final = float(np.mean([s.final_entropy for s in agent_signals]))

        # Aggregate entropy trajectories (pad shorter with last value, then mean)
        max_len = max(len(s.entropies) for s in agent_signals)
        if max_len == 0:
            avg_entropies: list[float] = []
        else:
            padded = []
            for s in agent_signals:
                ents = list(s.entropies)
                if ents:
                    ents.extend([ents[-1]] * (max_len - len(ents)))
                else:
                    ents = [0.0] * max_len
                padded.append(ents)
            avg_entropies = [
                float(np.mean([p[i] for p in padded])) for i in range(max_len)
            ]

        return TrajectorySignal(
            n_steps=max_len,
            monotonicity=all_mono,
            entropy_slope=avg_slope,
            final_entropy=avg_final,
            entropies=avg_entropies,
            confidence_score=avg_confidence,
        )

    @staticmethod
    def _text_to_log_probs(
        text: str, vocab_size: int = _SYNTH_VOCAB_SIZE
    ) -> np.ndarray:
        """Convert text to a synthetic log-probability distribution.

        Uses deterministic token-frequency hashing as a proxy for model
        confidence.  More focused/repetitive text → concentrated distribution
        → lower entropy.  Diverse text → spread distribution → higher entropy.

        Args:
            text: Raw text of one reasoning step or answer.
            vocab_size: Number of hash buckets (controls distribution granularity).

        Returns:
            Log-probability array of shape ``(vocab_size,)``.
        """
        tokens = text.lower().split()
        if not tokens:
            # Uniform distribution → maximum entropy
            return np.full(vocab_size, np.log(1.0 / vocab_size))

        freq = np.zeros(vocab_size, dtype=np.float64)
        for token in tokens:
            idx = hash(token) % vocab_size
            freq[idx] += 1

        # Laplace smoothing to avoid log(0)
        freq += 1e-6
        probs = freq / freq.sum()
        return np.log(probs)

    # ------------------------------------------------------------------
    # Signal fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_signals(
        traj: TrajectorySignal, disa: DisagreementSignal
    ) -> SignalVector:
        """Combine entropy trajectory + disagreement signals into S(t).

        Maps component signals to the unified vector::

            S(t) = [monotonicity_flag, entropy_slope, disagreement_score,
                    dispersion_score, cohesion_score, divergence_depth]

        ``disagreement_score`` is inverted from ``confidence_score`` so that
        higher values consistently indicate *more* concern.
        """
        return SignalVector(
            monotonicity_flag=traj.monotonicity,
            entropy_slope=traj.entropy_slope,
            disagreement_score=1.0 - disa.confidence_score,
            dispersion_score=disa.dispersion,
            cohesion_score=disa.cohesion,
            divergence_depth=disa.divergence_depth,
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_step_context(self, user_context: Optional[dict] = None) -> dict:
        """Merge internal orchestrator state with user-provided context.

        Internal state fields provide defaults; user_context keys override.
        """
        ctx: dict = {
            "n_rerolls": self._state.n_rerolls,
            "has_retrieved": self._state.has_retrieved,
            "step_num": self._state.step_num,
            "max_rerolls": self.config.max_rerolls,
            "previous_action": self._state.current_action,
        }
        if user_context:
            ctx.update(user_context)
        return ctx

    # ------------------------------------------------------------------
    # Hysteresis (step-count based)
    # ------------------------------------------------------------------

    def _apply_hysteresis(self, proposed: OrchestrationAction) -> OrchestrationAction:
        """Apply step-count hysteresis to prevent action flickering.

        The utility engine handles *threshold* hysteresis (the margin between
        the best and previous action must exceed ``switch_threshold``).

        This method adds *step-count* hysteresis: a different action must
        be proposed for ``hysteresis_steps`` consecutive ``step()`` calls
        before the switch actually takes effect.

        If ``hysteresis_steps <= 1``, the switch happens immediately (no
        additional delay beyond the utility engine's threshold check).
        """
        current = self._state.current_action
        steps_required = self.config.utility.hysteresis_steps

        if proposed == current:
            # Same as current — reset any pending switch
            self._state.pending_action = None
            self._state.pending_count = 0
            return current

        if steps_required <= 1:
            # No step-count hysteresis — switch immediately
            self._state.pending_action = None
            self._state.pending_count = 0
            return proposed

        # Track consecutive proposals for the same new action
        if proposed == self._state.pending_action:
            self._state.pending_count += 1
        else:
            self._state.pending_action = proposed
            self._state.pending_count = 1

        if self._state.pending_count >= steps_required:
            # Enough consecutive proposals — execute the switch
            self._state.pending_action = None
            self._state.pending_count = 0
            return proposed

        # Not enough consecutive proposals — stay with current action
        return current

    # ------------------------------------------------------------------
    # Reason generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reason(
        action: OrchestrationAction,
        traj: TrajectorySignal,
        disa: DisagreementSignal,
        sv: SignalVector,
    ) -> str:
        """Build a human-readable explanation for the chosen action."""
        parts: list[str] = []

        # Entropy trajectory assessment
        if traj.monotonicity:
            parts.append("Entropy trajectory is monotonically decreasing (good).")
        else:
            parts.append(
                f"Non-monotone entropy trajectory detected "
                f"(slope={traj.entropy_slope:.3f})."
            )

        # Disagreement assessment
        tier = disa.disagreement_tier
        if tier == "low":
            parts.append("Agent disagreement is low — high consensus.")
        elif tier == "medium":
            parts.append("Medium agent disagreement — some uncertainty.")
        else:
            parts.append("Weak agreement among agents — high uncertainty.")

        # Combined confidence
        combined_conf = 0.5 * traj.confidence_score + 0.5 * disa.confidence_score
        parts.append(f"Combined confidence: {combined_conf:.2f}.")

        # Action rationale
        _rationale = {
            OrchestrationAction.RESPOND: "Committing to current answer.",
            OrchestrationAction.REROLL: "Generating new reasoning chain(s).",
            OrchestrationAction.VERIFY: "Triggering verification step.",
            OrchestrationAction.ESCALATE: "Escalating to stronger model/human.",
            OrchestrationAction.RETRIEVE: "Retrieving additional context.",
            OrchestrationAction.STOP: "Halting — insufficient confidence to proceed.",
        }
        parts.append(
            f"Action: {action.value} — {_rationale.get(action, 'Unknown action.')}"
        )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _update_state(
        self,
        action: OrchestrationAction,
        signal_vector: SignalVector,
        scores: list[ActionScore],
    ) -> None:
        """Update internal state after a completed step."""
        self._state.step_num += 1
        self._state.current_action = action

        if action == OrchestrationAction.REROLL:
            self._state.n_rerolls += 1
        if action == OrchestrationAction.RETRIEVE:
            self._state.has_retrieved = True

        self._state.signal_history.append(
            {
                "step": self._state.step_num,
                "signal_vector": signal_vector.to_dict(),
                "action": action.value,
                "scores": {s.action.value: round(s.utility, 6) for s in scores},
            }
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_signals(
        self,
        decision: OrchestrationDecision,
        extra_context: Optional[dict] = None,
    ) -> None:
        """Append a structured JSON record to the session-level log file.

        Each session writes to a single JSONL file:
        ``{log_dir}/session_{session_id}.jsonl``

        Records contain the full signal vector, trajectory and disagreement
        details, utility breakdown, and the chosen action — enabling offline
        replay and post-hoc analysis.
        """
        if not self.config.log_signal_history:
            return

        record: dict = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "step": self._state.step_num,
            "action": decision.action.value,
            "utility": decision.utility_score,
            "signal_vector": decision.signal_vector.to_dict(),
            "traj_signal": {
                "n_steps": decision.traj_signal.n_steps,
                "monotonicity": decision.traj_signal.monotonicity,
                "entropy_slope": decision.traj_signal.entropy_slope,
                "final_entropy": decision.traj_signal.final_entropy,
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

        log_file = self._log_dir / f"session_{self._session_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
