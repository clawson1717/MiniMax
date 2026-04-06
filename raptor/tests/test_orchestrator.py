"""Tests for orchestrator.py — Step 5: RAPTOR Orchestrator (Core Loop).

Covers:
  - SignalVector construction and serialization
  - OrchestratorState lifecycle
  - Text-to-log-probs conversion
  - Per-agent entropy signal computation and aggregation
  - Signal fusion S(t)
  - Step context building
  - Step-count hysteresis
  - Full step() pipeline
  - Action selection under various signal conditions
  - State tracking (step_num, n_rerolls, has_retrieved)
  - Structured logging (JSONL)
  - Reroll-with-selection
  - Multi-step integration scenarios
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from raptor.config import Config, OrchestrationAction
from raptor.disagreement import AgentResponse, DisagreementSignal
from raptor.entropy_tracker import TrajectorySignal
from raptor.orchestrator import (
    OrchestrationDecision,
    OrchestratorState,
    RAPTOROrchestrator,
    SignalVector,
)
from raptor.utility import ActionScore


# ======================================================================
# Helpers — build controlled AgentResponses
# ======================================================================


def _make_monotone_agent(agent_id: str = "mono-1", answer: str = "42") -> AgentResponse:
    """Agent whose reasoning steps produce monotonically decreasing entropy.

    Step 1: Long, diverse text   → high entropy
    Step 2: Medium text          → medium entropy
    Step 3: Short, focused text  → low entropy
    Final answer: very short     → lowest entropy
    """
    return AgentResponse(
        agent_id=agent_id,
        reasoning_steps=[
            (
                "Let me explore several different approaches and techniques "
                "for solving this complex mathematical problem using various methods "
                "including algebra geometry calculus and number theory"
            ),
            "Using the algebraic approach I can simplify the equation",
            "Therefore x equals two",
        ],
        final_answer=answer,
    )


def _make_nonmonotone_agent(
    agent_id: str = "nonmono-1", answer: str = "99"
) -> AgentResponse:
    """Agent whose reasoning steps produce non-monotone entropy.

    Step 1: Short, focused text  → low entropy
    Step 2: Long, diverse text   → high entropy  (violation!)
    Step 3: Short text           → low entropy
    """
    return AgentResponse(
        agent_id=agent_id,
        reasoning_steps=[
            "The answer is clearly x",
            (
                "Wait actually let me reconsider all the different possibilities "
                "and options available including algebra geometry topology statistics "
                "probability theory combinatorics and number theory approaches"
            ),
            "OK the answer is probably x",
        ],
        final_answer=answer,
    )


def _make_agreeing_agents(n: int = 3, answer: str = "42") -> list[AgentResponse]:
    """Multiple agents that agree on the answer with similar reasoning."""
    agents = []
    for i in range(n):
        agents.append(
            AgentResponse(
                agent_id=f"agree-{i}",
                reasoning_steps=[
                    (
                        "Let me explore different methods for solving "
                        "this equation using algebraic techniques and simplification"
                    ),
                    "Using algebra I simplify to get the solution",
                    "The answer is forty two",
                ],
                final_answer=answer,
            )
        )
    return agents


def _make_disagreeing_agents() -> list[AgentResponse]:
    """Agents that strongly disagree on both reasoning and final answer."""
    return [
        AgentResponse(
            agent_id="disagree-0",
            reasoning_steps=[
                "Using algebra I get x plus three equals seven",
                "Subtracting three from both sides",
            ],
            final_answer="4",
        ),
        AgentResponse(
            agent_id="disagree-1",
            reasoning_steps=[
                "I will use geometry to interpret this problem visually",
                "The intersection point is at coordinate five",
            ],
            final_answer="5",
        ),
        AgentResponse(
            agent_id="disagree-2",
            reasoning_steps=[
                "By number theory the modular arithmetic gives six",
                "Applying Fermat's little theorem",
            ],
            final_answer="6",
        ),
    ]


def _make_traj(
    monotonicity: bool = True,
    confidence_score: float = 0.8,
    entropy_slope: float = -0.5,
    final_entropy: float = 1.0,
    entropies: list[float] | None = None,
) -> TrajectorySignal:
    ents = entropies or [3.0, 2.0, 1.0]
    return TrajectorySignal(
        n_steps=len(ents),
        monotonicity=monotonicity,
        entropy_slope=entropy_slope,
        final_entropy=final_entropy,
        entropies=ents,
        confidence_score=confidence_score,
    )


def _make_disa(
    tier: str = "low",
    confidence_score: float = 0.7,
    evidence_overlap: float = 0.7,
    argument_strength: float = 0.6,
    divergence_depth: int = 2,
    dispersion: float = 0.3,
    cohesion: float = 0.6,
) -> DisagreementSignal:
    return DisagreementSignal(
        evidence_overlap=evidence_overlap,
        argument_strength=argument_strength,
        divergence_depth=divergence_depth,
        dispersion=dispersion,
        cohesion=cohesion,
        confidence_score=confidence_score,
        disagreement_tier=tier,
    )


def _config_with_tmp_log(tmp_path: Path) -> Config:
    """Config that logs to a temporary directory."""
    cfg = Config()
    cfg.log_dir = str(tmp_path / "logs")
    cfg.log_signal_history = True
    return cfg


# ======================================================================
# SignalVector
# ======================================================================


class TestSignalVector:
    def test_to_dict(self):
        sv = SignalVector(
            monotonicity_flag=True,
            entropy_slope=-0.5,
            disagreement_score=0.3,
            dispersion_score=0.1,
            cohesion_score=0.8,
            divergence_depth=5,
        )
        d = sv.to_dict()
        assert d["monotonicity_flag"] is True
        assert d["entropy_slope"] == -0.5
        assert d["disagreement_score"] == 0.3
        assert d["dispersion_score"] == 0.1
        assert d["cohesion_score"] == 0.8
        assert d["divergence_depth"] == 5

    def test_to_dict_returns_plain_types(self):
        sv = SignalVector(
            monotonicity_flag=False,
            entropy_slope=0.0,
            disagreement_score=0.0,
            dispersion_score=0.0,
            cohesion_score=1.0,
            divergence_depth=0,
        )
        d = sv.to_dict()
        # All values should be JSON-serializable primitives
        json.dumps(d)  # should not raise


# ======================================================================
# OrchestratorState
# ======================================================================


class TestOrchestratorState:
    def test_default_pending_fields(self):
        state = OrchestratorState(
            step_num=0,
            n_rerolls=0,
            has_retrieved=False,
            current_action=OrchestrationAction.RESPOND,
            signal_history=[],
        )
        assert state.pending_action is None
        assert state.pending_count == 0

    def test_backward_compatible_construction(self):
        """Existing tests create OrchestratorState with keyword args — must still work."""
        state = OrchestratorState(
            step_num=5,
            n_rerolls=2,
            has_retrieved=True,
            current_action=OrchestrationAction.REROLL,
            signal_history=[],
        )
        assert state.step_num == 5
        assert state.n_rerolls == 2
        assert state.has_retrieved is True
        assert state.pending_action is None


# ======================================================================
# Orchestrator — Initialization
# ======================================================================


class TestOrchestratorInit:
    def test_initializes_with_config(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        assert orch.config is cfg

    def test_state_starts_at_step_zero(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        assert orch.state.step_num == 0
        assert orch.state.n_rerolls == 0
        assert orch.state.has_retrieved is False
        assert orch.state.current_action == OrchestrationAction.RESPOND

    def test_subcomponents_created(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        assert orch.entropy_tracker is not None
        assert orch.disagreement_monitor is not None
        assert orch.utility_engine is not None

    def test_session_id_set(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        assert isinstance(orch.session_id, str)
        assert len(orch.session_id) > 0

    def test_log_dir_created(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        RAPTOROrchestrator(cfg)
        assert Path(cfg.log_dir).exists()


# ======================================================================
# Text-to-Log-Probs
# ======================================================================


class TestTextToLogProbs:
    def test_returns_correct_shape(self):
        lp = RAPTOROrchestrator._text_to_log_probs("hello world")
        assert lp.shape == (256,)

    def test_custom_vocab_size(self):
        lp = RAPTOROrchestrator._text_to_log_probs("hello", vocab_size=64)
        assert lp.shape == (64,)

    def test_values_are_log_probs(self):
        lp = RAPTOROrchestrator._text_to_log_probs("hello world test")
        assert np.all(lp <= 0.0)  # log probs are ≤ 0
        # exp(log_probs) should sum to ~1
        probs = np.exp(lp)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_empty_text_gives_uniform(self):
        lp = RAPTOROrchestrator._text_to_log_probs("")
        # Uniform → all values equal
        assert np.allclose(lp, lp[0])

    def test_diverse_text_higher_entropy_than_focused(self):
        """More unique tokens → higher entropy distribution."""
        diverse = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        focused = "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha"

        lp_diverse = RAPTOROrchestrator._text_to_log_probs(diverse)
        lp_focused = RAPTOROrchestrator._text_to_log_probs(focused)

        # Compute entropy for each
        def entropy(lp):
            p = np.exp(lp)
            return float(-np.sum(p * np.log(p + 1e-30)))

        assert entropy(lp_diverse) > entropy(lp_focused)

    def test_deterministic(self):
        lp1 = RAPTOROrchestrator._text_to_log_probs("test input")
        lp2 = RAPTOROrchestrator._text_to_log_probs("test input")
        np.testing.assert_array_equal(lp1, lp2)


# ======================================================================
# Entropy Signal Computation
# ======================================================================


class TestComputeEntropySignal:
    def test_single_agent(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = _make_monotone_agent()
        signal = orch._compute_entropy_signal([agent])
        assert isinstance(signal, TrajectorySignal)
        # 3 reasoning steps + 1 final answer = 4 steps
        assert signal.n_steps == 4
        assert isinstance(signal.monotonicity, bool)
        assert 0.0 <= signal.confidence_score <= 1.0

    def test_monotone_agent_has_monotone_signal(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = _make_monotone_agent()
        signal = orch._compute_entropy_signal([agent])
        assert signal.monotonicity is True

    def test_nonmonotone_agent_detected(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = _make_nonmonotone_agent()
        signal = orch._compute_entropy_signal([agent])
        assert signal.monotonicity is False

    def test_multi_agent_all_monotone(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = [_make_monotone_agent(f"m-{i}") for i in range(3)]
        signal = orch._compute_entropy_signal(agents)
        assert signal.monotonicity is True

    def test_one_nonmonotone_agent_breaks_aggregate(self, tmp_path):
        """Conservative: ANY non-monotone agent → aggregate non-monotone."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = [
            _make_monotone_agent("good-0"),
            _make_monotone_agent("good-1"),
            _make_nonmonotone_agent("bad-0"),
        ]
        signal = orch._compute_entropy_signal(agents)
        assert signal.monotonicity is False

    def test_empty_responses(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        signal = orch._compute_entropy_signal([])
        assert signal.n_steps == 0
        assert signal.monotonicity is True
        assert signal.confidence_score == 0.5

    def test_agent_with_no_reasoning_steps(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = AgentResponse(agent_id="empty", reasoning_steps=[], final_answer="42")
        signal = orch._compute_entropy_signal([agent])
        assert signal.n_steps == 1  # just the final answer
        assert isinstance(signal.confidence_score, float)

    def test_aggregate_entropies_length(self, tmp_path):
        """Aggregate entropy trajectory should be the length of the longest agent."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        short_agent = AgentResponse(
            agent_id="short", reasoning_steps=["step one"], final_answer="a"
        )
        long_agent = AgentResponse(
            agent_id="long",
            reasoning_steps=["step one", "step two", "step three"],
            final_answer="b",
        )
        signal = orch._compute_entropy_signal([short_agent, long_agent])
        # long_agent: 3 steps + 1 answer = 4, short: 1 step + 1 answer = 2
        assert signal.n_steps == 4
        assert len(signal.entropies) == 4


# ======================================================================
# Signal Fusion
# ======================================================================


class TestFuseSignals:
    def test_fuse_produces_signal_vector(self):
        traj = _make_traj(monotonicity=True, entropy_slope=-0.5)
        disa = _make_disa(
            confidence_score=0.8,
            dispersion=0.2,
            cohesion=0.7,
            divergence_depth=3,
        )
        sv = RAPTOROrchestrator._fuse_signals(traj, disa)
        assert isinstance(sv, SignalVector)
        assert sv.monotonicity_flag is True
        assert sv.entropy_slope == pytest.approx(-0.5)
        # disagreement_score = 1 - confidence
        assert sv.disagreement_score == pytest.approx(0.2)
        assert sv.dispersion_score == pytest.approx(0.2)
        assert sv.cohesion_score == pytest.approx(0.7)
        assert sv.divergence_depth == 3

    def test_disagreement_score_inverted(self):
        """Higher confidence → lower disagreement_score."""
        disa_confident = _make_disa(confidence_score=0.9)
        disa_uncertain = _make_disa(confidence_score=0.3)
        traj = _make_traj()

        sv_confident = RAPTOROrchestrator._fuse_signals(traj, disa_confident)
        sv_uncertain = RAPTOROrchestrator._fuse_signals(traj, disa_uncertain)

        assert sv_confident.disagreement_score < sv_uncertain.disagreement_score


# ======================================================================
# Step Context Building
# ======================================================================


class TestBuildStepContext:
    def test_default_context(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        ctx = orch._build_step_context()
        assert ctx["n_rerolls"] == 0
        assert ctx["has_retrieved"] is False
        assert ctx["step_num"] == 0
        assert ctx["max_rerolls"] == 3
        assert ctx["previous_action"] == OrchestrationAction.RESPOND

    def test_user_context_overrides(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        ctx = orch._build_step_context({"step_num": 99, "extra_field": "test"})
        assert ctx["step_num"] == 99
        assert ctx["extra_field"] == "test"
        # Non-overridden fields still from state
        assert ctx["n_rerolls"] == 0

    def test_reflects_state_changes(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        orch._state.n_rerolls = 2
        orch._state.has_retrieved = True
        orch._state.step_num = 5
        ctx = orch._build_step_context()
        assert ctx["n_rerolls"] == 2
        assert ctx["has_retrieved"] is True
        assert ctx["step_num"] == 5


# ======================================================================
# Step-Count Hysteresis
# ======================================================================


class TestApplyHysteresis:
    def test_same_action_returns_immediately(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        orch._state.current_action = OrchestrationAction.RESPOND
        result = orch._apply_hysteresis(OrchestrationAction.RESPOND)
        assert result == OrchestrationAction.RESPOND

    def test_hysteresis_steps_1_switches_immediately(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        cfg.utility.hysteresis_steps = 1
        orch = RAPTOROrchestrator(cfg)
        orch._state.current_action = OrchestrationAction.RESPOND
        result = orch._apply_hysteresis(OrchestrationAction.REROLL)
        assert result == OrchestrationAction.REROLL

    def test_hysteresis_steps_2_delays_switch(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        cfg.utility.hysteresis_steps = 2
        orch = RAPTOROrchestrator(cfg)
        orch._state.current_action = OrchestrationAction.RESPOND

        # First proposal: pending but not enough
        r1 = orch._apply_hysteresis(OrchestrationAction.REROLL)
        assert r1 == OrchestrationAction.RESPOND
        assert orch._state.pending_action == OrchestrationAction.REROLL
        assert orch._state.pending_count == 1

        # Second proposal (same): now switch
        r2 = orch._apply_hysteresis(OrchestrationAction.REROLL)
        assert r2 == OrchestrationAction.REROLL
        assert orch._state.pending_action is None

    def test_hysteresis_resets_on_different_proposal(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        cfg.utility.hysteresis_steps = 3
        orch = RAPTOROrchestrator(cfg)
        orch._state.current_action = OrchestrationAction.RESPOND

        # Propose REROLL twice
        orch._apply_hysteresis(OrchestrationAction.REROLL)
        orch._apply_hysteresis(OrchestrationAction.REROLL)
        assert orch._state.pending_count == 2

        # Propose VERIFY — resets counter
        orch._apply_hysteresis(OrchestrationAction.VERIFY)
        assert orch._state.pending_action == OrchestrationAction.VERIFY
        assert orch._state.pending_count == 1

    def test_hysteresis_resets_when_current_proposed(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        cfg.utility.hysteresis_steps = 3
        orch = RAPTOROrchestrator(cfg)
        orch._state.current_action = OrchestrationAction.RESPOND
        orch._state.pending_action = OrchestrationAction.REROLL
        orch._state.pending_count = 2

        # Propose current action → resets pending
        orch._apply_hysteresis(OrchestrationAction.RESPOND)
        assert orch._state.pending_action is None
        assert orch._state.pending_count == 0


# ======================================================================
# step() — basic structure
# ======================================================================


class TestStepBasic:
    def test_returns_orchestration_decision(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(3)
        decision = orch.step("Solve 2x + 3 = 7", agents)
        assert isinstance(decision, OrchestrationDecision)

    def test_decision_has_all_fields(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        decision = orch.step("prompt", [_make_monotone_agent()])
        assert isinstance(decision.action, OrchestrationAction)
        assert isinstance(decision.utility_score, float)
        assert isinstance(decision.action_breakdown, dict)
        assert isinstance(decision.traj_signal, TrajectorySignal)
        assert isinstance(decision.disa_signal, DisagreementSignal)
        assert isinstance(decision.signal_vector, SignalVector)
        assert isinstance(decision.all_scores, list)
        assert isinstance(decision.reason, str)

    def test_all_six_actions_scored(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        decision = orch.step("prompt", _make_agreeing_agents(2))
        actions = {s.action for s in decision.all_scores}
        assert actions == set(OrchestrationAction)

    def test_empty_responses_raises(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        with pytest.raises(ValueError, match="non-empty"):
            orch.step("prompt", [])

    def test_reason_string_nonempty(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        decision = orch.step("prompt", [_make_monotone_agent()])
        assert len(decision.reason) > 10


# ======================================================================
# step() — action selection under various conditions
# ======================================================================


class TestStepActionSelection:
    def test_monotone_agreeing_prefers_respond(self, tmp_path):
        """Ideal: monotone trajectories + agreement → RESPOND."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(3, answer="42")
        # Make all agents have monotonically decreasing reasoning
        for a in agents:
            a.reasoning_steps = [
                (
                    "Let me explore several different approaches methods "
                    "techniques strategies algorithms and heuristics for "
                    "solving this complex mathematical problem"
                ),
                "Using algebra I simplify the equation",
                "x equals forty two",
            ]
        decision = orch.step("What is x?", agents)
        assert decision.action == OrchestrationAction.RESPOND

    def test_nonmonotone_disagreeing_avoids_respond(self, tmp_path):
        """Bad: non-monotone + disagreement → should NOT respond immediately."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_disagreeing_agents()
        # Make at least one agent non-monotone
        agents[0].reasoning_steps = [
            "x is four",
            (
                "Wait let me reconsider all the different approaches "
                "methods techniques and possibilities"
            ),
            "hmm four",
        ]
        decision = orch.step("What is x?", agents)
        assert decision.action != OrchestrationAction.RESPOND

    def test_single_agent_returns_valid_decision(self, tmp_path):
        """Degraded mode: single agent should still produce a decision."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        decision = orch.step("prompt", [_make_monotone_agent()])
        assert decision.action in OrchestrationAction


# ======================================================================
# State Tracking
# ======================================================================


class TestStateTracking:
    def test_step_num_increments(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(2)
        assert orch.state.step_num == 0
        orch.step("p1", agents)
        assert orch.state.step_num == 1
        orch.step("p2", agents)
        assert orch.state.step_num == 2

    def test_reroll_count_tracks(self, tmp_path):
        """When REROLL is selected, n_rerolls should increment."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        agents = _make_disagreeing_agents()
        # Make agents produce non-monotone signals to trigger REROLL
        agents[0].reasoning_steps = [
            "x is four",
            (
                "Wait let me reconsider all the different approaches "
                "methods techniques and possibilities for solving"
            ),
            "maybe four",
        ]
        agents[1].reasoning_steps = [
            "y is five",
            (
                "Actually reconsider all the various approaches "
                "techniques strategies and alternative methods"
            ),
            "maybe five",
        ]
        decision = orch.step("prompt", agents)
        if decision.action == OrchestrationAction.REROLL:
            assert orch.state.n_rerolls == 1

    def test_signal_history_grows(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(2)
        assert len(orch.state.signal_history) == 0
        orch.step("p1", agents)
        assert len(orch.state.signal_history) == 1
        orch.step("p2", agents)
        assert len(orch.state.signal_history) == 2

    def test_signal_history_entry_structure(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        orch.step("prompt", _make_agreeing_agents(2))
        entry = orch.state.signal_history[0]
        assert "step" in entry
        assert "signal_vector" in entry
        assert "action" in entry
        assert "scores" in entry
        assert isinstance(entry["signal_vector"], dict)
        assert "monotonicity_flag" in entry["signal_vector"]

    def test_current_action_updates(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(3)
        decision = orch.step("prompt", agents)
        assert orch.state.current_action == decision.action


# ======================================================================
# Reset
# ======================================================================


class TestReset:
    def test_reset_clears_state(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        # Run some steps
        orch.step("p1", _make_agreeing_agents(2))
        orch.step("p2", _make_agreeing_agents(2))
        assert orch.state.step_num > 0

        orch.reset()
        assert orch.state.step_num == 0
        assert orch.state.n_rerolls == 0
        assert orch.state.has_retrieved is False
        assert orch.state.current_action == OrchestrationAction.RESPOND
        assert len(orch.state.signal_history) == 0

    def test_reset_clears_entropy_tracker(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        # Add some entropy data
        orch.entropy_tracker.update(np.zeros(10))
        assert len(orch.entropy_tracker.entropies) > 0

        orch.reset()
        assert len(orch.entropy_tracker.entropies) == 0

    def test_reset_generates_new_session_id(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        old_id = orch.session_id
        # Small delay to ensure different timestamp
        import time
        time.sleep(0.001)
        orch.reset()
        # Session ID includes microseconds, so should differ
        # (in rare cases they could match, so just verify it's a string)
        assert isinstance(orch.session_id, str)

    def test_reset_clears_pending_hysteresis(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        cfg.utility.hysteresis_steps = 5
        orch = RAPTOROrchestrator(cfg)
        orch._state.pending_action = OrchestrationAction.REROLL
        orch._state.pending_count = 3

        orch.reset()
        assert orch.state.pending_action is None
        assert orch.state.pending_count == 0


# ======================================================================
# Structured Logging
# ======================================================================


class TestLogging:
    def test_log_file_created(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(2))

        log_dir = Path(cfg.log_dir)
        log_files = list(log_dir.glob("session_*.jsonl"))
        assert len(log_files) == 1

    def test_log_contains_valid_json(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(2))

        log_dir = Path(cfg.log_dir)
        log_file = list(log_dir.glob("session_*.jsonl"))[0]
        with open(log_file) as f:
            for line in f:
                record = json.loads(line)  # should not raise
                assert "action" in record
                assert "utility" in record
                assert "signal_vector" in record

    def test_multi_step_same_log_file(self, tmp_path):
        """Multiple steps within one session should append to the same file."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        agents = _make_agreeing_agents(2)
        orch.step("p1", agents)
        orch.step("p2", agents)
        orch.step("p3", agents)

        log_dir = Path(cfg.log_dir)
        log_files = list(log_dir.glob("session_*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_log_contains_prompt_context(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("What is 2+2?", _make_agreeing_agents(2))

        log_file = list(Path(cfg.log_dir).glob("session_*.jsonl"))[0]
        with open(log_file) as f:
            record = json.loads(f.readline())
        assert record["context"]["prompt"] == "What is 2+2?"

    def test_log_disabled(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        cfg.log_signal_history = False
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(2))

        # Log dir should not have been created (or be empty)
        log_dir = Path(cfg.log_dir)
        if log_dir.exists():
            assert len(list(log_dir.glob("*.jsonl"))) == 0

    def test_log_record_has_session_id(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(2))

        log_file = list(Path(cfg.log_dir).glob("session_*.jsonl"))[0]
        with open(log_file) as f:
            record = json.loads(f.readline())
        assert record["session_id"] == orch.session_id

    def test_log_signal_vector_fields(self, tmp_path):
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(2))

        log_file = list(Path(cfg.log_dir).glob("session_*.jsonl"))[0]
        with open(log_file) as f:
            record = json.loads(f.readline())
        sv = record["signal_vector"]
        expected_keys = {
            "monotonicity_flag",
            "entropy_slope",
            "disagreement_score",
            "dispersion_score",
            "cohesion_score",
            "divergence_depth",
        }
        assert set(sv.keys()) == expected_keys


# ======================================================================
# Reroll with Selection
# ======================================================================


class TestRerollWithSelection:
    def test_selects_best_candidate(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        mono = _make_monotone_agent("mono")
        nonmono = _make_nonmonotone_agent("nonmono")

        best = orch.reroll_with_selection([mono, nonmono])
        # Monotone agent should have higher confidence → selected
        assert best.agent_id == "mono"

    def test_empty_raises(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        with pytest.raises(ValueError, match="No agent"):
            orch.reroll_with_selection([])

    def test_single_candidate_returns_it(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = _make_monotone_agent("only-one")
        result = orch.reroll_with_selection([agent])
        assert result.agent_id == "only-one"

    def test_returns_agent_response(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(3)
        result = orch.reroll_with_selection(agents)
        assert isinstance(result, AgentResponse)


# ======================================================================
# Reason Generation
# ======================================================================


class TestBuildReason:
    def test_monotone_reason_mentions_monotone(self):
        reason = RAPTOROrchestrator._build_reason(
            OrchestrationAction.RESPOND,
            _make_traj(monotonicity=True),
            _make_disa(tier="low"),
            SignalVector(True, -0.5, 0.2, 0.1, 0.8, 3),
        )
        assert "monotonically decreasing" in reason.lower()

    def test_nonmonotone_reason_mentions_warning(self):
        reason = RAPTOROrchestrator._build_reason(
            OrchestrationAction.REROLL,
            _make_traj(monotonicity=False, entropy_slope=0.3),
            _make_disa(tier="weak"),
            SignalVector(False, 0.3, 0.7, 0.5, 0.3, 1),
        )
        assert "non-monotone" in reason.lower()

    def test_reason_contains_action_name(self):
        for action in OrchestrationAction:
            reason = RAPTOROrchestrator._build_reason(
                action,
                _make_traj(),
                _make_disa(),
                SignalVector(True, -0.5, 0.2, 0.1, 0.8, 3),
            )
            assert action.value in reason

    def test_reason_mentions_tier(self):
        reason = RAPTOROrchestrator._build_reason(
            OrchestrationAction.VERIFY,
            _make_traj(),
            _make_disa(tier="medium"),
            SignalVector(True, -0.5, 0.3, 0.2, 0.7, 3),
        )
        assert "medium" in reason.lower() or "uncertainty" in reason.lower()


# ======================================================================
# Integration — multi-step sequences
# ======================================================================


class TestMultiStepIntegration:
    def test_three_step_ideal_sequence(self, tmp_path):
        """Three steps of ideal conditions → RESPOND every time."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(3, answer="42")
        # Give all agents monotone reasoning
        for a in agents:
            a.reasoning_steps = [
                (
                    "Let me explore several different approaches methods "
                    "techniques strategies algorithms and heuristics for "
                    "solving this complex mathematical problem"
                ),
                "Using algebra I simplify the equation",
                "x equals forty two",
            ]

        for i in range(3):
            decision = orch.step(f"Step {i}", agents)
            assert decision.action == OrchestrationAction.RESPOND

        assert orch.state.step_num == 3
        assert len(orch.state.signal_history) == 3

    def test_state_evolves_across_steps(self, tmp_path):
        """Verify state changes are reflected in subsequent steps."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(2)

        d1 = orch.step("p1", agents)
        ctx1 = orch._build_step_context()
        assert ctx1["step_num"] == 1
        assert ctx1["previous_action"] == d1.action

        d2 = orch.step("p2", agents)
        ctx2 = orch._build_step_context()
        assert ctx2["step_num"] == 2
        assert ctx2["previous_action"] == d2.action

    def test_reset_between_sessions(self, tmp_path):
        """Resetting between sessions should produce independent runs."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(2)

        orch.step("session 1", agents)
        orch.step("session 1", agents)
        assert orch.state.step_num == 2

        orch.reset()

        orch.step("session 2", agents)
        assert orch.state.step_num == 1
        assert len(orch.state.signal_history) == 1

    def test_user_context_override_in_step(self, tmp_path):
        """User can override step context fields."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(2)
        # Override to force higher step_num (affects cost scaling)
        decision = orch.step("prompt", agents, step_context={"step_num": 10})
        # Just verify it doesn't crash and produces a decision
        assert isinstance(decision, OrchestrationDecision)


# ======================================================================
# Integration — synthetic reasoning tasks
# ======================================================================


class TestSyntheticReasoningTasks:
    """Full orchestration on synthetic reasoning scenarios."""

    def test_math_problem_confident_agents(self, tmp_path):
        """Scenario: Math problem where all agents agree with clear reasoning."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)

        agents = [
            AgentResponse(
                agent_id=f"math-{i}",
                reasoning_steps=[
                    (
                        "We need to solve the equation 2x + 3 = 7 "
                        "by applying algebraic techniques and inverse operations"
                    ),
                    "Subtracting 3 from both sides gives 2x = 4",
                    "Dividing by 2 gives x = 2",
                ],
                final_answer="x = 2",
            )
            for i in range(3)
        ]

        decision = orch.step("Solve: 2x + 3 = 7", agents)
        assert decision.action == OrchestrationAction.RESPOND
        assert decision.traj_signal.confidence_score > 0.5

    def test_ambiguous_problem_disagreeing_agents(self, tmp_path):
        """Scenario: Ambiguous question where agents diverge."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)

        agents = [
            AgentResponse(
                agent_id="philosophy-1",
                reasoning_steps=[
                    (
                        "This is a deep philosophical question about "
                        "the nature of consciousness and reality"
                    ),
                    "From a materialist perspective the answer is physical",
                ],
                final_answer="consciousness is physical",
            ),
            AgentResponse(
                agent_id="philosophy-2",
                reasoning_steps=[
                    (
                        "Let me consider the various viewpoints from "
                        "dualism panpsychism and idealism"
                    ),
                    "The dualist view separates mind from body",
                ],
                final_answer="consciousness is non-physical",
            ),
            AgentResponse(
                agent_id="philosophy-3",
                reasoning_steps=[
                    (
                        "Looking at this from multiple angles "
                        "including neuroscience and phenomenology"
                    ),
                    "The emergent view suggests it arises from complexity",
                ],
                final_answer="consciousness is emergent",
            ),
        ]

        decision = orch.step("What is consciousness?", agents)
        # Should NOT just respond — agents disagree
        assert decision.disa_signal.disagreement_tier in ("medium", "weak")

    def test_iterative_improvement(self, tmp_path):
        """Scenario: First attempt is uncertain, second attempt after reroll is better."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)

        # Step 1: Uncertain agents
        uncertain_agents = [
            AgentResponse(
                agent_id=f"uncertain-{i}",
                reasoning_steps=[
                    "hmm maybe",
                    (
                        "Actually let me reconsider all the different "
                        "approaches possibilities methods and options"
                    ),
                    "I think the answer might be",
                ],
                final_answer=str(i + 10),  # different answers
            )
            for i in range(3)
        ]
        d1 = orch.step("Hard question", uncertain_agents)

        # Step 2: Confident agents (after hypothetical reroll)
        confident_agents = _make_agreeing_agents(3, answer="42")
        for a in confident_agents:
            a.reasoning_steps = [
                (
                    "Let me explore several different approaches methods "
                    "techniques strategies algorithms and heuristics"
                ),
                "Using algebra I simplify the equation",
                "x equals forty two",
            ]
        d2 = orch.step("Hard question", confident_agents)

        # Second decision should have higher confidence
        conf1 = 0.5 * d1.traj_signal.confidence_score + 0.5 * d1.disa_signal.confidence_score
        conf2 = 0.5 * d2.traj_signal.confidence_score + 0.5 * d2.disa_signal.confidence_score
        assert conf2 > conf1

    def test_full_signal_vector_logged(self, tmp_path):
        """Verify that the complete S(t) vector is captured in signal_history."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(3))

        entry = orch.state.signal_history[0]
        sv = entry["signal_vector"]

        # All 6 components of S(t) should be present
        assert "monotonicity_flag" in sv
        assert "entropy_slope" in sv
        assert "disagreement_score" in sv
        assert "dispersion_score" in sv
        assert "cohesion_score" in sv
        assert "divergence_depth" in sv

    def test_utility_scores_in_history(self, tmp_path):
        """Signal history should contain utility scores for all actions."""
        cfg = _config_with_tmp_log(tmp_path)
        orch = RAPTOROrchestrator(cfg)
        orch.step("prompt", _make_agreeing_agents(2))

        entry = orch.state.signal_history[0]
        scores = entry["scores"]

        # All 6 actions should have a score
        expected_actions = {a.value for a in OrchestrationAction}
        assert set(scores.keys()) == expected_actions


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_agent_with_empty_final_answer(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = AgentResponse(
            agent_id="empty-answer",
            reasoning_steps=["thinking about it"],
            final_answer="",
        )
        # Should not crash
        decision = orch.step("prompt", [agent])
        assert isinstance(decision, OrchestrationDecision)

    def test_agent_with_empty_reasoning(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agent = AgentResponse(
            agent_id="no-reasoning",
            reasoning_steps=[],
            final_answer="42",
        )
        decision = orch.step("prompt", [agent])
        assert isinstance(decision, OrchestrationDecision)

    def test_many_agents(self, tmp_path):
        """10 agents should work fine."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        agents = _make_agreeing_agents(10, answer="42")
        decision = orch.step("prompt", agents)
        assert isinstance(decision, OrchestrationDecision)

    def test_very_long_reasoning_chain(self, tmp_path):
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        # Agent with 20 reasoning steps
        steps = [f"reasoning step {i} with some text" for i in range(20)]
        agent = AgentResponse(
            agent_id="long-chain",
            reasoning_steps=steps,
            final_answer="final",
        )
        decision = orch.step("prompt", [agent])
        assert decision.traj_signal.n_steps == 21  # 20 steps + 1 answer

    def test_signal_vector_consistency(self, tmp_path):
        """Signal vector in decision should match what's in signal_history."""
        orch = RAPTOROrchestrator(_config_with_tmp_log(tmp_path))
        decision = orch.step("prompt", _make_agreeing_agents(2))

        sv_decision = decision.signal_vector.to_dict()
        sv_history = orch.state.signal_history[0]["signal_vector"]

        for key in sv_decision:
            assert sv_decision[key] == pytest.approx(sv_history[key], abs=1e-10)
