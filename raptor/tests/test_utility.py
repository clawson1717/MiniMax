"""Tests for utility.py — Step 4: Utility Score Engine."""

import pytest

from raptor.utility import UtilityEngine, _StubUtilityEngine, ActionScore
from raptor.config import Config, OrchestrationAction, UtilityConfig
from raptor.entropy_tracker import TrajectorySignal
from raptor.disagreement import DisagreementSignal


# ======================================================================
# Helpers
# ======================================================================


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


def _default_context(**overrides) -> dict:
    ctx = {
        "n_rerolls": 0,
        "has_retrieved": False,
        "step_num": 0,
        "max_rerolls": 3,
        "previous_action": None,
    }
    ctx.update(overrides)
    return ctx


# ======================================================================
# Backward compatibility — _StubUtilityEngine
# ======================================================================


class TestStubUtilityEngine:
    """Ensure the stub still works (backward compat)."""

    def test_stub_scores_all_returns_list(self):
        config = Config()
        engine = _StubUtilityEngine(config)
        scores = engine.score_all(_make_traj(), _make_disa())
        assert isinstance(scores, list)
        assert len(scores) >= 1
        assert isinstance(scores[0], ActionScore)

    def test_stub_select_best_returns_respond(self):
        config = Config()
        engine = _StubUtilityEngine(config)
        scores = [
            ActionScore(action=OrchestrationAction.REROLL, utility=0.8, breakdown={}),
            ActionScore(action=OrchestrationAction.RESPOND, utility=0.7, breakdown={}),
        ]
        best = engine.select_best(scores)
        assert best == OrchestrationAction.RESPOND


class TestOrchestrationActionEnum:
    """Tests for the OrchestrationAction enum."""

    def test_all_actions_defined(self):
        expected = {"respond", "reroll", "verify", "escalate", "retrieve", "stop"}
        actual = {a.value for a in OrchestrationAction}
        assert expected == actual

    def test_action_values_are_lowercase(self):
        for action in OrchestrationAction:
            assert action.value == action.value.lower()


# ======================================================================
# UtilityEngine — initialization
# ======================================================================


class TestUtilityEngineInit:
    def test_default_init(self):
        engine = UtilityEngine(Config())
        assert engine.mode == "fixed"
        assert set(engine.weights.keys()) == {
            "gain", "confidence", "cost_penalty", "redundancy_penalty", "severity",
        }

    def test_learned_mode(self):
        engine = UtilityEngine(Config(), mode="learned")
        assert engine.mode == "learned"

    def test_missing_weights_raises(self):
        cfg = Config()
        cfg.utility.weights = {"gain": 0.3}  # missing others
        with pytest.raises(ValueError, match="Missing utility weights"):
            UtilityEngine(cfg)

    def test_weights_are_copies(self):
        """Modifying returned weights dict doesn't mutate engine."""
        engine = UtilityEngine(Config())
        w = engine.weights
        w["gain"] = 999.0
        assert engine.weights["gain"] != 999.0


# ======================================================================
# Feature functions — φ_gain
# ======================================================================


class TestPhiGain:
    """Test expected-gain feature function."""

    def _gain(self, action, traj, disa):
        engine = UtilityEngine(Config())
        return engine._phi_gain(action, traj, disa)

    # RESPOND
    def test_respond_monotone_low(self):
        g = self._gain(OrchestrationAction.RESPOND, _make_traj(True), _make_disa("low"))
        assert g == pytest.approx(0.9)

    def test_respond_monotone_medium(self):
        g = self._gain(OrchestrationAction.RESPOND, _make_traj(True), _make_disa("medium"))
        assert g == pytest.approx(0.6)

    def test_respond_nonmonotone_low(self):
        g = self._gain(OrchestrationAction.RESPOND, _make_traj(False), _make_disa("low"))
        assert g == pytest.approx(0.3)

    def test_respond_nonmonotone_weak(self):
        g = self._gain(OrchestrationAction.RESPOND, _make_traj(False), _make_disa("weak"))
        assert g == pytest.approx(0.1)

    # REROLL
    def test_reroll_nonmonotone_weak(self):
        g = self._gain(OrchestrationAction.REROLL, _make_traj(False), _make_disa("weak"))
        assert g == pytest.approx(0.9)

    def test_reroll_nonmonotone_medium(self):
        g = self._gain(OrchestrationAction.REROLL, _make_traj(False), _make_disa("medium"))
        assert g == pytest.approx(0.5)

    def test_reroll_monotone_weak(self):
        g = self._gain(OrchestrationAction.REROLL, _make_traj(True), _make_disa("weak"))
        assert g == pytest.approx(0.4)

    def test_reroll_monotone_low(self):
        g = self._gain(OrchestrationAction.REROLL, _make_traj(True), _make_disa("low"))
        assert g == pytest.approx(0.15)

    # VERIFY
    def test_verify_nonmonotone_medium(self):
        g = self._gain(OrchestrationAction.VERIFY, _make_traj(False), _make_disa("medium"))
        assert g == pytest.approx(0.9)

    def test_verify_nonmonotone_weak(self):
        g = self._gain(OrchestrationAction.VERIFY, _make_traj(False), _make_disa("weak"))
        assert g == pytest.approx(0.5)

    def test_verify_monotone_medium(self):
        g = self._gain(OrchestrationAction.VERIFY, _make_traj(True), _make_disa("medium"))
        assert g == pytest.approx(0.4)

    def test_verify_monotone_low(self):
        g = self._gain(OrchestrationAction.VERIFY, _make_traj(True), _make_disa("low"))
        assert g == pytest.approx(0.15)

    # ESCALATE
    def test_escalate_nonmonotone_weak(self):
        g = self._gain(OrchestrationAction.ESCALATE, _make_traj(False), _make_disa("weak"))
        assert g == pytest.approx(0.7)

    def test_escalate_monotone_low(self):
        g = self._gain(OrchestrationAction.ESCALATE, _make_traj(True), _make_disa("low"))
        assert g == pytest.approx(0.1)

    # RETRIEVE — depends on combined confidence
    def test_retrieve_low_confidence(self):
        traj = _make_traj(confidence_score=0.2)
        disa = _make_disa(confidence_score=0.2)  # combined = 0.2
        g = self._gain(OrchestrationAction.RETRIEVE, traj, disa)
        assert g == pytest.approx(0.6)

    def test_retrieve_medium_confidence(self):
        traj = _make_traj(confidence_score=0.6)
        disa = _make_disa(confidence_score=0.6)  # combined = 0.6
        g = self._gain(OrchestrationAction.RETRIEVE, traj, disa)
        assert g == pytest.approx(0.3)

    def test_retrieve_high_confidence(self):
        traj = _make_traj(confidence_score=0.9)
        disa = _make_disa(confidence_score=0.9)  # combined = 0.9
        g = self._gain(OrchestrationAction.RETRIEVE, traj, disa)
        assert g == pytest.approx(0.1)

    # STOP
    def test_stop_very_low_confidence(self):
        traj = _make_traj(confidence_score=0.1)
        disa = _make_disa(confidence_score=0.1)  # combined = 0.1
        g = self._gain(OrchestrationAction.STOP, traj, disa)
        assert g == pytest.approx(0.3)

    def test_stop_normal_confidence(self):
        g = self._gain(OrchestrationAction.STOP, _make_traj(), _make_disa())
        assert g == pytest.approx(0.05)


# ======================================================================
# Feature functions — φ_confidence
# ======================================================================


class TestPhiConfidence:
    def test_combined_average(self):
        engine = UtilityEngine(Config())
        traj = _make_traj(confidence_score=0.8)
        disa = _make_disa(confidence_score=0.6)
        val = engine._phi_confidence(OrchestrationAction.RESPOND, traj, disa)
        assert val == pytest.approx(0.7)

    def test_same_for_all_actions(self):
        engine = UtilityEngine(Config())
        traj = _make_traj(confidence_score=0.5)
        disa = _make_disa(confidence_score=0.5)
        values = {
            a: engine._phi_confidence(a, traj, disa) for a in OrchestrationAction
        }
        assert len(set(values.values())) == 1
        assert list(values.values())[0] == pytest.approx(0.5)


# ======================================================================
# Feature functions — φ_cost
# ======================================================================


class TestPhiCost:
    def test_zero_cost_actions(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(step_num=0)
        assert engine._phi_cost(OrchestrationAction.RESPOND, ctx) == pytest.approx(0.0)
        assert engine._phi_cost(OrchestrationAction.STOP, ctx) == pytest.approx(0.0)

    def test_nonzero_cost_actions(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(step_num=0)
        assert engine._phi_cost(OrchestrationAction.REROLL, ctx) == pytest.approx(0.5)
        assert engine._phi_cost(OrchestrationAction.VERIFY, ctx) == pytest.approx(0.3)
        assert engine._phi_cost(OrchestrationAction.RETRIEVE, ctx) == pytest.approx(0.4)
        assert engine._phi_cost(OrchestrationAction.ESCALATE, ctx) == pytest.approx(1.0)

    def test_cost_scales_with_step(self):
        engine = UtilityEngine(Config())
        ctx0 = _default_context(step_num=0)
        ctx10 = _default_context(step_num=10)
        cost0 = engine._phi_cost(OrchestrationAction.REROLL, ctx0)
        cost10 = engine._phi_cost(OrchestrationAction.REROLL, ctx10)
        assert cost10 > cost0
        # step=10: factor = 1 + 10/20 = 1.5 → 0.5 * 1.5 = 0.75
        assert cost10 == pytest.approx(0.75)

    def test_cost_capped_at_one(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(step_num=100)
        cost = engine._phi_cost(OrchestrationAction.ESCALATE, ctx)
        assert cost == pytest.approx(1.0)


# ======================================================================
# Feature functions — φ_redundancy
# ======================================================================


class TestPhiRedundancy:
    def test_retrieve_not_yet_done(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(has_retrieved=False)
        assert engine._phi_redundancy(OrchestrationAction.RETRIEVE, ctx) == 0.0

    def test_retrieve_already_done(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(has_retrieved=True)
        assert engine._phi_redundancy(OrchestrationAction.RETRIEVE, ctx) == 1.0

    def test_reroll_zero_of_three(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(n_rerolls=0, max_rerolls=3)
        assert engine._phi_redundancy(OrchestrationAction.REROLL, ctx) == pytest.approx(0.0)

    def test_reroll_one_of_three(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(n_rerolls=1, max_rerolls=3)
        assert engine._phi_redundancy(OrchestrationAction.REROLL, ctx) == pytest.approx(1 / 3)

    def test_reroll_at_budget(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(n_rerolls=3, max_rerolls=3)
        assert engine._phi_redundancy(OrchestrationAction.REROLL, ctx) == 1.0

    def test_reroll_over_budget(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(n_rerolls=5, max_rerolls=3)
        assert engine._phi_redundancy(OrchestrationAction.REROLL, ctx) == 1.0

    def test_reroll_zero_max_with_rerolls(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(n_rerolls=1, max_rerolls=0)
        assert engine._phi_redundancy(OrchestrationAction.REROLL, ctx) == 1.0

    def test_reroll_zero_max_zero_used(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(n_rerolls=0, max_rerolls=0)
        assert engine._phi_redundancy(OrchestrationAction.REROLL, ctx) == 0.0

    def test_other_actions_zero(self):
        engine = UtilityEngine(Config())
        ctx = _default_context(has_retrieved=True, n_rerolls=3)
        for action in [
            OrchestrationAction.RESPOND,
            OrchestrationAction.VERIFY,
            OrchestrationAction.ESCALATE,
            OrchestrationAction.STOP,
        ]:
            assert engine._phi_redundancy(action, ctx) == 0.0


# ======================================================================
# Feature functions — φ_severity
# ======================================================================


class TestPhiSeverity:
    def _sev(self, action, mono, tier):
        engine = UtilityEngine(Config())
        return engine._phi_severity(action, _make_traj(mono), _make_disa(tier))

    # Base severity table
    def test_nonmonotone_weak_is_highest_base(self):
        # VERIFY passes through base directly
        assert self._sev(OrchestrationAction.VERIFY, False, "weak") == pytest.approx(0.9)

    def test_monotone_low_is_lowest_base(self):
        assert self._sev(OrchestrationAction.VERIFY, True, "low") == pytest.approx(0.1)

    # RESPOND / STOP invert severity
    def test_respond_high_severity_inverted(self):
        # non-mono+weak → base=0.9 → respond=1-0.9=0.1
        assert self._sev(OrchestrationAction.RESPOND, False, "weak") == pytest.approx(0.1)

    def test_respond_low_severity_inverted(self):
        # mono+low → base=0.1 → respond=1-0.1=0.9
        assert self._sev(OrchestrationAction.RESPOND, True, "low") == pytest.approx(0.9)

    def test_stop_same_as_respond(self):
        assert self._sev(OrchestrationAction.STOP, False, "weak") == pytest.approx(0.1)
        assert self._sev(OrchestrationAction.STOP, True, "low") == pytest.approx(0.9)

    # ESCALATE same as VERIFY (passes through base)
    def test_escalate_passes_base(self):
        assert self._sev(OrchestrationAction.ESCALATE, False, "weak") == pytest.approx(0.9)
        assert self._sev(OrchestrationAction.ESCALATE, True, "low") == pytest.approx(0.1)

    # REROLL = base * 0.7
    def test_reroll_scaled(self):
        assert self._sev(OrchestrationAction.REROLL, False, "weak") == pytest.approx(0.63)

    # RETRIEVE = base * 0.5
    def test_retrieve_scaled(self):
        assert self._sev(OrchestrationAction.RETRIEVE, False, "weak") == pytest.approx(0.45)

    # Middle tiers
    def test_nonmonotone_medium(self):
        assert self._sev(OrchestrationAction.VERIFY, False, "medium") == pytest.approx(0.6)

    def test_monotone_weak(self):
        assert self._sev(OrchestrationAction.VERIFY, True, "weak") == pytest.approx(0.5)

    def test_monotone_medium(self):
        assert self._sev(OrchestrationAction.VERIFY, True, "medium") == pytest.approx(0.3)

    def test_nonmonotone_low(self):
        assert self._sev(OrchestrationAction.VERIFY, False, "low") == pytest.approx(0.3)


# ======================================================================
# score_all
# ======================================================================


class TestScoreAll:
    def test_returns_all_six_actions(self):
        engine = UtilityEngine(Config())
        scores = engine.score_all(_make_traj(), _make_disa(), _default_context())
        actions = {s.action for s in scores}
        assert actions == set(OrchestrationAction)

    def test_sorted_descending(self):
        engine = UtilityEngine(Config())
        scores = engine.score_all(_make_traj(), _make_disa(), _default_context())
        utilities = [s.utility for s in scores]
        assert utilities == sorted(utilities, reverse=True)

    def test_breakdown_has_all_features(self):
        engine = UtilityEngine(Config())
        scores = engine.score_all(_make_traj(), _make_disa(), _default_context())
        expected_keys = {"gain", "confidence", "cost_penalty", "redundancy_penalty", "severity"}
        for s in scores:
            assert set(s.breakdown.keys()) == expected_keys

    def test_monotone_low_prefers_respond(self):
        """Ideal conditions: monotone + low disagreement → RESPOND wins."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=True, confidence_score=0.9)
        disa = _make_disa(tier="low", confidence_score=0.9)
        scores = engine.score_all(traj, disa, _default_context())
        assert scores[0].action == OrchestrationAction.RESPOND

    def test_nonmonotone_weak_prefers_reroll(self):
        """Bad signals: non-monotone + weak disagreement → REROLL wins."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.3)
        disa = _make_disa(tier="weak", confidence_score=0.3)
        scores = engine.score_all(traj, disa, _default_context())
        # REROLL should be top (highest gain in this regime)
        assert scores[0].action == OrchestrationAction.REROLL

    def test_nonmonotone_medium_prefers_verify(self):
        """Medium disagreement + non-monotone → VERIFY wins."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.5)
        disa = _make_disa(tier="medium", confidence_score=0.5)
        scores = engine.score_all(traj, disa, _default_context())
        assert scores[0].action == OrchestrationAction.VERIFY

    def test_nil_step_context(self):
        """score_all should work with step_context=None."""
        engine = UtilityEngine(Config())
        scores = engine.score_all(_make_traj(), _make_disa(), None)
        assert len(scores) == 6

    def test_cost_reduces_utility(self):
        """Actions with higher cost should have lower utility, all else equal."""
        engine = UtilityEngine(Config())
        scores = engine.score_all(_make_traj(), _make_disa(), _default_context())
        score_map = {s.action: s.utility for s in scores}
        # RESPOND cost=0, ESCALATE cost=1.0
        # In identical signal conditions RESPOND should beat ESCALATE
        assert score_map[OrchestrationAction.RESPOND] > score_map[OrchestrationAction.ESCALATE]

    def test_redundancy_penalizes_retrieve(self):
        """After retrieval, RETRIEVE should drop in utility."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=True, confidence_score=0.4)
        disa = _make_disa(tier="low", confidence_score=0.4)

        ctx_before = _default_context(has_retrieved=False)
        ctx_after = _default_context(has_retrieved=True)

        scores_before = engine.score_all(traj, disa, ctx_before)
        scores_after = engine.score_all(traj, disa, ctx_after)

        util_before = next(s.utility for s in scores_before if s.action == OrchestrationAction.RETRIEVE)
        util_after = next(s.utility for s in scores_after if s.action == OrchestrationAction.RETRIEVE)
        assert util_after < util_before

    def test_redundancy_penalizes_reroll_at_budget(self):
        """At max rerolls, REROLL utility drops."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.3)
        disa = _make_disa(tier="weak", confidence_score=0.3)

        ctx_fresh = _default_context(n_rerolls=0)
        ctx_exhausted = _default_context(n_rerolls=3, max_rerolls=3)

        scores_fresh = engine.score_all(traj, disa, ctx_fresh)
        scores_exhausted = engine.score_all(traj, disa, ctx_exhausted)

        util_fresh = next(s.utility for s in scores_fresh if s.action == OrchestrationAction.REROLL)
        util_exhausted = next(s.utility for s in scores_exhausted if s.action == OrchestrationAction.REROLL)
        assert util_exhausted < util_fresh


# ======================================================================
# select_best
# ======================================================================


class TestSelectBest:
    def test_returns_highest_utility(self):
        engine = UtilityEngine(Config())
        scores = [
            ActionScore(OrchestrationAction.VERIFY, 0.8, {}),
            ActionScore(OrchestrationAction.RESPOND, 0.5, {}),
            ActionScore(OrchestrationAction.REROLL, 0.3, {}),
        ]
        assert engine.select_best(scores) == OrchestrationAction.VERIFY

    def test_no_previous_action(self):
        engine = UtilityEngine(Config())
        scores = [
            ActionScore(OrchestrationAction.REROLL, 0.9, {}),
            ActionScore(OrchestrationAction.RESPOND, 0.85, {}),
        ]
        assert engine.select_best(scores) == OrchestrationAction.REROLL

    def test_empty_scores_returns_respond(self):
        engine = UtilityEngine(Config())
        assert engine.select_best([]) == OrchestrationAction.RESPOND

    def test_same_previous_no_hysteresis(self):
        """If best == previous, return it immediately."""
        engine = UtilityEngine(Config())
        scores = [
            ActionScore(OrchestrationAction.RESPOND, 0.9, {}),
            ActionScore(OrchestrationAction.REROLL, 0.5, {}),
        ]
        result = engine.select_best(scores, previous_action=OrchestrationAction.RESPOND)
        assert result == OrchestrationAction.RESPOND


class TestHysteresis:
    """Hysteresis: require margin > switch_threshold to switch actions."""

    def test_switch_when_margin_exceeds_threshold(self):
        cfg = Config()
        cfg.utility.switch_threshold = 0.05
        engine = UtilityEngine(cfg)
        scores = [
            ActionScore(OrchestrationAction.VERIFY, 0.9, {}),
            ActionScore(OrchestrationAction.RESPOND, 0.7, {}),
        ]
        # Margin = 0.2 > 0.05 → should switch
        result = engine.select_best(scores, previous_action=OrchestrationAction.RESPOND)
        assert result == OrchestrationAction.VERIFY

    def test_no_switch_when_margin_below_threshold(self):
        cfg = Config()
        cfg.utility.switch_threshold = 0.1
        engine = UtilityEngine(cfg)
        scores = [
            ActionScore(OrchestrationAction.VERIFY, 0.76, {}),
            ActionScore(OrchestrationAction.RESPOND, 0.72, {}),
        ]
        # Margin = 0.04 < 0.1 → stick with previous
        result = engine.select_best(scores, previous_action=OrchestrationAction.RESPOND)
        assert result == OrchestrationAction.RESPOND

    def test_no_switch_when_margin_equals_threshold(self):
        cfg = Config()
        cfg.utility.switch_threshold = 0.15
        engine = UtilityEngine(cfg)
        scores = [
            ActionScore(OrchestrationAction.VERIFY, 0.80, {}),
            ActionScore(OrchestrationAction.RESPOND, 0.70, {}),
        ]
        # Margin = 0.10 <= 0.15 threshold → stick with previous
        result = engine.select_best(scores, previous_action=OrchestrationAction.RESPOND)
        assert result == OrchestrationAction.RESPOND

    def test_previous_not_in_scores(self):
        """If previous action isn't scored, switch freely."""
        engine = UtilityEngine(Config())
        scores = [
            ActionScore(OrchestrationAction.VERIFY, 0.9, {}),
            ActionScore(OrchestrationAction.REROLL, 0.5, {}),
        ]
        result = engine.select_best(scores, previous_action=OrchestrationAction.ESCALATE)
        assert result == OrchestrationAction.VERIFY

    def test_zero_threshold_always_switches(self):
        cfg = Config()
        cfg.utility.switch_threshold = 0.0
        engine = UtilityEngine(cfg)
        scores = [
            ActionScore(OrchestrationAction.VERIFY, 0.701, {}),
            ActionScore(OrchestrationAction.RESPOND, 0.700, {}),
        ]
        result = engine.select_best(scores, previous_action=OrchestrationAction.RESPOND)
        assert result == OrchestrationAction.VERIFY


# ======================================================================
# Edge cases — all signals high / low / mixed
# ======================================================================


class TestEdgeCases:
    def test_all_signals_high(self):
        """High confidence everywhere → RESPOND should dominate."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=True, confidence_score=1.0)
        disa = _make_disa(tier="low", confidence_score=1.0)
        scores = engine.score_all(traj, disa, _default_context())
        assert scores[0].action == OrchestrationAction.RESPOND

    def test_all_signals_low(self):
        """Very low confidence + non-monotone + weak → REROLL should dominate."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.1)
        disa = _make_disa(tier="weak", confidence_score=0.1)
        scores = engine.score_all(traj, disa, _default_context())
        assert scores[0].action == OrchestrationAction.REROLL

    def test_mixed_monotone_weak(self):
        """Monotone but weak disagreement — somewhat uncertain."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=True, confidence_score=0.7)
        disa = _make_disa(tier="weak", confidence_score=0.4)
        scores = engine.score_all(traj, disa, _default_context())
        # Should prefer RESPOND or REROLL (both have decent gain here)
        top = scores[0].action
        assert top in (
            OrchestrationAction.RESPOND,
            OrchestrationAction.REROLL,
        )

    def test_mixed_nonmonotone_low(self):
        """Non-monotone but low disagreement."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.5)
        disa = _make_disa(tier="low", confidence_score=0.8)
        scores = engine.score_all(traj, disa, _default_context())
        top = scores[0].action
        # Non-monotone+low: VERIFY/REROLL get 0.5 gain; RESPOND gets 0.3
        assert top in (
            OrchestrationAction.VERIFY,
            OrchestrationAction.REROLL,
            OrchestrationAction.RESPOND,
        )


# ======================================================================
# Monotone vs non-monotone effects on action preference
# ======================================================================


class TestMonotoneEffects:
    def test_monotone_boosts_respond(self):
        engine = UtilityEngine(Config())
        disa = _make_disa(tier="low", confidence_score=0.7)

        scores_mono = engine.score_all(
            _make_traj(monotonicity=True, confidence_score=0.8), disa, _default_context()
        )
        scores_nonmono = engine.score_all(
            _make_traj(monotonicity=False, confidence_score=0.8), disa, _default_context()
        )

        respond_mono = next(s.utility for s in scores_mono if s.action == OrchestrationAction.RESPOND)
        respond_nonmono = next(s.utility for s in scores_nonmono if s.action == OrchestrationAction.RESPOND)
        assert respond_mono > respond_nonmono

    def test_nonmonotone_boosts_reroll(self):
        engine = UtilityEngine(Config())
        disa = _make_disa(tier="weak", confidence_score=0.3)

        scores_mono = engine.score_all(
            _make_traj(monotonicity=True, confidence_score=0.3), disa, _default_context()
        )
        scores_nonmono = engine.score_all(
            _make_traj(monotonicity=False, confidence_score=0.3), disa, _default_context()
        )

        reroll_mono = next(s.utility for s in scores_mono if s.action == OrchestrationAction.REROLL)
        reroll_nonmono = next(s.utility for s in scores_nonmono if s.action == OrchestrationAction.REROLL)
        assert reroll_nonmono > reroll_mono


# ======================================================================
# Disagreement tier effects
# ======================================================================


class TestDisagreementTierEffects:
    def test_weak_tier_boosts_reroll_over_verify(self):
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.3)
        disa = _make_disa(tier="weak", confidence_score=0.3)
        scores = engine.score_all(traj, disa, _default_context())
        score_map = {s.action: s.utility for s in scores}
        assert score_map[OrchestrationAction.REROLL] > score_map[OrchestrationAction.VERIFY]

    def test_medium_tier_boosts_verify_over_reroll(self):
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.5)
        disa = _make_disa(tier="medium", confidence_score=0.5)
        scores = engine.score_all(traj, disa, _default_context())
        score_map = {s.action: s.utility for s in scores}
        assert score_map[OrchestrationAction.VERIFY] > score_map[OrchestrationAction.REROLL]

    def test_low_tier_boosts_respond(self):
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=True, confidence_score=0.9)
        disa_low = _make_disa(tier="low", confidence_score=0.9)
        disa_weak = _make_disa(tier="weak", confidence_score=0.3)

        respond_low = next(
            s.utility
            for s in engine.score_all(traj, disa_low, _default_context())
            if s.action == OrchestrationAction.RESPOND
        )
        respond_weak = next(
            s.utility
            for s in engine.score_all(traj, disa_weak, _default_context())
            if s.action == OrchestrationAction.RESPOND
        )
        assert respond_low > respond_weak


# ======================================================================
# Learned-weight mode (online SGD)
# ======================================================================


class TestLearnedWeights:
    def test_fixed_mode_ignores_update(self):
        engine = UtilityEngine(Config(), mode="fixed")
        original = engine.weights.copy()
        engine.update_weights(
            features={"gain": 1.0, "confidence": 0.5, "cost_penalty": 0.3,
                       "redundancy_penalty": 0.0, "severity": 0.5},
            reward=1.0,
            predicted=0.5,
        )
        assert engine.weights == original

    def test_learned_mode_updates_weights(self):
        engine = UtilityEngine(Config(), mode="learned")
        original_gain = engine.weights["gain"]
        engine.update_weights(
            features={"gain": 1.0, "confidence": 0.5, "cost_penalty": 0.3,
                       "redundancy_penalty": 0.0, "severity": 0.5},
            reward=1.0,
            predicted=0.5,
            learning_rate=0.1,
        )
        # reward > predicted → error > 0 → gain feature * error > 0 → weight increases
        assert engine.weights["gain"] > original_gain

    def test_learned_mode_negative_error(self):
        engine = UtilityEngine(Config(), mode="learned")
        original_gain = engine.weights["gain"]
        engine.update_weights(
            features={"gain": 1.0, "confidence": 0.5, "cost_penalty": 0.3,
                       "redundancy_penalty": 0.0, "severity": 0.5},
            reward=0.0,
            predicted=0.5,
            learning_rate=0.1,
        )
        # reward < predicted → error < 0 → weight decreases
        assert engine.weights["gain"] < original_gain

    def test_learned_weights_affect_scoring(self):
        """After SGD update, scoring should use updated weights."""
        engine = UtilityEngine(Config(), mode="learned")
        traj = _make_traj(monotonicity=True, confidence_score=0.9)
        disa = _make_disa(tier="low", confidence_score=0.9)
        ctx = _default_context()

        scores_before = engine.score_all(traj, disa, ctx)
        respond_before = next(s.utility for s in scores_before if s.action == OrchestrationAction.RESPOND)

        # Massive positive update to gain weight
        engine.update_weights(
            features={"gain": 1.0, "confidence": 0.0, "cost_penalty": 0.0,
                       "redundancy_penalty": 0.0, "severity": 0.0},
            reward=1.0,
            predicted=0.0,
            learning_rate=0.5,
        )

        scores_after = engine.score_all(traj, disa, ctx)
        respond_after = next(s.utility for s in scores_after if s.action == OrchestrationAction.RESPOND)
        assert respond_after > respond_before

    def test_multiple_sgd_updates(self):
        engine = UtilityEngine(Config(), mode="learned")
        features = {
            "gain": 0.8, "confidence": 0.5, "cost_penalty": 0.2,
            "redundancy_penalty": 0.0, "severity": 0.3,
        }
        # Run multiple updates
        for _ in range(10):
            engine.update_weights(features, reward=1.0, predicted=0.3, learning_rate=0.01)
        # Gain weight should have increased meaningfully
        assert engine.weights["gain"] > Config().utility.weights["gain"]

    def test_zero_feature_no_update(self):
        """Features with value 0 shouldn't change their weight."""
        engine = UtilityEngine(Config(), mode="learned")
        original = engine.weights["redundancy_penalty"]
        engine.update_weights(
            features={"gain": 1.0, "confidence": 0.5, "cost_penalty": 0.3,
                       "redundancy_penalty": 0.0, "severity": 0.5},
            reward=1.0,
            predicted=0.5,
            learning_rate=0.1,
        )
        assert engine.weights["redundancy_penalty"] == original


# ======================================================================
# Cost penalties from config
# ======================================================================


class TestCostPenalties:
    def test_custom_action_costs(self):
        cfg = Config()
        cfg.utility.action_costs = {
            OrchestrationAction.RESPOND: 0.0,
            OrchestrationAction.REROLL: 0.9,  # very expensive
            OrchestrationAction.VERIFY: 0.1,
            OrchestrationAction.ESCALATE: 0.1,
            OrchestrationAction.RETRIEVE: 0.1,
            OrchestrationAction.STOP: 0.0,
        }
        engine = UtilityEngine(cfg)
        traj = _make_traj(monotonicity=False, confidence_score=0.5)
        disa = _make_disa(tier="medium", confidence_score=0.5)
        scores = engine.score_all(traj, disa, _default_context())
        score_map = {s.action: s.utility for s in scores}
        # VERIFY should now beat REROLL due to much lower cost
        assert score_map[OrchestrationAction.VERIFY] > score_map[OrchestrationAction.REROLL]

    def test_custom_weights(self):
        cfg = Config()
        cfg.utility.weights = {
            "gain": 1.0,           # heavily favor gain
            "confidence": 0.0,
            "cost_penalty": 0.0,
            "redundancy_penalty": 0.0,
            "severity": 0.0,
        }
        engine = UtilityEngine(cfg)
        # With only gain mattering, ideal conditions → RESPOND
        traj = _make_traj(monotonicity=True, confidence_score=0.9)
        disa = _make_disa(tier="low", confidence_score=0.9)
        scores = engine.score_all(traj, disa, _default_context())
        assert scores[0].action == OrchestrationAction.RESPOND


# ======================================================================
# Integration — full pipeline
# ======================================================================


class TestIntegration:
    def test_full_pipeline_respond(self):
        """End-to-end: ideal state → RESPOND."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=True, confidence_score=0.95)
        disa = _make_disa(tier="low", confidence_score=0.90)
        ctx = _default_context(step_num=0)
        scores = engine.score_all(traj, disa, ctx)
        best = engine.select_best(scores)
        assert best == OrchestrationAction.RESPOND

    def test_full_pipeline_reroll_with_hysteresis(self):
        """End-to-end with hysteresis: switch from RESPOND to REROLL
        only if margin is sufficient."""
        cfg = Config()
        cfg.utility.switch_threshold = 0.05
        engine = UtilityEngine(cfg)
        traj = _make_traj(monotonicity=False, confidence_score=0.2)
        disa = _make_disa(tier="weak", confidence_score=0.2)
        ctx = _default_context(step_num=1)

        scores = engine.score_all(traj, disa, ctx)
        best = engine.select_best(scores, previous_action=OrchestrationAction.RESPOND)

        # In this extreme case, REROLL should have enough margin to switch
        assert best == OrchestrationAction.REROLL

    def test_full_pipeline_verify(self):
        """End-to-end: medium disagreement → VERIFY."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.5)
        disa = _make_disa(tier="medium", confidence_score=0.5)
        scores = engine.score_all(traj, disa, _default_context())
        best = engine.select_best(scores)
        assert best == OrchestrationAction.VERIFY

    def test_exhausted_rerolls_shift_to_verify(self):
        """When reroll budget is exhausted, system should shift preference."""
        engine = UtilityEngine(Config())
        traj = _make_traj(monotonicity=False, confidence_score=0.3)
        disa = _make_disa(tier="weak", confidence_score=0.3)

        ctx_fresh = _default_context(n_rerolls=0)
        ctx_exhausted = _default_context(n_rerolls=3, max_rerolls=3)

        best_fresh = engine.select_best(engine.score_all(traj, disa, ctx_fresh))
        best_exhausted = engine.select_best(engine.score_all(traj, disa, ctx_exhausted))

        assert best_fresh == OrchestrationAction.REROLL
        # After exhausting rerolls, should prefer something else
        assert best_exhausted != OrchestrationAction.REROLL
