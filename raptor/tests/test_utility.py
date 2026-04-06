"""Tests for utility.py — Step 4."""

import pytest

from raptor.utility import _StubUtilityEngine
from raptor.config import Config, OrchestrationAction
from raptor.entropy_tracker import TrajectorySignal
from raptor.disagreement import DisagreementSignal


class TestUtilityEngineStub:
    """Tests for the stub utility engine (scaffold validation)."""

    def test_stub_scores_all_returns_list(self):
        config = Config()
        engine = _StubUtilityEngine(config)
        traj = TrajectorySignal(
            n_steps=3,
            monotonicity=True,
            entropy_slope=-0.5,
            final_entropy=1.0,
            entropies=[3.0, 2.0, 1.0],
            confidence_score=0.8,
        )
        disa = DisagreementSignal(
            evidence_overlap=0.7,
            argument_strength=0.6,
            divergence_depth=2,
            dispersion=0.3,
            cohesion=0.6,
            confidence_score=0.7,
            disagreement_tier="low",
        )
        scores = engine.score_all(traj, disa)
        assert isinstance(scores, list)
        assert len(scores) >= 1

    def test_stub_select_best_returns_action(self):
        config = Config()
        engine = _StubUtilityEngine(config)
        from raptor.utility import ActionScore

        scores = [
            ActionScore(action=OrchestrationAction.REROLL, utility=0.8, breakdown={}),
            ActionScore(action=OrchestrationAction.RESPOND, utility=0.7, breakdown={}),
        ]
        best = engine.select_best(scores)
        assert isinstance(best, OrchestrationAction)


class TestOrchestrationActionEnum:
    """Tests for the OrchestrationAction enum."""

    def test_all_actions_defined(self):
        expected = {"respond", "reroll", "verify", "escalate", "retrieve", "stop"}
        actual = {a.value for a in OrchestrationAction}
        assert expected == actual

    def test_action_values_are_lowercase(self):
        for action in OrchestrationAction:
            assert action.value == action.value.lower()
