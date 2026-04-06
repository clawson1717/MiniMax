"""Tests for orchestrator.py — Step 5."""

import pytest

from raptor.orchestrator import RAPTOROrchestrator
from raptor.config import Config
from raptor.disagreement import AgentResponse


class TestOrchestratorInit:
    """Tests for RAPTOROrchestrator initialization."""

    def test_orchestrator_initializes_with_config(self):
        config = Config()
        orch = RAPTOROrchestrator(config)
        assert orch.config is config

    def test_orchestrator_state_starts_at_step_zero(self):
        config = Config()
        orch = RAPTOROrchestrator(config)
        assert orch._state.step_num == 0
        assert orch._state.n_rerolls == 0
        assert orch._state.has_retrieved is False

    def test_reset_clears_state(self):
        config = Config()
        orch = RAPTOROrchestrator(config)
        orch._state = orch._state.__class__(
            step_num=5,
            n_rerolls=2,
            has_retrieved=True,
            current_action=orch._state.current_action,
            signal_history=[],
        )
        orch.reset()
        assert orch._state.step_num == 0
        assert orch._state.n_rerolls == 0
        assert orch._state.has_retrieved is False


class TestOrchestratorStepNotImplemented:
    """Verify step() raises NotImplementedError (Step 5 not done)."""

    def test_step_raises_not_implemented(self):
        config = Config()
        orch = RAPTOROrchestrator(config)
        with pytest.raises(NotImplementedError):
            orch.step(
                prompt="Test prompt",
                agent_responses=[
                    AgentResponse("a1", ["step1"], "answer")
                ],
            )
