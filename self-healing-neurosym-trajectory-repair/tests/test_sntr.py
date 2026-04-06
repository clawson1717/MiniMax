"""Tests for SNTR core components."""

import pytest
from sntr.data_structures import (
    FailureEvent, FailureType, RecoveryStrategy,
    TrajectoryPatch, ProofResult, SNTRResult, FailureTrajectory
)
from sntr.diagnostic import DiagnosticEngine
from sntr.heuristic_recovery import FASHAHeuristicRecovery
from sntr.patcher import TrajectoryPatcher
from sntr.hindsight_library import HindsightExperienceLibrary


class TestDiagnosticEngine:
    """Tests for FASHA-style diagnostic engine."""

    def test_classifies_reasoning_error_low_confidence(self):
        engine = DiagnosticEngine()
        event = engine.diagnose(
            step_content="Therefore Socrates is eternal because all humans are mortal",
            confidence=0.2,
            context={},
        )
        assert event.failure_type == FailureType.REASONING_ERROR

    def test_classifies_knowledge_gap_question_high_confidence(self):
        engine = DiagnosticEngine()
        event = engine.diagnose(
            step_content="What is the capital of France?",
            confidence=0.6,
            context={},
        )
        assert event.failure_type == FailureType.KNOWLEDGE_GAP

    def test_failure_stats_tracked(self):
        engine = DiagnosticEngine()
        engine.diagnose("step 1", 0.2, {})
        engine.diagnose("step 2", 0.8, {"tool_error": True})
        stats = engine.get_failure_counts()
        assert stats[FailureType.REASONING_ERROR] == 1
        assert stats[FailureType.EXTERNAL_FAILURE] == 1


class TestFASHAHeuristicRecovery:
    """Tests for FASHA heuristic recovery fallback."""

    def test_selects_ask_clarification_for_knowledge_gap(self):
        recovery = FASHAHeuristicRecovery()
        failure = FailureEvent(
            step_id="1",
            step_content="What is X?",
            failure_type=FailureType.KNOWLEDGE_GAP,
            confidence_before=0.5,
            failed_reason="Missing info",
        )
        strategy = recovery.select_strategy(failure)
        assert strategy == RecoveryStrategy.ASK_CLARIFICATION

    def test_selects_backtrack_for_unrepaired_reasoning_error(self):
        recovery = FASHAHeuristicRecovery()
        failure = FailureEvent(
            step_id="1",
            step_content="Therefore X",
            failure_type=FailureType.REASONING_ERROR,
            confidence_before=0.3,
            failed_reason="Logic error",
        )
        strategy = recovery.select_strategy(failure)
        assert strategy == RecoveryStrategy.BACKTRACK


class TestTrajectoryPatcher:
    """Tests for trajectory patch creation and application."""

    def test_create_patch(self):
        patcher = TrajectoryPatcher()
        patch = patcher.create_patch(
            original_step="All A are B. All B are C. Therefore A are eternal.",
            failed_reason="Logic error",
            hol_theorem="∀x. A x ⟶ C x",
            proof_trace=["assume H1: ∀x. A x ⟶ B x", "assume H2: ∀x. B x ⟶ C x", "show ∀x. A x ⟶ C x"],
            corrected_reasoning="All A are B. All B are C. Therefore A are C.",
            confidence_before=0.2,
            repair_time_ms=150,
        )
        assert patch.original_step != patch.corrected_step
        assert patch.confidence_after > patch.confidence_before
        assert patch.repair_time_ms == 150

    def test_apply_patch_non_destructive(self):
        patcher = TrajectoryPatcher()
        patch = TrajectoryPatcher().create_patch(
            original_step="Step 1",
            failed_reason="error",
            hol_theorem="theorem",
            proof_trace=["proof"],
            corrected_reasoning="Corrected Step 1",
            confidence_before=0.2,
            repair_time_ms=100,
        )
        trajectory = [
            {"id": "0", "content": "Step 0"},
            {"id": "1", "content": "Step 1"},
            {"id": "2", "content": "Step 2"},
        ]
        patched = patcher.apply_patch(trajectory, patch, step_index=1)
        assert patched[1]["content"] == "Corrected Step 1"
        assert patched[1]["patch_metadata"]["repaired"] is True


class TestHindsightExperienceLibrary:
    """Tests for HeRL-inspired experience library."""

    def test_add_and_retrieve(self):
        library = HindsightExperienceLibrary(capacity=10)
        patch = TrajectoryPatch(
            original_step="All A are B",
            failed_reason="Logic error",
            hol_theorem="∀x. A x ⟶ B x",
            proof_trace=["step1", "step2"],
            corrected_step="All A are C",
            confidence_before=0.2,
            confidence_after=0.9,
            repair_time_ms=100,
        )
        exp_id = library.add(patch)
        assert exp_id is not None

    def test_stats(self):
        library = HindsightExperienceLibrary(capacity=10)
        stats = library.stats()
        assert stats["total"] == 0
        assert "avg_success_rate" in stats


class TestFailureTrajectory:
    """Tests for FailureTrajectory data model."""

    def test_to_dict(self):
        trajectory = FailureTrajectory(
            task="Solve math problem",
            steps=[{"id": "1", "content": "Step 1"}],
            failures=[],
        )
        d = trajectory.to_dict()
        assert d["task"] == "Solve math problem"
        assert len(d["steps"]) == 1
