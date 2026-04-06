"""SNTR Self-Healing Loop — orchestrates detect → diagnose → repair → verify → continue."""

from __future__ import annotations
import time
import structlog
from typing import Optional

from .data_structures import (
    FailureEvent, FailureType, RecoveryStrategy,
    TrajectoryPatch, ProofResult, SNTRResult, FailureTrajectory
)
from .diagnostic import DiagnosticEngine
from .isabelle_engine import IsabelleProofEngine
from .heuristic_recovery import FASHAHeuristicRecovery
from .hindsight_library import HindsightExperienceLibrary
from .patcher import TrajectoryPatcher

logger = structlog.get_logger()


class SNTRSelfHealingLoop:
    """
    Main SNTR orchestrator.
    
    Combines:
    - FASHA diagnostic engine (fast pre-filter)
    - Isabelle/HOL neuro-symbolic proof repair (Stepwise)
    - HeRL hindsight experience replay (warm-start)
    - FASHA heuristic recovery (fallback)
    """

    def __init__(
        self,
        isabelle_timeout: int = 30,
        repair_threshold: float = 0.5,
        experience_cache_size: int = 1000,
    ):
        self.diagnostic = DiagnosticEngine()
        self.isabelle = IsabelleProofEngine(timeout=isabelle_timeout)
        self.heuristic = FASHAHeuristicRecovery()
        self.experience_lib = HindsightExperienceLibrary(capacity=experience_cache_size)
        self.patcher = TrajectoryPatcher()
        self.repair_threshold = repair_threshold

    def self_heal(self, trajectory: FailureTrajectory) -> SNTRResult:
        """
        Run the full SNTR self-healing loop on a failed trajectory.
        
        Steps:
        1. DETECT: Check each step for low confidence (TVC-style)
        2. DIAGNOSE: FASHA diagnostic engine classifies failure type
        3. REPAIR (if REASONING_ERROR): Isabelle proof search
        4. VERIFY: Confirm proof found, apply patch
        5. FALLBACK: If repair fails, use FASHA heuristic recovery
        """
        logger.info("sntr.loop.start", task=trajectory.task, num_steps=len(trajectory.steps))

        for step in trajectory.steps:
            result = self._heal_step(step, trajectory)
            if result.status == "REPAIR_SUCCEEDED":
                return result

        return SNTRResult(status="NO_FAILURE")

    def _heal_step(self, step: dict, trajectory: FailureTrajectory) -> SNTRResult:
        """Heal a single failed reasoning step."""
        step_content = step.get("content", "")
        confidence = step.get("confidence", 1.0)
        step_id = step.get("id", "unknown")

        # Step 1: Detect — TVC-style confidence tracking
        if confidence >= self.repair_threshold:
            return SNTRResult(status="NO_FAILURE")

        logger.info("sntr.detect.low_confidence", step_id=step_id, confidence=confidence)

        # Step 2: Diagnose — FASHA diagnostic engine
        failure = self.diagnostic.diagnose(
            step_content=step_content,
            confidence=confidence,
            context={"step_id": step_id, "trajectory": trajectory},
        )

        # Step 3: Route based on failure type
        if failure.failure_type == FailureType.REASONING_ERROR:
            return self._repair_reasoning_error(failure, step)
        else:
            return self._fallback_heuristic(failure, step)

    def _repair_reasoning_error(self, failure: FailureEvent, step: dict) -> SNTRResult:
        """
        Repair a REASONING_ERROR using Isabelle/HOL proof search.
        
        Steps:
        1. Try to warm-start from hindsight experience library
        2. Translate claim to HOL theorem
        3. Invoke Isabelle proof search
        4. If proof found → apply trajectory patch
        5. If proof fails → fallback to FASHA heuristic recovery
        """
        logger.info("sntr.repair.start", step_id=failure.step_id)

        # Try to warm-start from experience library
        similar = self.experience_lib.retrieve(failure.step_content, k=3)
        if similar:
            self.isabelle.warm_start(similar)
            logger.info("sntr.repair.warm_start", num_experiences=len(similar))

        # Run Isabelle proof search
        proof_result = self.isabelle.prove(
            claim=failure.step_content,
            context=self._extract_context(step),
        )

        if proof_result.proof_found:
            # Apply patch
            patch = self.patcher.create_patch(
                original_step=failure.step_content,
                failed_reason=failure.failed_reason,
                hol_theorem=proof_result.hol_term,
                proof_trace=proof_result.proof_steps,
                corrected_reasoning=self._extract_corrected_derivation(proof_result),
                confidence_before=failure.confidence_before,
                repair_time_ms=proof_result.search_time_ms,
            )

            # Store in hindsight experience library
            if patch.experience_id:
                self.experience_lib.add(patch)

            logger.info(
                "sntr.repair.success",
                step_id=failure.step_id,
                repair_time_ms=patch.repair_time_ms,
            )

            return SNTRResult(
                status="REPAIR_SUCCEEDED",
                original_step=failure.step_content,
                corrected_step=patch.corrected_step,
                hol_proof=proof_result,
                patch=patch,
                experience_hits=len(similar),
                failure_type=failure.failure_type,
            )
        else:
            # Fallback to FASHA heuristic recovery
            logger.warning(
                "sntr.repair.failed",
                step_id=failure.step_id,
                error=proof_result.error,
            )
            return self._fallback_heuristic(failure, step)

    def _fallback_heuristic(self, failure: FailureEvent, step: dict) -> SNTRResult:
        """Use FASHA heuristic recovery when Isabelle repair fails or isn't applicable."""
        strategy = self.heuristic.select_strategy(failure)
        result = self.heuristic.execute(strategy, failure)

        logger.info(
            "sntr.fallback",
            step_id=failure.step_id,
            failure_type=failure.failure_type.value,
            strategy=strategy.value,
        )

        return SNTRResult(
            status="REPAIR_FAILED",
            original_step=failure.step_content,
            failure_type=failure.failure_type,
            fallback_strategy=strategy,
        )

    def _extract_context(self, step: dict) -> list[str]:
        """Extract relevant context from step for Isabelle."""
        # Stub: would extract premises, antecedents, etc.
        return []

    def _extract_corrected_derivation(self, proof_result: ProofResult) -> str:
        """Extract natural language corrected derivation from Isabelle proof."""
        # Stub: would translate Isabelle proof steps back to natural language
        return "Derivation corrected via Isabelle/HOL proof search."
