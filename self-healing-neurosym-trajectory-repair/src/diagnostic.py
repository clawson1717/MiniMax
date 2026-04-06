"""FASHA-style diagnostic engine for SNTR — fast pre-filter for failure classification."""

from __future__ import annotations
from typing import Optional
import structlog

from .data_structures import FailureEvent, FailureType, RecoveryStrategy

logger = structlog.get_logger()


class DiagnosticEngine:
    """
    FASHA-inspired diagnostic engine.
    
    Classifies failures into types to route to appropriate recovery.
    Only REASONING_ERROR triggers neuro-symbolic Isabelle repair;
    other types use FASHA heuristic recovery directly.
    """

    def __init__(self):
        self._failure_counts = {ft: 0 for ft in FailureType}

    def diagnose(self, step_content: str, confidence: float, context: dict) -> FailureEvent:
        """
        Classify a reasoning step failure into a FailureType.
        
        Uses heuristics based on:
        - Confidence score (low = reasoning error or knowledge gap)
        - Step content patterns (question marks = knowledge gap,
          logical connectors = reasoning error)
        - Context (missing required info = knowledge gap,
          conflicting info = reasoning error)
        """
        failure_type = self._classify(step_content, confidence, context)
        self._failure_counts[failure_type] += 1

        event = FailureEvent(
            step_id=context.get("step_id", "unknown"),
            step_content=step_content,
            failure_type=failure_type,
            confidence_before=confidence,
            failed_reason=self._explain(failure_type, step_content),
            context=context,
        )
        
        logger.info(
            "sntr.diagnostic",
            step_id=event.step_id,
            failure_type=failure_type.value,
            confidence=confidence,
        )
        
        return event

    def _classify(self, step_content: str, confidence: float, context: dict) -> FailureType:
        # High confidence + contains question = knowledge gap
        if confidence > 0.5 and "?" in step_content:
            return FailureType.KNOWLEDGE_GAP

        # Contains logical keywords but wrong conclusion = reasoning error
        logical_keywords = ["therefore", "thus", "implies", "consequently", "hence", "so"]
        has_logic = any(kw in step_content.lower() for kw in logical_keywords)
        has_contradiction = any(phrase in step_content.lower() for phrase in [
            "but", "however", "contradicts", "false"
        ])
        if has_logic and has_contradiction:
            return FailureType.REASONING_ERROR

        # Very low confidence = likely reasoning error
        if confidence < 0.3:
            return FailureType.REASONING_ERROR

        # Context missing required info
        if context.get("missing_info"):
            return FailureType.KNOWLEDGE_GAP

        # Strategy mismatch: task type doesn't fit approach
        if context.get("strategy_failed"):
            return FailureType.STRATEGY_MISMATCH

        # External failure indicators
        if context.get("tool_error") or context.get("network_error"):
            return FailureType.EXTERNAL_FAILURE

        # Default to reasoning error ( Isabelle can try to repair)
        return FailureType.REASONING_ERROR

    def _explain(self, failure_type: FailureType, step_content: str) -> str:
        explanations = {
            FailureType.KNOWLEDGE_GAP: f"Missing knowledge required to complete reasoning: {step_content[:100]}",
            FailureType.REASONING_ERROR: f"Logical error in reasoning step: {step_content[:100]}",
            FailureType.CONTEXT_LOSS: f"Context lost or forgotten: {step_content[:100]}",
            FailureType.STRATEGY_MISMATCH: f"Wrong approach for task type: {step_content[:100]}",
            FailureType.EXTERNAL_FAILURE: f"External tool/environment failure: {step_content[:100]}",
            FailureType.UNKNOWN: f"Unknown failure: {step_content[:100]}",
        }
        return explanations.get(failure_type, explanations[FailureType.UNKNOWN])

    def get_failure_stats(self) -> dict:
        return dict(self._failure_counts)
