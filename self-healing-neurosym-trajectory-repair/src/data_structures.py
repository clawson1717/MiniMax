"""Core data structures for SNTR."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FailureType(Enum):
    """FASHA-style failure type classification."""
    KNOWLEDGE_GAP = "KNOWLEDGE_GAP"
    REASONING_ERROR = "REASONING_ERROR"
    CONTEXT_LOSS = "CONTEXT_LOSS"
    STRATEGY_MISMATCH = "STRATEGY_MISMATCH"
    EXTERNAL_FAILURE = "EXTERNAL_FAILURE"
    UNKNOWN = "UNKNOWN"


class RecoveryStrategy(Enum):
    """FASHA-inspired recovery strategies."""
    REREAD = "REREAD"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    BACKTRACK = "BACKTRACK"
    RETRY_DIFFERENT_APPROACH = "RETRY_DIFFERENT_APPROACH"
    SIMPLIFY = "SIMPLIFY"
    ESCALATE = "ESCALATE"
    APPLY_PATCH = "APPLY_PATCH"  # SNTR-specific: apply Isabelle proof patch


@dataclass
class FailureEvent:
    """Record of a single failure episode."""
    step_id: str
    step_content: str
    failure_type: FailureType
    confidence_before: float
    failed_reason: str
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_content": self.step_content,
            "failure_type": self.failure_type.value,
            "confidence_before": self.confidence_before,
            "failed_reason": self.failed_reason,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrajectoryPatch:
    """A formally verified correction to a reasoning step."""
    original_step: str
    failed_reason: str
    hol_theorem: str
    proof_trace: list[str]
    corrected_step: str
    confidence_before: float
    confidence_after: float
    repair_time_ms: int
    experience_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "original_step": self.original_step,
            "failed_reason": self.failed_reason,
            "hol_theorem": self.hol_theorem,
            "proof_trace": self.proof_trace,
            "corrected_step": self.corrected_step,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "repair_time_ms": self.repair_time_ms,
            "experience_id": self.experience_id,
        }


@dataclass
class ProofResult:
    """Result of an Isabelle/HOL proof search."""
    proof_found: bool
    hol_term: str
    derivation: list[str]
    proof_steps: list[str]
    search_time_ms: int
    timeout: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "proof_found": self.proof_found,
            "hol_term": self.hol_term,
            "derivation": self.derivation,
            "proof_steps": self.proof_steps,
            "search_time_ms": self.search_time_ms,
            "timeout": self.timeout,
            "error": self.error,
        }


@dataclass
class HindsightExperience:
    """A repaired trajectory stored for future retrieval (HeRL-inspired)."""
    id: str
    failed_claim: str
    hol_term: str
    proof: list[str]
    corrected_reasoning: str
    success_count: int = 0
    failure_count: int = 0
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def retrieval_score(self) -> float:
        """Success rate weighted by total usage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # neutral for unused experiences
        return self.success_count / total


@dataclass
class SNTRResult:
    """Result of running SNTR self-healing loop."""
    status: str  # REPAIR_SUCCEEDED / REPAIR_FAILED / ESCALATED / NO_FAILURE
    original_step: Optional[str] = None
    corrected_step: Optional[str] = None
    hol_proof: Optional[ProofResult] = None
    patch: Optional[TrajectoryPatch] = None
    experience_hits: int = 0
    fallback_strategy: Optional[RecoveryStrategy] = None
    failure_type: Optional[FailureType] = None


@dataclass
class FailureTrajectory:
    """A reasoning trajectory with potential failures."""
    task: str
    steps: list[dict]
    failures: list[FailureEvent] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> FailureTrajectory:
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(
            task=data.get("task", ""),
            steps=data.get("steps", []),
            failures=[FailureEvent(**f) if isinstance(f, dict) else f for f in data.get("failures", [])],
        )

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "steps": self.steps,
            "failures": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.failures],
        }
