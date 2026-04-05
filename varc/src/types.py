"""Core data types for VARC: enums and dataclasses for milestones and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(Enum):
    """Lifecycle state of a milestone as it moves through the VARC pipeline."""

    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    RECTIFIED = "rectified"
    PRUNED = "pruned"


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting (or contradicting) a milestone's claim."""

    source: str
    claim: str
    confidence: float = 0.0  # 0.0–1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class LayerResult:
    """Result emitted by a single verification layer (Box Maze, Themis, etc.)."""

    layer_name: str
    passed: bool
    reason: str
    confidence: float = 0.0  # 0.0–1.0


@dataclass
class Milestone:
    """A verifiable reasoning step in an agent trajectory."""

    id: str
    step_index: int
    content: str
    evidence_chain: list[EvidenceItem] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    layer_results: dict[str, LayerResult] = field(default_factory=dict)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
