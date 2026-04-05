"""VARC — Verifiable Agent Runtime Controller. Core package exports."""

from __future__ import annotations

from .trajectory import TrajectoryGraph
from .types import EvidenceItem, LayerResult, Milestone, VerificationStatus

__all__ = [
    "EvidenceItem",
    "LayerResult",
    "Milestone",
    "TrajectoryGraph",
    "VerificationStatus",
]
