"""Core data structures for CGPM."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceLevel(Enum):
    """Confidence level categories for quick filtering."""
    HIGH = "high"       # >= 0.8
    MEDIUM = "medium"   # >= 0.5, < 0.8
    LOW = "low"         # < 0.5


@dataclass
class ConfidenceScore:
    """Represents the confidence of a knowledge object.
    
    Combines prior confidence, observed reliability, and Bayesian update potential.
    """
    value: float = 0.5
    n_observations: int = 0
    reliability: float = 0.5  # Source reliability (0-1)
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence value must be in [0, 1], got {self.value}")
    
    @property
    def level(self) -> ConfidenceLevel:
        if self.value >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.value >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    def update(self, new_evidence: float, weight: float = 1.0) -> None:
        """Bayesian-style update with new evidence.
        
        Args:
            new_evidence: New observation in [0, 1]
            weight: Weight of new evidence vs. prior
        """
        self.value = (self.value * weight + new_evidence) / (weight + 1.0)
        self.n_observations += 1


@dataclass
class ProvenanceRecord:
    """Records the origin and history of a knowledge object."""
    source: str                    # e.g., "llm_generation", "user_input", "tool_call"
    agent_id: str | None = None    # Which agent created this
    tool_name: str | None = None   # Which tool produced the fact
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "context": self.context,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceRecord:
        return cls(**d)


@dataclass
class RuleConstraint:
    """A rule that can constrain or modify a knowledge object's confidence."""
    rule_type: str        # "exclusion", "endorsement", "decay_modifier", "confidence_ceiling"
    predicate: str        # Human-readable description of the rule
    domain: str           # e.g., "medical", "legal", "technical"
    strength: float = 1.0  # How strongly this rule applies (0-1)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "predicate": self.predicate,
            "domain": self.domain,
            "strength": self.strength,
        }


@dataclass
class KnowledgeObject:
    """A persistent, auditable knowledge fact with uncertainty.
    
    This is the fundamental unit of CGPM — a fact that knows how confident
    it is, where it came from, and how it should decay over time.
    """
    content: str
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(0.5))
    provenance: ProvenanceRecord | dict[str, Any] | None = None
    rules: list[RuleConstraint] = field(default_factory=list)
    decay_rate: float = 0.001   # Fractional decay per hour
    tags: list[str] = field(default_factory=list)
    ko_id: str = ""             # SHA-256 hash of (content, provenance_hash)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    decay_history: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.provenance is None:
            self.provenance = ProvenanceRecord(source="unknown")
        if isinstance(self.provenance, dict):
            self.provenance = ProvenanceRecord.from_dict(self.provenance)
        
        # Handle confidence being either a ConfidenceScore object or a float
        if isinstance(self.confidence, (float, int)):
            self.confidence = ConfidenceScore(value=float(self.confidence))
        
        if not self.ko_id:
            self._compute_id()
        
        if not 0.0 <= self.confidence.value <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence.value}")
    
    def _compute_id(self) -> None:
        """Compute deterministic SHA-256 ID from content + provenance."""
        prov_hash = ""
        if self.provenance:
            prov_bytes = json.dumps(self.provenance.to_dict(), sort_keys=True).encode()
            prov_hash = hashlib.sha256(prov_bytes).hexdigest()[:16]
        
        # Convert content to JSON string for hashing, handling both str and dict
        if isinstance(self.content, dict):
            content_str = json.dumps(self.content, sort_keys=True)
        else:
            content_str = str(self.content)
        
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        self.ko_id = f"ko_{content_hash}{prov_hash}"
    
    def apply_decay(self, hours_elapsed: float) -> float:
        """Apply temporal decay to confidence.
        
        Formula: c(t) = c0 * exp(-decay_rate * t)
        
        Returns:
            New confidence value
        """
        import math
        old_value = self.confidence.value
        new_value = old_value * math.exp(-self.decay_rate * hours_elapsed)
        new_value = max(0.0, new_value)  # Floor at 0
        self.confidence = ConfidenceScore(
            value=new_value,
            n_observations=self.confidence.n_observations,
            reliability=self.confidence.reliability
        )
        self.updated_at = time.time()
        self.decay_history.append({
            "hours_elapsed": hours_elapsed,
            "old_confidence": self.confidence.to_dict(),
            "new_confidence": self.confidence.to_dict(),
            "timestamp": self.updated_at,
        })
        return self.confidence.value
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ko_id": self.ko_id,
            "content": self.content,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict() if isinstance(self.provenance, ProvenanceRecord) else self.provenance,
            "rules": [r.to_dict() for r in self.rules],
            "decay_rate": self.decay_rate,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "decay_history": self.decay_history,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeObject:
        if "provenance" in d and isinstance(d["provenance"], dict):
            d["provenance"] = ProvenanceRecord.from_dict(d["provenance"])
        if "rules" in d:
            d["rules"] = [RuleConstraint(**r) for r in d["rules"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, s: str) -> KnowledgeObject:
        return cls.from_dict(json.loads(s))
