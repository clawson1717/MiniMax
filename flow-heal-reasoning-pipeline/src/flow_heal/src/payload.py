"""Payload module for ReasoningStep objects in the FLOW-HEAL pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict, astuple
from typing import Optional, Dict, Any


@dataclass
class ReasoningStep:
    """Represents a single reasoning step in a multi-agent pipeline.
    
    This class captures the essential information about a reasoning step,
    including the agent that produced it, its content, uncertainty measure,
    and its relationship to previous steps in the reasoning chain.
    
    Attributes:
        agent_id: Unique identifier for the agent that generated this step.
        intent_hash: Hash of the agent's intent or purpose for this step.
        content: The actual content or output from the reasoning step.
        uncertainty_score: A float between 0.0 and 1.0 representing uncertainty.
        causal_parent: Optional reference to the parent step's ID (if any).
    """
    
    agent_id: str
    intent_hash: str
    content: str
    uncertainty_score: float
    causal_parent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ReasoningStep to a dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningStep:
        """Create a ReasoningStep instance from a dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        """Return a string representation of the reasoning step."""
        return (
            f"ReasoningStep(agent={self.agent_id}, "
            f"intent_hash={self.intent_hash[:8]}..., "
            f"uncertainty={self.uncertainty_score:.2f}, "
            f"parent={self.causal_parent})"
        )