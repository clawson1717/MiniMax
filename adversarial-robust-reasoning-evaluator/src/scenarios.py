"""
AD-REASON Adversarial Thought-Cycle Models

Defines the schemas for the adversarial scenarios used to stress-test
reasoning agents.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class HintType(str, Enum):
    """The nature of the hint provided to the agent-under-test."""
    HELPFUL = "helpful"
    MISLEADING = "misleading"
    CORRUPTED = "corrupted"  # State-corruption


class AdversarialCycle(BaseModel):
    """
    A single 'Thought-Cycle' designed to test the resilience of the
    reasoning process itself.
    """
    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    base_logic: str      # The core causal problem to solve
    hint: str            # The hint/outside input to inject
    hint_type: HintType
    expected_drift: float = 0.0  # Measured semantic shift if misleading
    ground_truth: str    # The deterministic correct final conclusion
    metadata: Dict[str, Any] = {}
