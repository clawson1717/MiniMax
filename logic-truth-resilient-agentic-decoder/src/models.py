"""
L-TRAD Core Models

Defines the schemas for logic-grounded tasks and reasoning drafts,
including state-transition validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import uuid


class StateConstraint(BaseModel):
    """A requirement that must be met for a state transition to be valid."""
    key: str
    expected_value: Any
    comparator: str = "=="  # ==, !=, >, <, in, not in


class TaskStep(BaseModel):
    """A single deterministic step in a causal task."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    pre_constraints: List[StateConstraint] = []
    post_effects: Dict[str, Any] = {}


class CausalTask(BaseModel):
    """
    A unified task structure with deterministic state verification logic.
    Inspired by the LOGIGEN Architect/Designer/Explorer framework.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    steps: List[TaskStep]
    current_state: Dict[str, Any] = {}
    
    def validate_transition(self, step_index: int, proposed_state: Dict[str, Any]) -> bool:
        """Checks if a proposed state transition follows the step's constraints."""
        if step_index < 0 or step_index >= len(self.steps):
            return False
            
        step = self.steps[step_index]
        
        # 1. Check pre-constraints against current_state
        for constraint in step.pre_constraints:
            current_val = self.current_state.get(constraint.key)
            if constraint.comparator == "==":
                if current_val != constraint.expected_value: return False
            elif constraint.comparator == "!=":
                if current_val == constraint.expected_value: return False
        
        # 2. Check if proposed_state matches expected post_effects
        for key, expected_val in step.post_effects.items():
            if proposed_state.get(key) != expected_val:
                return False
                
        return True


class ReasoningDraft(BaseModel):
    """
    A concise 'Draft-Thinking' structure for exploring reasoning paths.
    """
    draft_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    thought_tokens: List[str]  # Concise reasoning snippets
    proposed_plan: List[str]   # The decoded sequence of step_ids
    confidence_score: float = 0.0
    is_denoised: bool = False
