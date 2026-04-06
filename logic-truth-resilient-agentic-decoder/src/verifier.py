"""
L-TRAD Triple-Agent Orchestrator: Part 3 - Set Designer (Verifier)

Responsible for validating a 'Reasoning Draft' against the deterministic
state-constraints of a CausalTask.
"""

from typing import List, Dict, Any, Optional, Tuple
from src.models import CausalTask, ReasoningDraft, TaskStep


class StateVerifier:
    """
    The 'Set Designer' / 'Verifier' provides the ground-truth check
    for draft solutions, pinpointing exactly where logic breaks down.
    """
    
    def verify_draft(self, task: CausalTask, draft: ReasoningDraft) -> Tuple[bool, List[str]]:
        """
        Validates the proposed plan in a draft step-by-step.
        Returns (is_valid, error_log).
        """
        temp_state = task.current_state.copy()
        errors = []
        
        # We need to map step_ids to the actual objects in the task
        step_map = {step.step_id: step for step in task.steps}
        
        for i, step_id in enumerate(draft.proposed_plan):
            step = step_map.get(step_id)
            if not step:
                errors.append(f"Step {i}: Invalid step_id {step_id}")
                return False, errors
                
            # Perform transition validation
            # We simulate the state effects if valid
            if not self._check_step(step, temp_state):
                errors.append(f"Step {i} ({step.description}): Causal constraint violation.")
                return False, errors
                
            # Apply post_effects to the temp_state for the next iteration
            temp_state.update(step.post_effects)
            
        # Optional: Check if the goal state was reached (simplified for now)
        return True, ["Draft validated successfully."]

    def _check_step(self, step: TaskStep, state: Dict[str, Any]) -> bool:
        """Internal logic check for a single state transition."""
        for constraint in step.pre_constraints:
            current_val = state.get(constraint.key)
            if constraint.comparator == "==":
                if current_val != constraint.expected_value: return False
            elif constraint.comparator == "!=":
                if current_val == constraint.expected_value: return False
        return True
