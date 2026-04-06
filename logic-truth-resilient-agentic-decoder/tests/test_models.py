import pytest
from src.models import CausalTask, TaskStep, StateConstraint, ReasoningDraft

def test_causal_task_validation():
    # Setup a simple task: Open door -> Enter room
    step1 = TaskStep(
        description="Open the door",
        pre_constraints=[StateConstraint(key="door_locked", expected_value=False)],
        post_effects={"door_open": True}
    )
    
    task = CausalTask(
        goal="Enter the room",
        steps=[step1],
        current_state={"door_locked": False, "door_open": False}
    )
    
    # Valid transition
    assert task.validate_transition(0, {"door_open": True}) is True
    
    # Invalid: Pre-constraint failed
    task.current_state["door_locked"] = True
    assert task.validate_transition(0, {"door_open": True}) is False
    
    # Invalid: Post-effect mismatch
    task.current_state["door_locked"] = False
    assert task.validate_transition(0, {"door_open": "maybe"}) is False

def test_reasoning_draft_init():
    draft = ReasoningDraft(
        task_id="test-task",
        thought_tokens=["door", "unlocked", "open"],
        proposed_plan=["step-1"]
    )
    assert draft.is_denoised is False
    assert len(draft.thought_tokens) == 3
