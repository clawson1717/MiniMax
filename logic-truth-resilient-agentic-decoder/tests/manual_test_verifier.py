import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.models import CausalTask, TaskStep, StateConstraint, ReasoningDraft
from src.verifier import StateVerifier

async def run_tests():
    verifier = StateVerifier()
    
    # Setup a mock task: Get Key -> Open Door
    step1 = TaskStep(description="Get Key", post_effects={"has_key": True})
    step2 = TaskStep(
        description="Open Door",
        pre_constraints=[StateConstraint(key="has_key", expected_value=True)],
        post_effects={"door_open": True}
    )
    
    task = CausalTask(goal="Open Door", steps=[step1, step2], current_state={"has_key": False})
    
    print("Testing Valid Draft...")
    valid_draft = ReasoningDraft(
        task_id=task.task_id, 
        thought_tokens=["find", "unlock"], 
        proposed_plan=[step1.step_id, step2.step_id]
    )
    is_valid, logs = verifier.verify_draft(task, valid_draft)
    assert is_valid is True
    print(f"✓ {logs[0]}")
    
    print("Testing Causally Invalid Draft (Out of Order)...")
    invalid_draft = ReasoningDraft(
        task_id=task.task_id, 
        thought_tokens=["rush", "door"], 
        proposed_plan=[step2.step_id]
    )
    is_valid, logs = verifier.verify_draft(task, invalid_draft)
    assert is_valid is False
    print(f"✓ Correctly identified failure: {logs[0]}")

if __name__ == "__main__":
    asyncio.run(run_tests())
