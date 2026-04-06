import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.models import CausalTask, TaskStep, ReasoningDraft
from src.sensing import AmbiguitySenser

async def run_tests():
    senser = AmbiguitySenser(use_mock=True)
    
    # Setup mock task
    task = CausalTask(goal="Unlock vault", steps=[TaskStep(description="V"), TaskStep(description="T")])
    
    print("Testing Clean Draft...")
    clean_draft = ReasoningDraft(
        task_id=task.task_id,
        thought_tokens=["verify", "linear"],
        proposed_plan=["s1", "s2"],
        confidence_score=0.9
    )
    score = await senser.sense_ambiguity(task, clean_draft)
    assert score < 0.2
    print(f"✓ Clean score: {score:.2f}")
    
    print("Testing Noisy Draft (Risky Tokens + Short Plan)...")
    noisy_draft = ReasoningDraft(
        task_id=task.task_id,
        thought_tokens=["rush", "ignore", "hope"],
        proposed_plan=["s1"],
        confidence_score=0.4
    )
    score = await senser.sense_ambiguity(task, noisy_draft)
    assert score > 0.6
    print(f"✓ Noisy score: {score:.2f}")

if __name__ == "__main__":
    asyncio.run(run_tests())
