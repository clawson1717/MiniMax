import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.models import CausalTask, TaskStep, ReasoningDraft
from src.explorer import DraftExplorer
from src.healing import ProcessHealer

async def run_tests():
    explorer = DraftExplorer(use_mock=True)
    healer = ProcessHealer(explorer)
    
    # Setup mock task
    task = CausalTask(goal="Unlock vault", steps=[TaskStep(description="Step 1"), TaskStep(description="Step 2")])
    
    print("Testing Targeted Healing...")
    # Simulated failing draft
    noisy_draft = ReasoningDraft(
        task_id=task.task_id,
        thought_tokens=["rush", "ignore"],
        proposed_plan=["s1"],
        confidence_score=0.4
    )
    
    # Simulate errors found by verifier
    errors = ["Step 0: Causal constraint violation."]
    
    healed = await healer.heal_draft(task, noisy_draft, errors)
    
    assert healed.is_denoised is True
    assert healed.confidence_score > noisy_draft.confidence_score
    print(f"✓ Healing successful. (Healed Confidence: {healed.confidence_score:.2f})")

if __name__ == "__main__":
    asyncio.run(run_tests())
