import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.models import CausalTask, TaskStep
from src.explorer import DraftExplorer

async def run_tests():
    explorer = DraftExplorer(use_mock=True)
    
    # Setup a mock task with 2 steps
    sample_task = CausalTask(
        goal="Open the vault",
        steps=[
            TaskStep(description="Verify key"),
            TaskStep(description="Turn handle")
        ]
    )
    
    print(f"Generating drafts for task: {sample_task.goal}...")
    drafts = await explorer.generate_drafts(sample_task, count=3)
    
    assert len(drafts) == 3
    print(f"✓ Generated {len(drafts)} drafts.")
    
    # Check draft diversity
    assert drafts[0].thought_tokens == ["execute", "immediate", "linear"]
    assert drafts[1].confidence_score > drafts[2].confidence_score
    print("✓ Draft diversity verified.")
    
    # Check plan alignment
    assert len(drafts[0].proposed_plan) == 2
    print("✓ Plan alignment verified.")

if __name__ == "__main__":
    asyncio.run(run_tests())
