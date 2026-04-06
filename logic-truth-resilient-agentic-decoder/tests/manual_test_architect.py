import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.orchestrator import TaskArchitect

async def run_tests():
    architect = TaskArchitect(use_mock=True)
    
    print("Testing Access Decomposition...")
    task = await architect.decompose_goal("Access the secure file", {"door_locked": True})
    assert len(task.steps) == 3
    assert "Verify" in task.steps[0].description
    print("✓ Passed")
    
    print("Testing Generic Fallback...")
    task = await architect.decompose_goal("Bake a cake", {})
    assert len(task.steps) == 1
    assert "Bake a cake" in task.steps[0].description
    print("✓ Passed")

if __name__ == "__main__":
    asyncio.run(run_tests())
