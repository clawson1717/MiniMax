import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.agent import LTRADAgent

async def run_integration_test():
    agent = LTRADAgent(use_mock=True)
    
    print("Executing L-TRAD Integration Test...")
    goal = "Open the secure vault and synthesis the data"
    initial_state = {"power_on": True}
    
    winner, summary = await agent.solve(goal, initial_state)
    
    print("-" * 20)
    print(f"Goal: {goal}")
    print(f"Outcome Plan: {winner.proposed_plan}")
    print(f"Thought Tokens: {winner.thought_tokens}")
    print(f"Status Summary: {summary}")
    print("-" * 20)
    
    assert len(winner.proposed_plan) > 0
    assert "Score" in summary
    print("✓ Integration test successful.")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
