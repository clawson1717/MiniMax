import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.models import CausalTask, TaskStep, ReasoningDraft
from src.sensing import AmbiguitySenser
from src.decoder import TruthDecoder

async def run_tests():
    senser = AmbiguitySenser(use_mock=True)
    decoder = TruthDecoder(senser)
    
    # Setup mock task
    task = CausalTask(goal="Unlock vault", steps=[TaskStep(description="V"), TaskStep(description="T")])
    
    print("Testing Drafting Competition...")
    
    # Draft 1: High confidence, High noise (Risky)
    risky_draft = ReasoningDraft(
        task_id=task.task_id,
        thought_tokens=["rush", "ignore", "hope"],
        proposed_plan=["s1"],
        confidence_score=0.9
    )
    
    # Draft 2: Moderate confidence, Low noise (Resilient)
    resilient_draft = ReasoningDraft(
        task_id=task.task_id,
        thought_tokens=["verify", "linear"],
        proposed_plan=["s1", "s2"],
        confidence_score=0.7
    )
    
    winner, score = await decoder.decode_winner(task, [risky_draft, resilient_draft])
    
    # The resilient draft should win even with lower individual confidence
    # because the risky tokens (rush, ignore) drive its noise score to 1.0.
    assert winner.thought_tokens == ["verify", "linear"]
    print(f"✓ Resilient draft won as expected (TR: {score:.2f})")

if __name__ == "__main__":
    asyncio.run(run_tests())
