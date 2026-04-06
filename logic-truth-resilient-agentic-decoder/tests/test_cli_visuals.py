import asyncio
from src.models import ReasoningDraft
from src.display import ReasoningTerminal

def test_visuals():
    terminal = ReasoningTerminal()
    
    # 1. Simulate Direct Draft
    direct_draft = ReasoningDraft(
        task_id="task-1",
        thought_tokens=["execute", "quick"],
        proposed_plan=["step-a", "step-b"],
        confidence_score=0.88,
        is_denoised=False
    )
    
    terminal.display_cascade(
        "Open door",
        direct_draft,
        "Reasoning DIRECT. Truth-Resilience Score: 1.25."
    )
    
    # 2. Simulate Healing Event
    terminal.display_healing("Pre-constraint 'is_locked' mismatch at Step 2.")
    
    # 3. Simulate Denoised Winner
    denoised_draft = ReasoningDraft(
        task_id="task-1",
        thought_tokens=["verify", "logic", "slow"],
        proposed_plan=["step-a", "step-c", "step-b"],
        confidence_score=0.95,
        is_denoised=True
    )
    
    terminal.display_cascade(
        "Open door",
        denoised_draft,
        "Reasoning DENOISED. Truth-Resilience Score: 1.45."
    )

if __name__ == "__main__":
    test_visuals()
