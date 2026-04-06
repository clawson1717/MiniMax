"""
L-TRAD Triple-Agent Orchestrator: Part 4 - Denoising Senser

Responsible for measuring the semantic ambiguity and causal validity of
a ReasoningDraft to determine its denoising priority.
"""

from typing import List, Dict, Any, Optional
from src.models import CausalTask, ReasoningDraft, TaskStep
import asyncio


class AmbiguitySenser:
    """
    The 'Senser' identifies where a draft might be hallucinating or
    drifting from the original intent by measuring uncertainty over tokens.
    """
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        
    async def sense_ambiguity(self, task: CausalTask, draft: ReasoningDraft) -> float:
        """
        Calculates a 'Noise Score' for a given draft.
        Lower is better (0.0 = zero noise).
        """
        if self.use_mock:
            return await self._mock_sensing(task, draft)
            
        # TODO: Implement actual embedding-based sensing in Step 9
        raise NotImplementedError("LLM sensing not yet implemented.")
        
    async def _mock_sensing(self, task: CausalTask, draft: ReasoningDraft) -> float:
        """Simulates uncertainty sensing based on token-set complexity."""
        noise = 0.0
        
        # 1. Check for 'risky' tokens in reasoning
        risky_tokens = ["maybe", "should", "hope", "rush", "ignore"]
        for token in draft.thought_tokens:
            if token.lower() in risky_tokens:
                noise += 0.2
        
        # 2. Check for plan length mismatch relative to goal complexity
        # If the task has 3 steps and the draft has only 1, increase noise.
        if len(draft.proposed_plan) < len(task.steps):
            noise += 0.15 * (len(task.steps) - len(draft.proposed_plan))
            
        # 3. Base confidence inverse
        noise += (1.0 - draft.confidence_score) * 0.5
        
        return min(noise, 1.0)
