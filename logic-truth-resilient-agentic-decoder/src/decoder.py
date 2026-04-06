"""
L-TRAD Triple-Agent Orchestrator: Part 5 - Truth Decoder (Regulator)

Responsible for aggregating multiple ReasoningDrafts and selecting the
one with the highest Truth-Resilience (high confidence, low noise).
"""

from typing import List, Dict, Any, Optional, Tuple
from src.models import ReasoningDraft
from src.sensing import AmbiguitySenser


class TruthDecoder:
    """
    The 'Decoder' / 'Regulator' uses the scores from the Senser and
    Verifier to pick the optimal path and flag noisy segments.
    """
    
    def __init__(self, senser: AmbiguitySenser):
        self.senser = senser
        
    async def decode_winner(self, task, drafts: List[ReasoningDraft]) -> Tuple[ReasoningDraft, float]:
        """
        Selects the draft with the highest Truth-Resilience Score.
        TR = (Confidence * 1.5) - (Noise * 2.0).
        """
        best_draft = None
        best_score = -float('inf')
        
        for draft in drafts:
            # 1. Sense noise
            noise = await self.senser.sense_ambiguity(task, draft)
            
            # 2. Calculate Truth-Resilience
            tr_score = (draft.confidence_score * 1.5) - (noise * 2.0)
            
            if tr_score > best_score:
                best_score = tr_score
                best_draft = draft
                
        if not best_draft:
            raise RuntimeError("No valid drafts found for decoding.")
            
        return best_draft, best_score
