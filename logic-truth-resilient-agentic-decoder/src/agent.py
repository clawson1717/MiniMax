"""
L-TRAD Core Agent Orchestrator

The main entry point for the Logic-Driven Truth-Resilient Agentic Decoder.
Unifies all components into a single 'Reasoning Cascade' pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple
from src.models import CausalTask, ReasoningDraft
from src.orchestrator import TaskArchitect
from src.explorer import DraftExplorer
from src.verifier import StateVerifier
from src.sensing import AmbiguitySenser
from src.decoder import TruthDecoder
from src.healing import ProcessHealer
import asyncio


class LTRADAgent:
    """
    Unified reasoning agent that identifies goals, explores drafts, 
    and heals logic before producing a final answer.
    """
    
    def __init__(self, use_mock: bool = True):
        self.architect = TaskArchitect(use_mock=use_mock)
        self.explorer = DraftExplorer(use_mock=use_mock)
        self.verifier = StateVerifier()
        self.senser = AmbiguitySenser(use_mock=use_mock)
        self.decoder = TruthDecoder(self.senser)
        self.healer = ProcessHealer(self.explorer)
        
    async def solve(self, goal: str, initial_state: Dict[str, Any] = {}) -> Tuple[ReasoningDraft, str]:
        """
        Executes the full L-TRAD reasoning pipeline.
        Returns the winning draft and a process summary.
        """
        # 1. ARCHITECT: Decompose Goal
        task = await self.architect.decompose_goal(goal, initial_state)
        
        # 2. EXPLORER: Generate Drafts
        drafts = await self.explorer.generate_drafts(task, count=3)
        
        # 3. VERIFIER & SENSER: High-level check
        # For simplicity, we just check the first draft for noise
        noise_score = await self.senser.sense_ambiguity(task, drafts[0])
        is_valid, errors = self.verifier.verify_draft(task, drafts[0])
        
        # 4. HEALER: Trigger correction if necessary
        if noise_score > 0.5 or not is_valid:
            print(f"[Core] Draft noise detected ({noise_score:.2f}). Triggering Cascade Healing...")
            healed_draft = await self.healer.heal_draft(task, drafts[0], errors)
            drafts.append(healed_draft)
            
        # 5. DECODER: Select best path
        winner, tr_score = await self.decoder.decode_winner(task, drafts)
        
        status = "DENOISED" if winner.is_denoised else "DIRECT"
        summary = f"Reasoning {status}. Truth-Resilience Score: {tr_score:.2f}."
        
        return winner, summary

async def main():
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How to bake a cake with zero gravity constraints?"
    agent = LTRADAgent(use_mock=True)
    winner, summary = await agent.solve(query)
    print(f"\nResulting Winner Draft ID: {winner.draft_id}")
    print(f"Summary: {summary}")
    print(f"Thought Tokens: {winner.thought_tokens}")
    print(f"Proposed Plan: {winner.proposed_plan}")

if __name__ == "__main__":
    asyncio.run(main())
