"""
L-TRAD Reasoning Visualization CLI

Provides a colored terminal interface for observing the 
'Draft -> Denoised' reasoning progression in real-time.
"""

from typing import List, Dict, Any, Optional
from src.models import ReasoningDraft


class ReasoningTerminal:
    """Displays the L-TRAD reasoning cascade in a human-readable CLI format."""
    
    # Simple ANSI colors
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"
    
    def display_cascade(self, goal: str, winner: ReasoningDraft, logs: str):
        """Prints a detailed breakdown of the reasoning outcome."""
        print(f"\n{self.BOLD}L-TRAD REASONING CASCADE{self.END}")
        print("=" * 40)
        print(f"{self.BOLD}Goal:{self.END} {goal}")
        print(f"{self.BOLD}Status:{self.END} {logs}")
        
        print(f"\n{self.BOLD}Winning Path [{winner.draft_id[:8]}]:{self.END}")
        
        # Determine Color based on denoised status
        color = self.GREEN if winner.is_denoised else self.BLUE
        marker = "✨ [DENOISED]" if winner.is_denoised else "🚀 [DIRECT]"
        
        print(f"  {color}{marker}{self.END}")
        print(f"  {self.BOLD}Tokens:{self.END} {', '.join(winner.thought_tokens)}")
        print(f"  {self.BOLD}Confidence:{self.END} {winner.confidence_score:.2f}")
        
        print(f"\n{self.BOLD}Execution Plan:{self.END}")
        for i, step_id in enumerate(winner.proposed_plan):
            print(f"  {i+1}. {step_id}")
            
        print("=" * 40 + "\n")

    def display_healing(self, error: str):
        """Displays a specific healing event alert."""
        print(f"  {self.RED}⚠️ CAUSAL VIOLATION DETECTED:{self.END} {error}")
        print(f"  {self.YELLOW}⚙️ TRIGGERING HEALER...{self.END}")
