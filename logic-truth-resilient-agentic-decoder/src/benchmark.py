"""
L-TRAD Deterministic Benchmark Suite

Evaluates reasoning accuracy vs. token efficiency by subjecting the
agent to LOGIGEN-inspired tasks.
"""

from typing import List, Dict, Any, Optional
from src.agent import LTRADAgent
import time
import asyncio


class TruthResilienceSuite:
    """
    Evaluates how well the L-TRAD agent handles logic-heavy tasks
    compared to a naive draft-only approach.
    """
    
    def __init__(self, agent: LTRADAgent):
        self.agent = agent
        
    async def run_suite(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs a series of tasks and returns performance metrics."""
        results = {
            "total_tasks": len(tasks),
            "success_rate": 0.0,
            "avg_tr_score": 0.0,
            "denoising_frequency": 0.0,
            "total_time_ms": 0
        }
        
        successes = 0
        tr_total = 0.0
        denoised_count = 0
        
        start_time = time.time()
        
        for task_data in tasks:
            winner, summary = await self.agent.solve(task_data["goal"], task_data["initial_state"])
            
            # Simple success metric: Did the winner plan match the expected step count?
            # In a full LOGIGEN test, we would run the environment.
            if len(winner.proposed_plan) >= task_data["min_expected_steps"]:
                successes += 1
                
            if winner.is_denoised:
                denoised_count += 1
                
            # Extract TR score from summary string
            try:
                tr_score = float(summary.split("Score: ")[1].strip("."))
                tr_total += tr_score
            except (IndexError, ValueError):
                pass
                
        results["total_time_ms"] = int((time.time() - start_time) * 1000)
        results["success_rate"] = successes / len(tasks)
        results["avg_tr_score"] = tr_total / len(tasks)
        results["denoising_frequency"] = denoised_count / len(tasks)
        
        return results
