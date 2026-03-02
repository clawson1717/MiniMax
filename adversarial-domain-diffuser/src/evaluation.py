import json
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.integrated_loop import IntegratedAdversarialLoop

class Evaluator:
    """
    Evaluator for the adversarial-domain-diffuser system.
    Measures accuracy improvement, sample efficiency, and convergence stability
    across domain-specific benchmarks.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads mock domain-specific benchmarks.
        Returns a dictionary where keys are domain names and values are lists of benchmark items.
        """
        benchmarks = {
            "LegalBench": [
                {
                    "query": "Interpret the liability clause in this contract regarding force majeure.",
                    "expert_reference": "Liability is limited except in cases of gross negligence, even under force majeure.",
                    "domain_configs": [
                        {"id": "legal_expert", "domain": "law"},
                        {"id": "contract_specialist", "domain": "contracts"}
                    ]
                }
            ],
            "MedicalQA": [
                {
                    "query": "What are the contraindications for administering drug X in a patient with hypertension?",
                    "expert_reference": "Drug X should be avoided in hypertensive patients due to risk of acute renal failure.",
                    "domain_configs": [
                        {"id": "cardiologist", "domain": "cardiology"},
                        {"id": "pharmacist", "domain": "pharmacy"}
                    ]
                }
            ]
        }
        return benchmarks

    def run_evaluation(
        self, 
        domain: str, 
        benchmark_items: List[Dict[str, Any]], 
        max_iterations: int = 5,
        target_gap_reduction: float = 0.5,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Runs the evaluation for a specific domain.
        
        Args:
            domain: Name of the domain (e.g., 'LegalBench').
            benchmark_items: List of queries and expert references.
            max_iterations: Max iterations per adversarial loop.
            target_gap_reduction: The reduction in gap score to measure sample efficiency.
            num_runs: Number of independent runs to measure stability.
        """
        self.logger.info(f"Starting evaluation for domain: {domain}")
        
        domain_results = []
        
        for item in benchmark_items:
            query = item["query"]
            expert_ref = item["expert_reference"]
            configs = item["domain_configs"]
            
            item_runs = []
            for run_id in range(num_runs):
                loop = IntegratedAdversarialLoop(
                    agent_configs=configs,
                    expert_reference=expert_ref,
                    max_iterations=max_iterations
                )
                
                result = loop.run_iteration(query)
                history = result["history"]
                
                # Calculate metrics for this run
                initial_gap = history[0]["gap_score"]
                final_gap = history[-1]["gap_score"]
                improvement = initial_gap - final_gap
                
                # Sample Efficiency: queries required to reach target reduction
                queries_to_target = None
                for idx, step in enumerate(history):
                    reduction = initial_gap - step["gap_score"]
                    if reduction >= target_gap_reduction:
                        queries_to_target = idx + 1
                        break
                
                item_runs.append({
                    "run_id": run_id,
                    "initial_gap": initial_gap,
                    "final_gap": final_gap,
                    "improvement": improvement,
                    "queries_to_target": queries_to_target,
                    "final_consensus": history[-1]["consensus_summary"]
                })
            
            # Aggregate metrics across runs for this item
            improvements = [r["improvement"] for r in item_runs]
            avg_improvement = float(np.mean(improvements))
            stability = float(np.var(improvements))
            
            sample_efficiencies = [r["queries_to_target"] for r in item_runs if r["queries_to_target"] is not None]
            avg_sample_efficiency = float(np.mean(sample_efficiencies)) if sample_efficiencies else None

            domain_results.append({
                "query": query,
                "avg_accuracy_improvement": avg_improvement,
                "convergence_stability": stability,
                "avg_sample_efficiency": avg_sample_efficiency,
                "runs": item_runs
            })

        report = {
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "results": domain_results
        }
        
        self.save_report(domain, report)
        return report

    def save_report(self, domain: str, report: Dict[str, Any]):
        filename = f"{domain}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        self.logger.info(f"Report saved to {filepath}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = Evaluator()
    benchmarks = evaluator.load_benchmarks()
    
    for domain, items in benchmarks.items():
        evaluator.run_evaluation(domain, items)
