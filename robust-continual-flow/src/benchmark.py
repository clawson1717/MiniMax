"""
Integrated Benchmark Runner

Executes the RCF agent against the adversarial environment and measures
accuracy vs. token efficiency.
"""

import json
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

from src.environment import AdversarialEnvironment, create_sample_scenarios
from src.agent import NavigationAgent


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    scenario_id: str
    scenario_name: str
    scenario_type: str
    success: bool
    final_score: float
    turns_taken: int
    total_tokens: int
    tokens_per_turn: float
    time_seconds: float
    alerts_triggered: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""
    timestamp: str
    total_scenarios: int
    successful_scenarios: int
    success_rate: float
    avg_score: float
    total_tokens: int
    avg_tokens_per_scenario: float
    avg_tokens_per_turn: float
    total_time_seconds: float
    avg_time_per_scenario: float
    results: List[BenchmarkResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_scenarios": self.total_scenarios,
                "successful_scenarios": self.successful_scenarios,
                "success_rate": self.success_rate,
                "avg_score": self.avg_score,
                "total_tokens": self.total_tokens,
                "avg_tokens_per_scenario": self.avg_tokens_per_scenario,
                "avg_tokens_per_turn": self.avg_tokens_per_turn,
                "total_time_seconds": self.total_time_seconds,
                "avg_time_per_scenario": self.avg_time_per_scenario,
            },
            "results": [asdict(r) for r in self.results]
        }


class MockModel:
    """
    A mock model for benchmarking without actual LLM calls.
    Simulates varying quality responses based on scenario difficulty.
    """
    
    def __init__(self, quality: float = 0.7, seed: Optional[int] = None):
        """
        Initialize mock model.
        
        Args:
            quality: Base quality level (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.quality = quality
        self.token_count = 0
        self.call_count = 0
        
        if seed is not None:
            import random
            random.seed(seed)
    
    def __call__(self, prompt: str, compute_budget: int) -> List[str]:
        """
        Generate mock votes based on prompt and compute budget.
        
        Args:
            prompt: Input prompt
            compute_budget: Number of votes to generate
            
        Returns:
            List of vote strings
        """
        self.call_count += 1
        
        # Estimate tokens (rough approximation)
        self.token_count += len(prompt.split()) * 1.3  # ~1.3 tokens per word
        self.token_count += compute_budget * 5  # Each vote adds tokens
        
        # Generate votes with varying quality
        votes = []
        import random
        
        # Base action set
        actions = [
            "look for information",
            "click button",
            "read details",
            "submit form",
            "navigate forward",
            "decline offer",
            "confirm action",
            "search for item"
        ]
        
        for _ in range(compute_budget):
            if random.random() < self.quality:
                # Higher quality vote - more specific
                votes.append(random.choice(actions[:4]))
            else:
                # Lower quality vote - more random
                votes.append(random.choice(actions))
        
        return votes
    
    def get_token_count(self) -> int:
        return int(self.token_count)
    
    def reset_token_count(self) -> None:
        self.token_count = 0
        self.call_count = 0


class BenchmarkRunner:
    """
    Runs integrated benchmarks of the RCF agent against adversarial scenarios.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def run_single_scenario(
        self,
        agent: NavigationAgent,
        env: AdversarialEnvironment,
        scenario_id: str,
    ) -> BenchmarkResult:
        """
        Run a single scenario benchmark.
        
        Args:
            agent: The navigation agent
            env: The adversarial environment
            scenario_id: ID of scenario to run
            
        Returns:
            BenchmarkResult for the scenario
        """
        # Reset token counter if using MockModel
        if isinstance(agent.model, MockModel):
            agent.model.reset_token_count()
        
        start_time = time.time()
        alerts_count = 0
        
        try:
            # Reset environment
            observation = env.reset(scenario_id)
            scenario = env.current_scenario
            
            # Run agent
            result = agent.run(goal=scenario.goal, max_steps=10)
            
            # Count alerts from monitor
            if hasattr(agent, 'monitor'):
                alerts_count = len(agent.monitor.alerts_history) if hasattr(agent.monitor, 'alerts_history') else 0
            
            end_time = time.time()
            
            # Get final score
            final_score = 0.0
            success = False
            if hasattr(env, '_evaluate_episode'):
                final_score = env._evaluate_episode()
                threshold = scenario.success_criteria.get('threshold', 0.7)
                success = final_score >= threshold
            
            # Get token count
            total_tokens = 0
            if isinstance(agent.model, MockModel):
                total_tokens = agent.model.get_token_count()
            else:
                # Estimate from trajectory
                total_tokens = sum(
                    len(str(node.get('data', '')).split()) * 1.3
                    for node in result.get('trajectory', {}).get('nodes', [])
                )
            
            turns_taken = env.current_turn
            
        except Exception as e:
            if self.verbose:
                print(f"Error running scenario {scenario_id}: {e}")
            end_time = time.time()
            final_score = 0.0
            success = False
            total_tokens = 0
            turns_taken = 0
            scenario = env.current_scenario
        
        result = BenchmarkResult(
            scenario_id=scenario_id,
            scenario_name=scenario.name if scenario else "Unknown",
            scenario_type=scenario.scenario_type.value if scenario else "unknown",
            success=success,
            final_score=final_score,
            turns_taken=turns_taken,
            total_tokens=total_tokens,
            tokens_per_turn=total_tokens / max(turns_taken, 1),
            time_seconds=end_time - start_time,
            alerts_triggered=alerts_count,
        )
        
        return result
    
    def run_benchmark(
        self,
        scenarios: Optional[List[str]] = None,
        model_quality: float = 0.7,
        seed: Optional[int] = None,
    ) -> BenchmarkReport:
        """
        Run full benchmark suite.
        
        Args:
            scenarios: List of specific scenario IDs to run, or None for all
            model_quality: Quality level for mock model (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            BenchmarkReport with aggregated results
        """
        self.results = []
        
        # Create environment and load scenarios
        env = AdversarialEnvironment(seed=seed)
        
        # Add sample scenarios
        for scenario in create_sample_scenarios():
            env.add_scenario(scenario)
        
        # Create mock model and agent
        model = MockModel(quality=model_quality, seed=seed)
        agent = NavigationAgent(
            env=env,
            model=model,
            base_compute=3,
            uncertainty_threshold=0.5
        )
        
        # Determine which scenarios to run
        available = env.get_available_scenarios()
        scenario_ids = scenarios if scenarios else [s['id'] for s in available]
        
        if self.verbose:
            print(f"Running benchmark with {len(scenario_ids)} scenarios...")
        
        # Run each scenario
        for scenario_id in scenario_ids:
            if self.verbose:
                print(f"  Running {scenario_id}...")
            
            result = self.run_single_scenario(agent, env, scenario_id)
            self.results.append(result)
        
        # Generate report
        report = self._generate_report()
        
        # Save if output directory specified
        if self.output_dir:
            self._save_report(report)
        
        return report
    
    def _generate_report(self) -> BenchmarkReport:
        """Generate aggregated report from results."""
        if not self.results:
            return BenchmarkReport(
                timestamp=datetime.now().isoformat(),
                total_scenarios=0,
                successful_scenarios=0,
                success_rate=0.0,
                avg_score=0.0,
                total_tokens=0,
                avg_tokens_per_scenario=0.0,
                avg_tokens_per_turn=0.0,
                total_time_seconds=0.0,
                avg_time_per_scenario=0.0,
                results=[],
            )
        
        successful = sum(1 for r in self.results if r.success)
        total_tokens = sum(r.total_tokens for r in self.results)
        total_time = sum(r.time_seconds for r in self.results)
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_scenarios=len(self.results),
            successful_scenarios=successful,
            success_rate=successful / len(self.results),
            avg_score=sum(r.final_score for r in self.results) / len(self.results),
            total_tokens=total_tokens,
            avg_tokens_per_scenario=total_tokens / len(self.results),
            avg_tokens_per_turn=sum(r.tokens_per_turn for r in self.results) / len(self.results),
            total_time_seconds=total_time,
            avg_time_per_scenario=total_time / len(self.results),
            results=self.results,
        )
    
    def _save_report(self, report: BenchmarkReport) -> None:
        """Save report to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"benchmark_{report.timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        if self.verbose:
            print(f"Report saved to {filepath}")


def run_quick_benchmark(
    model_quality: float = 0.7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quick benchmark function for testing.
    
    Args:
        model_quality: Quality level for mock model
        verbose: Print progress
        
    Returns:
        Dictionary with benchmark summary
    """
    runner = BenchmarkRunner(verbose=verbose)
    report = runner.run_benchmark(model_quality=model_quality)
    
    if verbose:
        print("\n=== Benchmark Summary ===")
        print(f"Success Rate: {report.success_rate:.1%}")
        print(f"Avg Score: {report.avg_score:.2f}")
        print(f"Total Tokens: {report.total_tokens}")
        print(f"Avg Tokens/Turn: {report.avg_tokens_per_turn:.1f}")
        print(f"Total Time: {report.total_time_seconds:.2f}s")
    
    return report.to_dict()


if __name__ == "__main__":
    run_quick_benchmark()
