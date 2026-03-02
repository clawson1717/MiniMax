"""
Token Efficiency Analysis

Measures token usage vs. accuracy to demonstrate efficiency gains from
capacity-based allocation. Provides detailed efficiency metrics and reports.
"""

import random
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .evaluation import (
    RoutingStrategy,
    RoutingMetrics,
    StrategyResult,
    ComparativeEvaluator,
    MockModel,
    create_evaluation_environment,
)
from .benchmarks import (
    BenchmarkTask,
    TaskDataset,
    BenchmarkType,
    Difficulty,
    MultiAgentBenchmarkRunner,
)
from .environment import MultiAgentEnvironment, AgentRole, MessageType
from .agent import ReasoningAgent, AgentConfig


@dataclass
class EfficiencyMetrics:
    """Metrics for token efficiency analysis."""
    tokens_per_correct_answer: float = 0.0
    accuracy_per_1000_tokens: float = 0.0
    total_tokens: int = 0
    correct_answers: int = 0
    total_tasks: int = 0
    efficiency_ratio: float = 0.0  # vs baseline (uniform)
    cost_savings_percent: float = 0.0
    quality_efficiency_score: float = 0.0


@dataclass
class StrategyEfficiencyResult:
    """Efficiency results for a single routing strategy."""
    strategy: RoutingStrategy
    metrics: EfficiencyMetrics
    token_breakdown: Dict[str, int] = field(default_factory=dict)
    accuracy_breakdown: Dict[str, float] = field(default_factory=dict)
    efficiency_curve: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class EfficiencyReport:
    """Complete efficiency report across all strategies."""
    baseline_strategy: RoutingStrategy
    results_by_strategy: Dict[str, StrategyEfficiencyResult]
    efficiency_ranking: List[Tuple[str, float]]
    savings_summary: str
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]


class TokenEfficiencyAnalyzer:
    """
    Analyzes token efficiency of different routing strategies.
    
    Measures the tradeoff between token usage (compute cost) and accuracy,
    demonstrating the efficiency gains from capacity-based allocation.
    """
    
    def __init__(
        self,
        evaluator: ComparativeEvaluator,
        baseline_strategy: RoutingStrategy = RoutingStrategy.UNIFORM,
    ):
        """
        Initialize the efficiency analyzer.
        
        Args:
            evaluator: ComparativeEvaluator with results
            baseline_strategy: Strategy to use as baseline (default: uniform)
        """
        self.evaluator = evaluator
        self.baseline_strategy = baseline_strategy
        self.results: Dict[str, StrategyEfficiencyResult] = {}
    
    def compute_efficiency_metrics(
        self,
        strategy_result: StrategyResult,
        baseline_result: Optional[StrategyResult] = None,
    ) -> EfficiencyMetrics:
        """
        Compute efficiency metrics for a strategy result.
        
        Args:
            strategy_result: Results from evaluating a strategy
            baseline_result: Baseline results for comparison
            
        Returns:
            EfficiencyMetrics with computed values
        """
        total_tokens = strategy_result.routing_metrics.total_tokens_used
        correct = strategy_result.benchmark_results.correct
        total_tasks = strategy_result.benchmark_results.total_tasks
        
        # Tokens per correct answer
        tokens_per_correct = total_tokens / correct if correct > 0 else float('inf')
        
        # Accuracy per 1000 tokens
        accuracy = strategy_result.benchmark_results.accuracy
        acc_per_1k_tokens = (accuracy * 1000) / (total_tokens / 1000) if total_tokens > 0 else 0
        
        # Quality-efficiency score (combines accuracy and token efficiency)
        quality_eff = accuracy * (1 - min(total_tokens / 10000, 0.9))
        
        # Efficiency ratio vs baseline
        efficiency_ratio = 1.0
        cost_savings = 0.0
        
        if baseline_result is not None:
            baseline_tokens = baseline_result.routing_metrics.total_tokens_used
            baseline_correct = baseline_result.benchmark_results.correct
            baseline_accuracy = baseline_result.benchmark_results.accuracy
            
            # Efficiency ratio: (accuracy / tokens) relative to baseline
            if baseline_tokens > 0 and total_tokens > 0:
                strategy_eff = accuracy / (total_tokens / 1000)
                baseline_eff = baseline_accuracy / (baseline_tokens / 1000)
                efficiency_ratio = strategy_eff / baseline_eff if baseline_eff > 0 else 1.0
            
            # Cost savings: if same accuracy achieved with fewer tokens
            if accuracy >= baseline_accuracy * 0.95:  # Within 5% accuracy
                cost_savings = ((baseline_tokens - total_tokens) / baseline_tokens) * 100
        
        return EfficiencyMetrics(
            tokens_per_correct_answer=tokens_per_correct,
            accuracy_per_1000_tokens=acc_per_1k_tokens,
            total_tokens=total_tokens,
            correct_answers=correct,
            total_tasks=total_tasks,
            efficiency_ratio=efficiency_ratio,
            cost_savings_percent=max(0, cost_savings),
            quality_efficiency_score=quality_eff,
        )
    
    def analyze_strategy(
        self,
        strategy: RoutingStrategy,
        strategy_result: StrategyResult,
        baseline_result: Optional[StrategyResult] = None,
    ) -> StrategyEfficiencyResult:
        """
        Analyze efficiency for a single strategy.
        
        Args:
            strategy: The routing strategy
            strategy_result: Results from evaluation
            baseline_result: Baseline results for comparison
            
        Returns:
            StrategyEfficiencyResult with detailed metrics
        """
        metrics = self.compute_efficiency_metrics(strategy_result, baseline_result)
        
        # Token breakdown by task type
        token_breakdown = defaultdict(int)
        accuracy_breakdown = defaultdict(list)
        
        for task_metric in strategy_result.per_task_metrics:
            task_type = task_metric.get("task_type", "unknown")
            tokens = task_metric.get("tokens", 0)
            score = task_metric.get("score", 0)
            
            token_breakdown[task_type] += tokens
            accuracy_breakdown[task_type].append(score)
        
        # Compute average accuracy per type
        accuracy_avg = {}
        for task_type, scores in accuracy_breakdown.items():
            accuracy_avg[task_type] = sum(scores) / len(scores) if scores else 0
        
        # Build efficiency curve (cumulative accuracy vs cumulative tokens)
        efficiency_curve = []
        cumulative_tokens = 0
        cumulative_correct = 0
        
        sorted_metrics = sorted(
            strategy_result.per_task_metrics,
            key=lambda x: x.get("tokens", 0)
        )
        
        for i, tm in enumerate(sorted_metrics):
            cumulative_tokens += tm.get("tokens", 0)
            cumulative_correct += 1 if tm.get("is_correct", False) else 0
            cumulative_acc = cumulative_correct / (i + 1) if i >= 0 else 0
            efficiency_curve.append((cumulative_tokens, cumulative_acc))
        
        return StrategyEfficiencyResult(
            strategy=strategy,
            metrics=metrics,
            token_breakdown=dict(token_breakdown),
            accuracy_breakdown=accuracy_avg,
            efficiency_curve=efficiency_curve,
        )
    
    def run_analysis(
        self,
        results: Optional[Dict[str, StrategyResult]] = None,
    ) -> EfficiencyReport:
        """
        Run full efficiency analysis across all strategies.
        
        Args:
            results: Pre-computed results (uses evaluator results if None)
            
        Returns:
            EfficiencyReport with complete analysis
        """
        if results is None:
            results = self.evaluator.results
        
        if not results:
            raise ValueError("No results available for analysis")
        
        # Get baseline result
        baseline_key = self.baseline_strategy.value
        baseline_result = results.get(baseline_key)
        
        # Analyze each strategy
        efficiency_results = {}
        for strategy_name, strategy_result in results.items():
            strategy = RoutingStrategy(strategy_name)
            efficiency_results[strategy_name] = self.analyze_strategy(
                strategy=strategy,
                strategy_result=strategy_result,
                baseline_result=baseline_result if strategy_name != baseline_key else None,
            )
        
        self.results = efficiency_results
        
        # Create ranking by quality-efficiency score
        ranking = sorted(
            [(name, res.metrics.quality_efficiency_score) for name, res in efficiency_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate summary
        summary = self._generate_summary(efficiency_results)
        recommendations = self._generate_recommendations(efficiency_results)
        detailed = self._compute_detailed_metrics(efficiency_results)
        
        return EfficiencyReport(
            baseline_strategy=self.baseline_strategy,
            results_by_strategy=efficiency_results,
            efficiency_ranking=ranking,
            savings_summary=summary,
            recommendations=recommendations,
            detailed_metrics=detailed,
        )
    
    def _generate_summary(
        self,
        results: Dict[str, StrategyEfficiencyResult],
    ) -> str:
        """Generate a text summary of efficiency gains."""
        lines = ["Token Efficiency Summary", "=" * 40, ""]
        
        for name, result in results.items():
            m = result.metrics
            lines.append(f"{name.upper()}:")
            lines.append(f"  Tokens per correct answer: {m.tokens_per_correct_answer:.1f}")
            lines.append(f"  Accuracy per 1K tokens: {m.accuracy_per_1000_tokens:.3f}")
            lines.append(f"  Efficiency ratio: {m.efficiency_ratio:.2f}x")
            lines.append(f"  Cost savings: {m.cost_savings_percent:.1f}%")
            lines.append("")
        
        # Find most efficient
        best = max(results.items(), key=lambda x: x[1].metrics.quality_efficiency_score)
        lines.append(f"Most efficient: {best[0]} (score: {best[1].metrics.quality_efficiency_score:.3f})")
        
        return "\n".join(lines)
    
    def _generate_recommendations(
        self,
        results: Dict[str, StrategyEfficiencyResult],
    ) -> List[str]:
        """Generate efficiency recommendations."""
        recommendations = []
        
        info_efficient = results.get(RoutingStrategy.INFO_EFFICIENT.value)
        uniform = results.get(RoutingStrategy.UNIFORM.value)
        uncertainty = results.get(RoutingStrategy.UNCERTAINTY_BASED.value)
        
        if info_efficient and uniform:
            ratio = info_efficient.metrics.efficiency_ratio
            savings = info_efficient.metrics.cost_savings_percent
            
            if ratio > 1.0:
                recommendations.append(
                    f"Info-efficient routing is {ratio:.1f}x more efficient than uniform routing. "
                    f"Use capacity-weighted allocation for cost-sensitive applications."
                )
            
            if savings > 10:
                recommendations.append(
                    f"Info-efficient routing saves {savings:.0f}% tokens while maintaining accuracy. "
                    f"Consider this for production deployments with high query volume."
                )
        
        if info_efficient and uncertainty:
            if info_efficient.metrics.quality_efficiency_score > uncertainty.metrics.quality_efficiency_score:
                recommendations.append(
                    "Capacity-based routing outperforms uncertainty-based routing in efficiency. "
                    "Prefer information capacity as the primary routing signal."
                )
        
        # General recommendations
        recommendations.append(
            "Monitor token efficiency over time as agent capacities evolve. "
            "Re-evaluate routing strategies periodically."
        )
        
        return recommendations
    
    def _compute_detailed_metrics(
        self,
        results: Dict[str, StrategyEfficiencyResult],
    ) -> Dict[str, Any]:
        """Compute detailed metrics for analysis."""
        detailed = {
            "by_strategy": {},
            "comparative": {},
            "efficiency_gains": {},
        }
        
        for name, result in results.items():
            detailed["by_strategy"][name] = {
                "total_tokens": result.metrics.total_tokens,
                "correct_answers": result.metrics.correct_answers,
                "tokens_per_correct": result.metrics.tokens_per_correct_answer,
                "accuracy_per_1k": result.metrics.accuracy_per_1000_tokens,
                "quality_eff_score": result.metrics.quality_efficiency_score,
                "token_breakdown": result.token_breakdown,
                "accuracy_breakdown": result.accuracy_breakdown,
            }
        
        # Comparative metrics
        strategies = list(results.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                m1 = results[s1].metrics
                m2 = results[s2].metrics
                
                key = f"{s1}_vs_{s2}"
                detailed["comparative"][key] = {
                    "token_diff": m1.total_tokens - m2.total_tokens,
                    "accuracy_diff": m1.accuracy_per_1000_tokens - m2.accuracy_per_1000_tokens,
                    "efficiency_ratio": m1.efficiency_ratio / m2.efficiency_ratio if m2.efficiency_ratio > 0 else 1.0,
                }
        
        # Efficiency gains (info-efficient vs others)
        info = results.get(RoutingStrategy.INFO_EFFICIENT.value)
        if info:
            for s in strategies:
                if s != RoutingStrategy.INFO_EFFICIENT.value:
                    other = results[s]
                    gain = (other.metrics.tokens_per_correct_answer - info.metrics.tokens_per_correct_answer)
                    detailed["efficiency_gains"][f"info_efficient_vs_{s}"] = {
                        "tokens_saved_per_correct": gain,
                        "percent_improvement": (gain / other.metrics.tokens_per_correct_answer * 100) if other.metrics.tokens_per_correct_answer > 0 else 0,
                    }
        
        return detailed


def run_efficiency_analysis(
    num_agents: int = 3,
    num_tasks: int = 30,
    model_quality: float = 0.8,
    seed: int = 42,
) -> EfficiencyReport:
    """
    Run a complete efficiency analysis for testing.
    
    Args:
        num_agents: Number of agents
        num_tasks: Number of tasks
        model_quality: Mock model quality
        seed: Random seed
        
    Returns:
        EfficiencyReport
    """
    from .evaluation import ComparativeEvaluator, create_evaluation_environment
    
    # Create environment
    env, agents = create_evaluation_environment(num_agents, seed)
    
    # Create evaluator
    evaluator = ComparativeEvaluator(env, agents, seed)
    
    # Get tasks
    runner = MultiAgentBenchmarkRunner()
    tasks = runner.get_all_tasks().sample(num_tasks, seed=seed)
    
    # Run comparison
    evaluator.run_comparison(tasks=tasks, model_quality=model_quality)
    
    # Analyze efficiency
    analyzer = TokenEfficiencyAnalyzer(evaluator)
    return analyzer.run_analysis()


if __name__ == "__main__":
    print("Running token efficiency analysis...")
    
    report = run_efficiency_analysis(
        num_agents=3,
        num_tasks=30,
        model_quality=0.8,
        seed=42,
    )
    
    print("\n" + report.savings_summary)
    print("\nEfficiency Ranking:")
    for name, score in report.efficiency_ranking:
        print(f"  {name}: {score:.3f}")
    
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
