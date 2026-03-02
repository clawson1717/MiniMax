"""
Tests for Token Efficiency Analysis Module

Tests for measuring token usage vs accuracy efficiency gains.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import random

from src.efficiency import (
    EfficiencyMetrics,
    StrategyEfficiencyResult,
    EfficiencyReport,
    TokenEfficiencyAnalyzer,
    run_efficiency_analysis,
)
from src.evaluation import (
    RoutingStrategy,
    RoutingMetrics,
    StrategyResult,
    ComparativeEvaluator,
    create_evaluation_environment,
)
from src.benchmarks import (
    BenchmarkTask,
    TaskDataset,
    BenchmarkResults,
    EvaluationResult,
    BenchmarkType,
    Difficulty,
)


class TestEfficiencyMetrics:
    """Tests for EfficiencyMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        metrics = EfficiencyMetrics()
        
        assert metrics.tokens_per_correct_answer == 0.0
        assert metrics.accuracy_per_1000_tokens == 0.0
        assert metrics.total_tokens == 0
        assert metrics.correct_answers == 0
        assert metrics.efficiency_ratio == 0.0
    
    def test_custom_values(self):
        """Test custom values."""
        metrics = EfficiencyMetrics(
            tokens_per_correct_answer=150.5,
            accuracy_per_1000_tokens=0.75,
            total_tokens=1505,
            correct_answers=10,
            total_tasks=15,
            efficiency_ratio=1.25,
            cost_savings_percent=20.0,
            quality_efficiency_score=0.68,
        )
        
        assert metrics.tokens_per_correct_answer == 150.5
        assert metrics.accuracy_per_1000_tokens == 0.75
        assert metrics.total_tokens == 1505
        assert metrics.correct_answers == 10
        assert metrics.efficiency_ratio == 1.25
        assert metrics.cost_savings_percent == 20.0


class TestStrategyEfficiencyResult:
    """Tests for StrategyEfficiencyResult dataclass."""
    
    def test_creation(self):
        """Test creating a strategy efficiency result."""
        metrics = EfficiencyMetrics(
            tokens_per_correct_answer=100.0,
            total_tokens=1000,
            correct_answers=10,
        )
        
        result = StrategyEfficiencyResult(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            metrics=metrics,
            token_breakdown={"math": 500, "qa": 500},
            accuracy_breakdown={"math": 0.8, "qa": 0.7},
        )
        
        assert result.strategy == RoutingStrategy.INFO_EFFICIENT
        assert result.metrics.total_tokens == 1000
        assert result.token_breakdown["math"] == 500


class TestEfficiencyReport:
    """Tests for EfficiencyReport dataclass."""
    
    def test_creation(self):
        """Test creating an efficiency report."""
        report = EfficiencyReport(
            baseline_strategy=RoutingStrategy.UNIFORM,
            results_by_strategy={},
            efficiency_ranking=[("info_efficient", 0.75), ("uniform", 0.65)],
            savings_summary="Test summary",
            recommendations=["Use capacity-based routing"],
            detailed_metrics={},
        )
        
        assert report.baseline_strategy == RoutingStrategy.UNIFORM
        assert len(report.efficiency_ranking) == 2
        assert len(report.recommendations) == 1


class TestTokenEfficiencyAnalyzer:
    """Tests for TokenEfficiencyAnalyzer class."""
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator with results."""
        evaluator = Mock(spec=ComparativeEvaluator)
        
        # Create mock results
        results = {}
        for strategy in [RoutingStrategy.INFO_EFFICIENT, RoutingStrategy.UNIFORM, RoutingStrategy.UNCERTAINTY_BASED]:
            routing_metrics = RoutingMetrics(
                total_messages=30,
                total_tokens_used=1500 + random.randint(-200, 200),
            )
            
            benchmark_results = BenchmarkResults(
                benchmark_name=f"{strategy.value}_benchmark",
                total_tasks=30,
                correct=24,
                total_score=24.5,
                accuracy=0.8,
                avg_response_time_ms=100.0,
                total_tokens_used=routing_metrics.total_tokens_used,
                total_capacity_used=50.0,
            )
            
            # Create per-task metrics
            per_task = []
            for i in range(30):
                per_task.append({
                    "task_id": f"task_{i}",
                    "is_correct": random.random() < 0.8,
                    "score": random.uniform(0.5, 1.0),
                    "tokens": random.randint(40, 60),
                    "task_type": random.choice(["math", "qa", "reasoning"]),
                })
            
            results[strategy.value] = StrategyResult(
                strategy=strategy,
                benchmark_results=benchmark_results,
                routing_metrics=routing_metrics,
                per_task_metrics=per_task,
            )
        
        evaluator.results = results
        return evaluator
    
    @pytest.fixture
    def analyzer(self, mock_evaluator):
        """Create an analyzer with mock evaluator."""
        return TokenEfficiencyAnalyzer(mock_evaluator)
    
    def test_initialization(self, mock_evaluator):
        """Test analyzer initialization."""
        analyzer = TokenEfficiencyAnalyzer(mock_evaluator)
        
        assert analyzer.evaluator is mock_evaluator
        assert analyzer.baseline_strategy == RoutingStrategy.UNIFORM
    
    def test_compute_efficiency_metrics(self, analyzer):
        """Test computing efficiency metrics."""
        strategy_result = analyzer.evaluator.results[RoutingStrategy.INFO_EFFICIENT.value]
        baseline_result = analyzer.evaluator.results[RoutingStrategy.UNIFORM.value]
        
        metrics = analyzer.compute_efficiency_metrics(strategy_result, baseline_result)
        
        assert metrics.total_tokens > 0
        assert metrics.tokens_per_correct_answer > 0
        assert metrics.accuracy_per_1000_tokens >= 0
        assert metrics.efficiency_ratio > 0
    
    def test_compute_efficiency_metrics_no_baseline(self, analyzer):
        """Test computing metrics without baseline."""
        strategy_result = analyzer.evaluator.results[RoutingStrategy.INFO_EFFICIENT.value]
        
        metrics = analyzer.compute_efficiency_metrics(strategy_result)
        
        assert metrics.efficiency_ratio == 1.0  # No baseline comparison
        assert metrics.cost_savings_percent == 0.0
    
    def test_analyze_strategy(self, analyzer):
        """Test analyzing a single strategy."""
        strategy_result = analyzer.evaluator.results[RoutingStrategy.INFO_EFFICIENT.value]
        
        result = analyzer.analyze_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            strategy_result=strategy_result,
        )
        
        assert result.strategy == RoutingStrategy.INFO_EFFICIENT
        assert result.metrics.total_tokens > 0
        assert len(result.efficiency_curve) > 0
    
    def test_run_analysis(self, analyzer):
        """Test running full analysis."""
        report = analyzer.run_analysis()
        
        assert report is not None
        assert report.baseline_strategy == RoutingStrategy.UNIFORM
        assert len(report.results_by_strategy) == 3
        assert len(report.efficiency_ranking) == 3
        assert report.savings_summary is not None
        assert len(report.recommendations) > 0
    
    def test_efficiency_ranking(self, analyzer):
        """Test that efficiency ranking is correct."""
        report = analyzer.run_analysis()
        
        # Ranking should be sorted descending
        scores = [score for _, score in report.efficiency_ranking]
        assert scores == sorted(scores, reverse=True)
    
    def test_detailed_metrics_structure(self, analyzer):
        """Test detailed metrics structure."""
        report = analyzer.run_analysis()
        
        assert "by_strategy" in report.detailed_metrics
        assert "comparative" in report.detailed_metrics
        assert "efficiency_gains" in report.detailed_metrics
        
        # Check by_strategy has all strategies
        for strategy in RoutingStrategy:
            assert strategy.value in report.detailed_metrics["by_strategy"]
    
    def test_token_breakdown(self, analyzer):
        """Test token breakdown by task type."""
        strategy_result = analyzer.evaluator.results[RoutingStrategy.INFO_EFFICIENT.value]
        
        result = analyzer.analyze_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            strategy_result=strategy_result,
        )
        
        # Token breakdown should have entries
        assert len(result.token_breakdown) > 0
    
    def test_efficiency_curve(self, analyzer):
        """Test efficiency curve generation."""
        strategy_result = analyzer.evaluator.results[RoutingStrategy.INFO_EFFICIENT.value]
        
        result = analyzer.analyze_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            strategy_result=strategy_result,
        )
        
        # Curve should have points
        assert len(result.efficiency_curve) > 0
        
        # Each point should be (tokens, accuracy)
        for tokens, acc in result.efficiency_curve:
            assert isinstance(tokens, int)
            assert 0 <= acc <= 1
    
    def test_empty_results_error(self):
        """Test error when no results available."""
        evaluator = Mock(spec=ComparativeEvaluator)
        evaluator.results = {}
        
        analyzer = TokenEfficiencyAnalyzer(evaluator)
        
        with pytest.raises(ValueError):
            analyzer.run_analysis()


class TestRunEfficiencyAnalysis:
    """Tests for run_efficiency_analysis function."""
    
    def test_basic_analysis(self):
        """Test running basic efficiency analysis."""
        report = run_efficiency_analysis(
            num_agents=2,
            num_tasks=10,
            model_quality=0.8,
            seed=42,
        )
        
        assert report is not None
        assert isinstance(report, EfficiencyReport)
        assert len(report.results_by_strategy) == 3
    
    def test_reproducibility(self):
        """Test reproducibility with seed."""
        report1 = run_efficiency_analysis(
            num_agents=2,
            num_tasks=5,
            seed=42,
        )
        
        report2 = run_efficiency_analysis(
            num_agents=2,
            num_tasks=5,
            seed=42,
        )
        
        # Should produce same ranking
        assert report1.efficiency_ranking == report2.efficiency_ranking
    
    def test_different_agent_counts(self):
        """Test with different agent counts."""
        for num_agents in [2, 3, 5]:
            report = run_efficiency_analysis(
                num_agents=num_agents,
                num_tasks=5,
                seed=42,
            )
            
            assert len(report.results_by_strategy) == 3


class TestEfficiencyComparisons:
    """Tests comparing efficiency across strategies."""
    
    @pytest.fixture
    def setup_analysis(self):
        """Setup for efficiency comparison tests."""
        report = run_efficiency_analysis(
            num_agents=3,
            num_tasks=20,
            model_quality=0.8,
            seed=42,
        )
        return report
    
    def test_all_strategies_analyzed(self, setup_analysis):
        """Test that all strategies are analyzed."""
        report = setup_analysis
        
        for strategy in RoutingStrategy:
            assert strategy.value in report.results_by_strategy
    
    def test_info_efficient_vs_uniform(self, setup_analysis):
        """Test info-efficient vs uniform comparison."""
        report = setup_analysis
        
        info = report.results_by_strategy.get(RoutingStrategy.INFO_EFFICIENT.value)
        uniform = report.results_by_strategy.get(RoutingStrategy.UNIFORM.value)
        
        assert info is not None
        assert uniform is not None
        
        # Both should have valid metrics
        assert info.metrics.total_tokens > 0
        assert uniform.metrics.total_tokens > 0
    
    def test_efficiency_gains_computed(self, setup_analysis):
        """Test that efficiency gains are computed."""
        report = setup_analysis
        
        # Should have efficiency gains computed
        assert len(report.detailed_metrics["efficiency_gains"]) > 0
    
    def test_recommendations_generated(self, setup_analysis):
        """Test that recommendations are generated."""
        report = setup_analysis
        
        assert len(report.recommendations) > 0
        
        # Each recommendation should be a non-empty string
        for rec in report.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_correct_answers(self):
        """Test handling zero correct answers."""
        metrics = EfficiencyMetrics(
            tokens_per_correct_answer=float('inf'),
            correct_answers=0,
            total_tokens=1000,
        )
        
        assert metrics.tokens_per_correct_answer == float('inf')
    
    def test_zero_tokens(self):
        """Test handling zero tokens."""
        metrics = EfficiencyMetrics(
            total_tokens=0,
            correct_answers=0,
            tokens_per_correct_answer=0.0,
        )
        
        assert metrics.total_tokens == 0
    
    def test_single_task(self):
        """Test with single task."""
        report = run_efficiency_analysis(
            num_agents=2,
            num_tasks=1,
            seed=42,
        )
        
        # Should handle single task gracefully
        assert len(report.results_by_strategy) == 3


class TestIntegration:
    """Integration tests for efficiency analysis."""
    
    def test_full_pipeline(self):
        """Test the full efficiency analysis pipeline."""
        # Create environment
        env, agents = create_evaluation_environment(num_agents=3, seed=42)
        
        # Create evaluator
        evaluator = ComparativeEvaluator(env, agents, seed=42)
        
        # Run comparison
        from src.benchmarks import MultiAgentBenchmarkRunner
        runner = MultiAgentBenchmarkRunner()
        tasks = runner.get_all_tasks().sample(15, seed=42)
        
        evaluator.run_comparison(tasks=tasks, model_quality=0.8)
        
        # Analyze efficiency
        analyzer = TokenEfficiencyAnalyzer(evaluator)
        report = analyzer.run_analysis()
        
        # Verify report
        assert report.savings_summary is not None
        assert len(report.recommendations) > 0
        assert len(report.efficiency_ranking) == 3
    
    def test_with_different_capacities(self):
        """Test with varying agent capacities."""
        env, agents = create_evaluation_environment(num_agents=3, seed=42)
        
        # Set very different capacities
        capacities = [0.95, 0.3, 0.2]
        
        evaluator = ComparativeEvaluator(env, agents, seed=42)
        evaluator.run_comparison(
            agent_capacities=capacities,
            model_quality=0.8,
        )
        
        analyzer = TokenEfficiencyAnalyzer(evaluator)
        report = analyzer.run_analysis()
        
        # Info-efficient should potentially benefit from high variance
        assert report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
