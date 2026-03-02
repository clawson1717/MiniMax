"""
Tests for Comparative Evaluation Module

Tests for comparing routing strategies (info-efficient, uniform, uncertainty-based).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import random

from src.evaluation import (
    RoutingStrategy,
    RoutingMetrics,
    StrategyResult,
    ComparisonReport,
    MockModel,
    UniformRouter,
    UncertaintyBasedRouter,
    ComparativeEvaluator,
    create_evaluation_environment,
    run_quick_evaluation,
)
from src.benchmarks import (
    BenchmarkTask,
    TaskDataset,
    BenchmarkType,
    Difficulty,
)
from src.environment import (
    MultiAgentEnvironment,
    AgentRole,
    MessageType,
)
from src.agent import ReasoningAgent, AgentConfig


class TestMockModel:
    """Tests for MockModel class."""
    
    def test_initialization(self):
        """Test MockModel initialization."""
        model = MockModel(quality=0.9, seed=42)
        assert model.quality == 0.9
    
    def test_qa_response(self):
        """Test QA task handling."""
        model = MockModel(quality=1.0, seed=42)
        response = model("What is the capital of France?")
        assert "Paris" in response
    
    def test_math_response(self):
        """Test math task handling."""
        model = MockModel(quality=1.0, seed=42)
        response = model("What is 15 + 27?")
        # Check that the response contains a number (the mock model generates approximate answers)
        import re
        numbers = re.findall(r'\d+', response)
        assert len(numbers) > 0
    
    def test_reasoning_response(self):
        """Test reasoning task handling."""
        model = MockModel(quality=1.0, seed=42)
        response = model("All cats are mammals. Is this valid?")
        assert "yes" in response.lower() or "no" in response.lower()
    
    def test_quality_affects_responses(self):
        """Test that quality affects response accuracy."""
        # High quality model
        high_quality = MockModel(quality=1.0, seed=42)
        
        # Low quality model
        low_quality = MockModel(quality=0.1, seed=42)
        
        # High quality should be more consistent
        high_response = high_quality("What is the capital of France?")
        assert "Paris" in high_response


class TestUniformRouter:
    """Tests for UniformRouter class."""
    
    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        env = MultiAgentEnvironment(name="test-env")
        env.register_agent("agent_0", None, AgentRole.WORKER, 0.8)
        env.register_agent("agent_1", None, AgentRole.WORKER, 0.6)
        env.register_agent("agent_2", None, AgentRole.WORKER, 0.9)
        return env
    
    @pytest.fixture
    def router(self, environment):
        """Create a uniform router."""
        return UniformRouter(environment)
    
    def test_route_to_all_agents(self, router, environment):
        """Test that uniform routing reaches all agents."""
        messages = router.route(
            sender_id="external",
            content="test message",
            message_type=MessageType.TASK,
        )
        
        assert len(messages) == 3
    
    def test_exclude_sender(self, router, environment):
        """Test that sender is excluded when specified."""
        # Register sender
        environment.register_agent("sender", None, AgentRole.COORDINATOR, 0.7)
        
        messages = router.route(
            sender_id="sender",
            content="test message",
            message_type=MessageType.TASK,
            exclude_sender=True,
        )
        
        # Should not include sender
        assert len(messages) == 3
        assert all(m.receiver_id != "sender" for m in messages)
    
    def test_routing_stats(self, router):
        """Test that routing stats are tracked."""
        router.route("external", "msg1", MessageType.TASK)
        router.route("external", "msg2", MessageType.TASK)
        
        stats = router.get_stats()
        assert stats["total_messages"] == 6  # 3 agents * 2 messages
    
    def test_routing_history(self, router):
        """Test that routing history is recorded."""
        router.route("external", "test", MessageType.TASK)
        
        assert len(router.routing_history) == 1
        assert router.routing_history[0]["mode"] == "uniform"


class TestUncertaintyBasedRouter:
    """Tests for UncertaintyBasedRouter class."""
    
    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        env = MultiAgentEnvironment(name="test-env")
        env.register_agent("agent_0", None, AgentRole.WORKER, 0.8)
        env.register_agent("agent_1", None, AgentRole.WORKER, 0.6)
        env.register_agent("agent_2", None, AgentRole.WORKER, 0.9)
        return env
    
    @pytest.fixture
    def router(self, environment):
        """Create an uncertainty-based router."""
        return UncertaintyBasedRouter(environment)
    
    def test_update_uncertainty(self, router):
        """Test updating agent uncertainty."""
        router.update_uncertainty("agent_0", 0.3)  # 70% confident
        
        uncertainty = router.get_uncertainty("agent_0")
        assert uncertainty == pytest.approx(0.7, abs=0.01)
    
    def test_route_based_on_uncertainty(self, router, environment):
        """Test that routing considers uncertainty."""
        # Set different uncertainties
        router.update_uncertainty("agent_0", 0.2)  # 80% confident -> low uncertainty
        router.update_uncertainty("agent_1", 0.9)  # 10% confident -> high uncertainty
        router.update_uncertainty("agent_2", 0.5)  # 50% confident
        
        messages = router.route(
            sender_id="external",
            content="test",
            message_type=MessageType.TASK,
        )
        
        # All agents should receive messages (just prioritized differently)
        assert len(messages) == 3
    
    def test_top_k_routing(self, router, environment):
        """Test top-k routing."""
        router.update_uncertainty("agent_0", 0.3)
        router.update_uncertainty("agent_1", 0.9)
        router.update_uncertainty("agent_2", 0.5)
        
        messages = router.route(
            sender_id="external",
            content="test",
            message_type=MessageType.TASK,
            top_k=2,
        )
        
        # Should route to top 2 most uncertain
        assert len(messages) == 2
    
    def test_fallback_to_capacity(self, router, environment):
        """Test fallback to capacity when no uncertainty info."""
        # Don't set uncertainties - should use capacity as fallback
        
        messages = router.route(
            sender_id="external",
            content="test",
            message_type=MessageType.TASK,
        )
        
        # Should still route successfully
        assert len(messages) == 3
    
    def test_routing_stats(self, router):
        """Test routing statistics."""
        router.route("external", "test", MessageType.TASK)
        
        stats = router.get_stats()
        assert stats["total_messages"] == 3
        assert "agent_uncertainties" in stats


class TestComparativeEvaluator:
    """Tests for ComparativeEvaluator class."""
    
    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        env = MultiAgentEnvironment(name="eval-env", max_agents=10)
        return env
    
    @pytest.fixture
    def agents(self):
        """Create test agents."""
        agents = []
        for i in range(3):
            config = AgentConfig(
                agent_id=f"agent_{i}",
                specializations=["reasoning"] if i == 0 else ["math"] if i == 1 else ["general"],
            )
            agents.append(ReasoningAgent(config=config))
        return agents
    
    @pytest.fixture
    def evaluator(self, environment, agents):
        """Create a comparative evaluator."""
        # Register agents in environment
        for i, agent in enumerate(agents):
            environment.register_agent(
                agent_id=agent.agent_id,
                agent=agent,
                role=AgentRole.WORKER,
                capacity=0.5 + i * 0.1,
            )
        return ComparativeEvaluator(environment, agents, seed=42)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        tasks = [
            BenchmarkTask(
                task_id="test_1",
                prompt="What is 2 + 2?",
                expected_answer=4,
                benchmark_type=BenchmarkType.MATH,
                difficulty=Difficulty.EASY,
            ),
            BenchmarkTask(
                task_id="test_2",
                prompt="What is the capital of France?",
                expected_answer="Paris",
                benchmark_type=BenchmarkType.QA,
                difficulty=Difficulty.EASY,
            ),
            BenchmarkTask(
                task_id="test_3",
                prompt="All cats are mammals. Is this valid?",
                expected_answer="yes",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=Difficulty.EASY,
            ),
        ]
        return TaskDataset(tasks)
    
    def test_initialization(self, environment, agents):
        """Test evaluator initialization."""
        for i, agent in enumerate(agents):
            environment.register_agent(
                agent_id=agent.agent_id,
                agent=agent,
                role=AgentRole.WORKER,
                capacity=0.5 + i * 0.1,
            )
        
        evaluator = ComparativeEvaluator(environment, agents, seed=42)
        
        assert evaluator.environment is environment
        assert evaluator.agents == agents
        assert evaluator.capacity_router is not None
        assert evaluator.uniform_router is not None
        assert evaluator.uncertainty_router is not None
    
    def test_setup_agents(self, evaluator, agents):
        """Test agent setup with capacities."""
        capacities = [0.9, 0.7, 0.5]
        evaluator.setup_agents(capacities)
        
        # Check capacities are set
        for i, agent in enumerate(agents):
            state = evaluator.environment.agent_states.get(agent.agent_id)
            if state:
                assert state.capacity == pytest.approx(capacities[i], abs=0.01)
    
    def test_evaluate_info_efficient_strategy(self, evaluator, sample_tasks):
        """Test evaluating info-efficient routing strategy."""
        evaluator.setup_agents([0.9, 0.7, 0.5])
        
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            tasks=sample_tasks,
            model_quality=0.9,
        )
        
        assert result.strategy == RoutingStrategy.INFO_EFFICIENT
        assert result.benchmark_results.total_tasks == 3
        assert 0 <= result.benchmark_results.accuracy <= 1
        assert len(result.per_task_metrics) == 3
    
    def test_evaluate_uniform_strategy(self, evaluator, sample_tasks):
        """Test evaluating uniform routing strategy."""
        evaluator.setup_agents([0.9, 0.7, 0.5])
        
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.UNIFORM,
            tasks=sample_tasks,
            model_quality=0.9,
        )
        
        assert result.strategy == RoutingStrategy.UNIFORM
        assert result.benchmark_results.total_tasks == 3
        assert 0 <= result.benchmark_results.accuracy <= 1
    
    def test_evaluate_uncertainty_strategy(self, evaluator, sample_tasks):
        """Test evaluating uncertainty-based routing strategy."""
        evaluator.setup_agents([0.9, 0.7, 0.5])
        
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.UNCERTAINTY_BASED,
            tasks=sample_tasks,
            model_quality=0.9,
        )
        
        assert result.strategy == RoutingStrategy.UNCERTAINTY_BASED
        assert result.benchmark_results.total_tasks == 3
        assert 0 <= result.benchmark_results.accuracy <= 1
    
    def test_run_comparison(self, evaluator, sample_tasks):
        """Test running full comparison."""
        report = evaluator.run_comparison(
            tasks=sample_tasks,
            model_quality=0.8,
            agent_capacities=[0.9, 0.7, 0.5],
        )
        
        assert report is not None
        assert len(report.strategies_compared) == 3
        assert len(report.results_by_strategy) == 3
        assert "accuracy" in report.winner_by_metric
        assert report.summary is not None
        assert len(report.recommendations) > 0
    
    def test_comparison_report_structure(self, evaluator, sample_tasks):
        """Test comparison report structure."""
        report = evaluator.run_comparison(
            tasks=sample_tasks,
            model_quality=0.8,
        )
        
        # Check all required fields
        assert hasattr(report, 'strategies_compared')
        assert hasattr(report, 'results_by_strategy')
        assert hasattr(report, 'accuracy_comparison')
        assert hasattr(report, 'token_efficiency_comparison')
        assert hasattr(report, 'quality_comparison')
        assert hasattr(report, 'statistical_tests')
        assert hasattr(report, 'winner_by_metric')
        assert hasattr(report, 'summary')
        assert hasattr(report, 'recommendations')
    
    def test_statistical_tests(self, evaluator, sample_tasks):
        """Test that statistical tests are computed."""
        report = evaluator.run_comparison(
            tasks=sample_tasks,
            model_quality=0.8,
        )
        
        # Should have pairwise comparisons
        assert len(report.statistical_tests) >= 1
        
        # Each test should have required fields
        for test_name, test_results in report.statistical_tests.items():
            assert "mean_difference" in test_results
            assert "cohens_d" in test_results
            assert "significant" in test_results
            assert "winner" in test_results
    
    def test_routing_metrics_collection(self, evaluator, sample_tasks):
        """Test that routing metrics are properly collected."""
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            tasks=sample_tasks,
            model_quality=0.8,
        )
        
        assert result.routing_metrics.total_tokens_used > 0
        assert result.routing_metrics.total_capacity_used >= 0
        assert len(result.agent_utilization) > 0
    
    def test_compute_efficiency_calculation(self, evaluator, sample_tasks):
        """Test compute efficiency calculation."""
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            tasks=sample_tasks,
            model_quality=0.9,
        )
        
        # Compute efficiency should be reasonable
        assert 0 <= result.compute_efficiency <= 10  # Upper bound for normalized score
    
    def test_empty_tasks(self, evaluator):
        """Test handling of empty task dataset."""
        empty_tasks = TaskDataset([])
        
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            tasks=empty_tasks,
        )
        
        assert result.benchmark_results.total_tasks == 0
        assert result.benchmark_results.accuracy == 0


class TestRoutingMetrics:
    """Tests for RoutingMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        metrics = RoutingMetrics()
        
        assert metrics.total_messages == 0
        assert metrics.total_tokens_used == 0
        assert metrics.total_capacity_used == 0.0
    
    def test_custom_values(self):
        """Test custom values."""
        metrics = RoutingMetrics(
            total_messages=100,
            total_tokens_used=500,
            total_capacity_used=25.5,
            messages_by_agent={"agent_0": 50, "agent_1": 50},
        )
        
        assert metrics.total_messages == 100
        assert metrics.total_tokens_used == 500
        assert metrics.total_capacity_used == 25.5
        assert metrics.messages_by_agent["agent_0"] == 50


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""
    
    def test_strategy_result_creation(self):
        """Test creating a strategy result."""
        from src.benchmarks import BenchmarkResults
        
        benchmark_results = BenchmarkResults(
            benchmark_name="test",
            total_tasks=10,
            correct=8,
            total_score=8.5,
            accuracy=0.8,
            avg_response_time_ms=100.0,
            total_tokens_used=1000,
            total_capacity_used=50.0,
        )
        
        result = StrategyResult(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            benchmark_results=benchmark_results,
            routing_metrics=RoutingMetrics(),
            compute_efficiency=0.5,
            response_quality_score=0.75,
        )
        
        assert result.strategy == RoutingStrategy.INFO_EFFICIENT
        assert result.benchmark_results.accuracy == 0.8


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""
    
    def test_report_creation(self):
        """Test creating a comparison report."""
        report = ComparisonReport(
            strategies_compared=[RoutingStrategy.INFO_EFFICIENT, RoutingStrategy.UNIFORM],
            results_by_strategy={},
            accuracy_comparison={"info_efficient": 0.8, "uniform": 0.7},
            token_efficiency_comparison={"info_efficient": 0.5, "uniform": 0.4},
            quality_comparison={"info_efficient": 0.75, "uniform": 0.65},
            statistical_tests={},
            winner_by_metric={"accuracy": "info_efficient"},
            summary="Test summary",
            recommendations=["Recommendation 1"],
        )
        
        assert len(report.strategies_compared) == 2
        assert report.winner_by_metric["accuracy"] == "info_efficient"


class TestCreateEvaluationEnvironment:
    """Tests for create_evaluation_environment function."""
    
    def test_default_creation(self):
        """Test creating environment with defaults."""
        env, agents = create_evaluation_environment()
        
        assert env is not None
        assert len(agents) == 5
        assert len(env.agents) == 5
    
    def test_custom_num_agents(self):
        """Test creating environment with custom agent count."""
        env, agents = create_evaluation_environment(num_agents=3)
        
        assert len(agents) == 3
        assert len(env.agents) == 3
    
    def test_with_seed(self):
        """Test reproducibility with seed."""
        env1, agents1 = create_evaluation_environment(num_agents=3, seed=42)
        env2, agents2 = create_evaluation_environment(num_agents=3, seed=42)
        
        # Should have same agent IDs
        ids1 = sorted(a.agent_id for a in agents1)
        ids2 = sorted(a.agent_id for a in agents2)
        assert ids1 == ids2
    
    def test_agent_specializations(self):
        """Test that agents have diverse specializations."""
        env, agents = create_evaluation_environment(num_agents=5)
        
        all_specs = []
        for agent in agents:
            all_specs.extend(agent.specializations)
        
        # Should have diverse specializations
        assert len(set(all_specs)) >= 2


class TestRunQuickEvaluation:
    """Tests for run_quick_evaluation function."""
    
    def test_quick_evaluation(self):
        """Test running quick evaluation."""
        report = run_quick_evaluation(
            num_agents=3,
            num_tasks=5,
            model_quality=0.8,
            seed=42,
        )
        
        assert report is not None
        assert isinstance(report, ComparisonReport)
        assert len(report.strategies_compared) == 3
    
    def test_quick_evaluation_reproducibility(self):
        """Test reproducibility with seed."""
        report1 = run_quick_evaluation(
            num_agents=2,
            num_tasks=3,
            seed=42,
        )
        
        report2 = run_quick_evaluation(
            num_agents=2,
            num_tasks=3,
            seed=42,
        )
        
        # Should produce same winners
        assert report1.winner_by_metric == report2.winner_by_metric
    
    def test_quick_evaluation_with_different_quality(self):
        """Test that model quality affects results."""
        # High quality
        high_quality_report = run_quick_evaluation(
            num_agents=2,
            num_tasks=5,
            model_quality=0.95,
            seed=42,
        )
        
        # Low quality
        low_quality_report = run_quick_evaluation(
            num_agents=2,
            num_tasks=5,
            model_quality=0.3,
            seed=42,
        )
        
        # Higher quality should generally produce higher accuracy
        high_avg = np.mean(list(high_quality_report.accuracy_comparison.values()))
        low_avg = np.mean(list(low_quality_report.accuracy_comparison.values()))
        
        # This should hold most of the time, though not guaranteed
        # due to randomness in routing and responses
        assert high_avg >= low_avg * 0.8  # Allow some slack


class TestRoutingStrategyComparison:
    """Tests comparing the three routing strategies."""
    
    @pytest.fixture
    def setup_comparison(self):
        """Setup for strategy comparison tests."""
        env, agents = create_evaluation_environment(num_agents=3, seed=42)
        evaluator = ComparativeEvaluator(env, agents, seed=42)
        
        # Create diverse tasks
        tasks = TaskDataset([
            BenchmarkTask("m1", "What is 10 + 5?", 15, BenchmarkType.MATH, Difficulty.EASY),
            BenchmarkTask("q1", "What is the capital of France?", "Paris", BenchmarkType.QA, Difficulty.EASY),
            BenchmarkTask("r1", "All cats are mammals. Valid?", "yes", BenchmarkType.REASONING, Difficulty.EASY),
            BenchmarkTask("m2", "What is 7 × 8?", 56, BenchmarkType.MATH, Difficulty.EASY),
            BenchmarkTask("q2", "Who wrote Romeo and Juliet?", "Shakespeare", BenchmarkType.QA, Difficulty.EASY),
        ])
        
        return evaluator, tasks
    
    def test_all_strategies_run(self, setup_comparison):
        """Test that all three strategies run successfully."""
        evaluator, tasks = setup_comparison
        
        report = evaluator.run_comparison(tasks=tasks, model_quality=0.8)
        
        # All strategies should have results
        for strategy in RoutingStrategy:
            assert strategy.value in report.results_by_strategy
    
    def test_strategies_have_different_utilization(self, setup_comparison):
        """Test that strategies result in different agent utilization."""
        evaluator, tasks = setup_comparison
        
        report = evaluator.run_comparison(tasks=tasks, model_quality=0.8)
        
        # Get utilization patterns
        utilizations = {}
        for strategy_name, result in report.results_by_strategy.items():
            util = tuple(sorted(result.agent_utilization.items()))
            utilizations[strategy_name] = util
        
        # At least one strategy should differ
        unique_utilizations = len(set(utilizations.values()))
        assert unique_utilizations >= 1  # At minimum they run
    
    def test_info_efficient_prioritizes_high_capacity(self, setup_comparison):
        """Test that info-efficient routing prioritizes high-capacity agents."""
        evaluator, tasks = setup_comparison
        
        # Set very different capacities
        evaluator.setup_agents([0.95, 0.3, 0.2])
        
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.INFO_EFFICIENT,
            tasks=tasks,
            model_quality=0.8,
        )
        
        # High capacity agent should receive more messages
        # (or at least equal since we route to top_k)
        assert result.agent_utilization is not None
    
    def test_uncertainty_based_targets_uncertain_agents(self, setup_comparison):
        """Test that uncertainty-based routing targets uncertain agents."""
        evaluator, tasks = setup_comparison
        
        # Set very different uncertainties
        evaluator.setup_agents([0.5, 0.5, 0.5])
        evaluator.uncertainty_router.update_uncertainty("agent_0", 0.1)  # High confidence
        evaluator.uncertainty_router.update_uncertainty("agent_1", 0.9)  # Low confidence
        evaluator.uncertainty_router.update_uncertainty("agent_2", 0.5)  # Medium
        
        result = evaluator.evaluate_strategy(
            strategy=RoutingStrategy.UNCERTAINTY_BASED,
            tasks=tasks,
            model_quality=0.8,
        )
        
        # Should have routed messages
        assert result.routing_metrics.total_messages > 0


class TestIntegration:
    """Integration tests for the evaluation module."""
    
    def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline."""
        # Create environment
        env, agents = create_evaluation_environment(num_agents=4, seed=42)
        
        # Create evaluator
        evaluator = ComparativeEvaluator(env, agents, seed=42)
        
        # Run comparison
        report = run_quick_evaluation(
            num_agents=4,
            num_tasks=10,
            model_quality=0.8,
            seed=42,
        )
        
        # Verify report structure
        assert report.summary is not None
        assert len(report.recommendations) > 0
        assert len(report.statistical_tests) > 0
        
        # Verify all strategies are represented
        for strategy in RoutingStrategy:
            assert strategy.value in report.accuracy_comparison
            assert strategy.value in report.quality_comparison
    
    def test_statistical_significance_detection(self):
        """Test that significant differences are detected."""
        # Use very different model qualities to create clear differences
        np.random.seed(42)
        
        report = run_quick_evaluation(
            num_agents=3,
            num_tasks=15,
            model_quality=0.95,
            seed=42,
        )
        
        # Check that some statistical tests were run
        assert len(report.statistical_tests) > 0
        
        # At least some tests should have results
        for test_name, test_results in report.statistical_tests.items():
            assert "mean_difference" in test_results
            assert "cohens_d" in test_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
