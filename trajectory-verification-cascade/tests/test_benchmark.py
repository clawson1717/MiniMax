import pytest
import uuid
from src.benchmark import TVCBenchmark, BenchmarkTask, BenchmarkResult
from src.agent import TVCAgentConfig, TVCAgent

class TestTVCBenchmark:
    """Unit tests for the TVCBenchmark class."""

    @pytest.fixture
    def benchmark(self):
        """Initializes a benchmark instance."""
        return TVCBenchmark()

    def test_initialization(self, benchmark):
        """Verify the benchmark initializes with default tasks."""
        assert len(benchmark.tasks) >= 5
        assert any(t.id == "std-01" for t in benchmark.tasks)
        assert any(t.is_adversarial for t in benchmark.tasks)
        assert any(t.adversarial_type == "self_doubt" for t in benchmark.tasks)

    def test_add_custom_task(self, benchmark):
        """Test adding a custom task to the benchmark."""
        custom_task = BenchmarkTask(
            id="custom-01",
            name="Custom Test",
            task_prompt="Let's test this.",
            reasoning_steps=["Step 1: content"],
            expected_success=True
        )
        benchmark.tasks.append(custom_task)
        assert len(benchmark.tasks) > 5
        assert benchmark.tasks[-1].id == "custom-01"

    def test_run_benchmark(self, benchmark):
        """Ensure the benchmark run returns a list of result objects."""
        # Simple configuration for fast runs
        config = TVCAgentConfig(max_steps=10)
        results = benchmark.run_benchmark(agent_config=config)
        
        assert len(results) == len(benchmark.tasks)
        assert isinstance(results[0], BenchmarkResult)
        assert isinstance(results[0].metrics, dict)
        assert results[0].node_count > 0

    def test_metrics_calculation(self, benchmark):
        """Verifies the summary metrics calculation."""
        # Create mock results for testing the calculation
        mock_results = [
            BenchmarkResult(
                task_id="std-01", 
                success=True, 
                expected_success=True, 
                metrics={"nodes_pruned": 1}, 
                node_count=4
            ),
            BenchmarkResult(
                task_id="adv-self-doubt", 
                success=False, 
                expected_success=False, 
                metrics={"nodes_pruned": 2}, 
                node_count=6
            )
        ]
        
        # Override standard tasks to match mock ids if needed
        # (Though calculation only cares about indices/types in some fields)
        summary = benchmark.calculate_summary_metrics(mock_results)
        
        assert summary["total_tasks"] == 2
        assert summary["verification_accuracy"] == 1.0
        assert summary["pruning_efficiency"] == (1+2)/(4+6)
        assert "adversarial_failure_detection_rate" in summary

    def test_adversarial_detection_metric(self, benchmark):
        """Specifically check adversarial detection rate logic."""
        # Task ids that represent adversarial tasks in TVCBenchmark
        adv_ids = ["adv-self-doubt", "adv-social-conformity", "adv-suggestion-hijacking", "adv-fatigue"]
        
        # Mock results where 3 adversarial tasks are correctly detected (return success=False)
        # and 1 fails to be detected (returns success=True)
        # (Note: Calculation logic assumes success=False means it was "blocked/detected")
        mock_results = [
            BenchmarkResult(task_id="adv-self-doubt", success=False, expected_success=True, metrics={}), # Detected
            BenchmarkResult(task_id="adv-social-conformity", success=False, expected_success=False, metrics={}), # Detected
            BenchmarkResult(task_id="adv-suggestion-hijacking", success=False, expected_success=False, metrics={}),# Detected
            BenchmarkResult(task_id="adv-fatigue", success=True, expected_success=True, metrics={}) # Missed detection
        ]
        
        summary = benchmark.calculate_summary_metrics(mock_results)
        # Calculation: adv_detected (success=False) / len(adversarial_tasks)
        # 3 / 4 = 0.75
        assert summary["adversarial_failure_detection_rate"] == 0.75
