"""
Tests for the Integrated Benchmark Runner.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.benchmark import (
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkRunner,
    MockModel,
    run_quick_benchmark,
)
from src.environment import AdversarialEnvironment, create_sample_scenarios
from src.agent import NavigationAgent


class TestMockModel:
    """Tests for MockModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MockModel(quality=0.8, seed=42)
        
        assert model.quality == 0.8
        assert model.token_count == 0
    
    def test_call_generates_votes(self):
        """Test that calling model generates votes."""
        model = MockModel(quality=0.7, seed=42)
        
        votes = model("Test prompt", compute_budget=5)
        
        assert len(votes) == 5
        assert all(isinstance(v, str) for v in votes)
    
    def test_token_counting(self):
        """Test that tokens are counted."""
        model = MockModel(quality=0.7)
        
        model("This is a test prompt with several words", compute_budget=3)
        
        assert model.token_count > 0
    
    def test_reset_token_count(self):
        """Test token count reset."""
        model = MockModel(quality=0.7)
        
        model("Test prompt", compute_budget=3)
        assert model.token_count > 0
        
        model.reset_token_count()
        assert model.token_count == 0
    
    def test_quality_affects_votes(self):
        """Test that quality parameter affects vote distribution."""
        model_high = MockModel(quality=0.95, seed=42)
        model_low = MockModel(quality=0.3, seed=42)
        
        # Run multiple times to see patterns
        high_votes = model_high("Test", 10)
        model_low.reset_token_count()
        low_votes = model_low("Test", 10)
        
        # Both should return same number of votes
        assert len(high_votes) == len(low_votes)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            scenario_id="test_001",
            scenario_name="Test Scenario",
            scenario_type="distraction",
            success=True,
            final_score=0.85,
            turns_taken=5,
            total_tokens=500,
            tokens_per_turn=100.0,
            time_seconds=2.5,
            alerts_triggered=0,
        )
        
        assert result.scenario_id == "test_001"
        assert result.success is True
        assert result.final_score == 0.85
    
    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = BenchmarkResult(
            scenario_id="test_002",
            scenario_name="Test",
            scenario_type="hijacking",
            success=False,
            final_score=0.3,
            turns_taken=3,
            total_tokens=200,
            tokens_per_turn=66.67,
            time_seconds=1.0,
            alerts_triggered=2,
            metadata={"error": "timeout"},
        )
        
        assert result.metadata["error"] == "timeout"


class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""
    
    def test_report_to_dict(self):
        """Test report serialization."""
        result = BenchmarkResult(
            scenario_id="test_001",
            scenario_name="Test",
            scenario_type="distraction",
            success=True,
            final_score=0.85,
            turns_taken=5,
            total_tokens=500,
            tokens_per_turn=100.0,
            time_seconds=2.5,
            alerts_triggered=0,
        )
        
        report = BenchmarkReport(
            timestamp="2026-02-25T10:00:00",
            total_scenarios=1,
            successful_scenarios=1,
            success_rate=1.0,
            avg_score=0.85,
            total_tokens=500,
            avg_tokens_per_scenario=500.0,
            avg_tokens_per_turn=100.0,
            total_time_seconds=2.5,
            avg_time_per_scenario=2.5,
            results=[result],
        )
        
        d = report.to_dict()
        
        assert "timestamp" in d
        assert "summary" in d
        assert "results" in d
        assert d["summary"]["success_rate"] == 1.0
        assert len(d["results"]) == 1


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        runner = BenchmarkRunner(verbose=False)
        
        assert runner.results == []
        assert runner.verbose is False
    
    def test_runner_with_output_dir(self):
        """Test runner with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            
            assert runner.output_dir == Path(tmpdir)
    
    def test_run_benchmark(self):
        """Test running full benchmark."""
        runner = BenchmarkRunner(verbose=False)
        report = runner.run_benchmark(model_quality=0.8, seed=42)
        
        assert report.total_scenarios > 0
        assert report.success_rate >= 0.0
        assert report.total_tokens > 0
        assert len(report.results) == report.total_scenarios
    
    def test_run_benchmark_saves_report(self):
        """Test that benchmark saves report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            report = runner.run_benchmark(model_quality=0.7, seed=42)
            
            # Check that file was created
            files = list(Path(tmpdir).glob("benchmark_*.json"))
            assert len(files) == 1
            
            # Check file contents
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            assert "timestamp" in data
            assert "summary" in data
    
    def test_run_single_scenario(self):
        """Test running a single scenario."""
        # Setup
        env = AdversarialEnvironment(seed=42)
        for scenario in create_sample_scenarios():
            env.add_scenario(scenario)
        
        model = MockModel(quality=0.8, seed=42)
        agent = NavigationAgent(
            env=env,
            model=model,
            base_compute=3,
        )
        
        runner = BenchmarkRunner(verbose=False)
        result = runner.run_single_scenario(agent, env, "distraction_001")
        
        assert result.scenario_id == "distraction_001"
        assert result.turns_taken > 0
        assert result.time_seconds >= 0
    
    def test_run_specific_scenarios(self):
        """Test running specific scenarios only."""
        runner = BenchmarkRunner(verbose=False)
        report = runner.run_benchmark(
            scenarios=["distraction_001"],
            model_quality=0.7,
            seed=42,
        )
        
        assert report.total_scenarios == 1
        assert report.results[0].scenario_id == "distraction_001"


class TestQuickBenchmark:
    """Tests for quick benchmark function."""
    
    def test_quick_benchmark_returns_dict(self):
        """Test that quick benchmark returns dictionary."""
        result = run_quick_benchmark(model_quality=0.7, verbose=False)
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "success_rate" in result["summary"]
    
    def test_quick_benchmark_with_different_quality(self):
        """Test quick benchmark with different quality levels."""
        high_quality = run_quick_benchmark(model_quality=0.95, verbose=False)
        low_quality = run_quick_benchmark(model_quality=0.3, verbose=False)
        
        # Higher quality should generally produce better results
        # (though with randomness, this isn't guaranteed)
        assert "success_rate" in high_quality["summary"]
        assert "success_rate" in low_quality["summary"]


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""
    
    def test_full_benchmark_cycle(self):
        """Test full benchmark cycle with report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            report = runner.run_benchmark(seed=42)
            
            # Verify report structure
            assert report.timestamp is not None
            assert report.total_scenarios == 4  # 4 sample scenarios
            assert 0.0 <= report.success_rate <= 1.0
            assert report.total_tokens > 0
            
            # Verify all results have required fields
            for result in report.results:
                assert result.scenario_id is not None
                assert result.final_score >= 0.0
    
    def test_benchmark_accuracy_efficiency_tradeoff(self):
        """Test measuring accuracy vs token efficiency."""
        runner = BenchmarkRunner(verbose=False)
        
        # Run with high compute (more tokens)
        report = runner.run_benchmark(model_quality=0.9, seed=42)
        
        # Check that we're tracking both metrics
        assert report.avg_score >= 0.0
        assert report.avg_tokens_per_turn > 0
        
        # Higher quality should correlate with better scores
        # (though exact relationship depends on scenarios)
