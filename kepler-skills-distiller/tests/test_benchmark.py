"""
Tests for Equation Discovery Benchmark
"""

import pytest
import math
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile

from src.benchmark import (
    EquationProblem,
    BenchmarkResult,
    BenchmarkReport,
    FeynmanDataset,
    EquationDiscoveryBenchmark,
    create_mock_expert,
    create_mock_apprentice,
    run_quick_benchmark,
)


class TestEquationProblem:
    """Tests for EquationProblem."""
    
    def test_init(self):
        """Test problem initialization."""
        problem = EquationProblem(
            name="test",
            category="physics",
            variables=["x", "y"],
            equation="x + y",
            data_points=[{"x": 1.0, "y": 2.0}],
            complexity=1,
        )
        
        assert problem.name == "test"
        assert problem.category == "physics"
        assert problem.variables == ["x", "y"]
        assert problem.equation == "x + y"
    
    def test_evaluate_simple(self):
        """Test simple equation evaluation."""
        problem = EquationProblem(
            name="addition",
            category="math",
            variables=["x", "y"],
            equation="x + y",
            data_points=[],
            complexity=1,
        )
        
        result = problem.evaluate({"x": 3.0, "y": 4.0})
        assert result == 7.0
    
    def test_evaluate_multiplication(self):
        """Test multiplication equation."""
        problem = EquationProblem(
            name="product",
            category="math",
            variables=["a", "b"],
            equation="a * b",
            data_points=[],
            complexity=1,
        )
        
        result = problem.evaluate({"a": 3.0, "b": 5.0})
        assert result == 15.0
    
    def test_evaluate_with_power(self):
        """Test equation with power."""
        problem = EquationProblem(
            name="squared",
            category="math",
            variables=["x"],
            equation="x**2",
            data_points=[],
            complexity=1,
        )
        
        result = problem.evaluate({"x": 4.0})
        assert result == 16.0
    
    def test_evaluate_with_functions(self):
        """Test equation with math functions."""
        problem = EquationProblem(
            name="sine",
            category="math",
            variables=["x"],
            equation="sin(x)",
            data_points=[],
            complexity=2,
        )
        
        result = problem.evaluate({"x": math.pi / 2})
        assert abs(result - 1.0) < 0.0001
    
    def test_evaluate_with_constants(self):
        """Test equation with constants."""
        problem = EquationProblem(
            name="circle",
            category="math",
            variables=["r", "pi"],
            equation="pi * r**2",
            data_points=[],
            complexity=1,
        )
        
        result = problem.evaluate({"r": 2.0, "pi": math.pi})
        expected = math.pi * 4
        assert abs(result - expected) < 0.0001


class TestFeynmanDataset:
    """Tests for FeynmanDataset."""
    
    def test_init(self):
        """Test dataset initialization."""
        dataset = FeynmanDataset()
        
        assert len(dataset.equations) > 0
    
    def test_equations_have_required_fields(self):
        """Test that all equations have required fields."""
        dataset = FeynmanDataset()
        
        for eq in dataset.equations:
            assert eq.name
            assert eq.category
            assert eq.variables
            assert eq.equation
            assert eq.data_points
            assert 1 <= eq.complexity <= 5
    
    def test_get_problems_all(self):
        """Test getting all problems."""
        dataset = FeynmanDataset()
        
        problems = dataset.get_problems()
        
        assert len(problems) == len(dataset.equations)
    
    def test_get_problems_by_category(self):
        """Test filtering by category."""
        dataset = FeynmanDataset()
        
        physics = dataset.get_problems(categories=["physics"])
        
        for p in physics:
            assert p.category == "physics"
    
    def test_get_problems_by_complexity(self):
        """Test filtering by complexity."""
        dataset = FeynmanDataset()
        
        simple = dataset.get_problems(max_complexity=1)
        
        for p in simple:
            assert p.complexity <= 1
    
    def test_data_points_generated(self):
        """Test that data points are generated."""
        dataset = FeynmanDataset()
        
        for eq in dataset.equations:
            assert len(eq.data_points) > 0
            for point in eq.data_points:
                assert isinstance(point, dict)


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""
    
    def test_init(self):
        """Test result initialization."""
        result = BenchmarkResult(
            problem_name="test",
            model_type="expert",
            predicted_equation="x + y",
            exact_match=True,
            relative_error=0.0,
            mae=0.0,
            rmse=0.0,
            time_seconds=1.5,
        )
        
        assert result.problem_name == "test"
        assert result.model_type == "expert"
        assert result.exact_match == True
        assert result.mae == 0.0


class TestBenchmarkReport:
    """Tests for BenchmarkReport."""
    
    def test_init(self):
        """Test report initialization."""
        report = BenchmarkReport(
            timestamp="2026-01-01",
            total_problems=5,
            expert_results=[],
            apprentice_results=[],
            expert_accuracy=0.8,
            apprentice_accuracy=0.6,
            expert_avg_mae=0.1,
            apprentice_avg_mae=0.2,
            expert_avg_time=1.0,
            apprentice_avg_time=0.5,
            improvement_ratio=0.75,
        )
        
        assert report.total_problems == 5
        assert report.expert_accuracy == 0.8
    
    def test_to_dict(self):
        """Test report serialization."""
        report = BenchmarkReport(
            timestamp="2026-01-01",
            total_problems=5,
            expert_results=[],
            apprentice_results=[],
            expert_accuracy=0.8,
            apprentice_accuracy=0.6,
            expert_avg_mae=0.1,
            apprentice_avg_mae=0.2,
            expert_avg_time=1.0,
            apprentice_avg_time=0.5,
            improvement_ratio=0.75,
        )
        
        d = report.to_dict()
        
        assert "timestamp" in d
        assert "summary" in d
        assert d["summary"]["total_problems"] == 5


class TestEquationDiscoveryBenchmark:
    """Tests for EquationDiscoveryBenchmark."""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return EquationDiscoveryBenchmark()
    
    @pytest.fixture
    def mock_expert(self):
        """Create mock expert model."""
        def expert(data_points, variables):
            # Return a simple equation
            return {"equation": " + ".join(variables[:2])}
        return expert
    
    @pytest.fixture
    def mock_apprentice(self):
        """Create mock apprentice model."""
        def apprentice(data_points, variables):
            return {"equation": variables[0]}
        return apprentice
    
    def test_init(self, benchmark):
        """Test benchmark initialization."""
        assert benchmark.dataset is not None
    
    def test_evaluate_model(self, benchmark, mock_expert):
        """Test model evaluation."""
        dataset = FeynmanDataset()
        problems = dataset.get_problems(max_complexity=1)[:2]
        
        results = benchmark.evaluate_model(mock_expert, "expert", problems)
        
        assert len(results) == 2
        assert all(r.model_type == "expert" for r in results)
    
    def test_run_benchmark_expert_only(self, benchmark, mock_expert):
        """Test running benchmark with expert only."""
        report = benchmark.run_benchmark(
            expert_model=mock_expert,
            max_complexity=1,
        )
        
        assert report.total_problems > 0
        assert len(report.expert_results) > 0
        assert report.expert_accuracy >= 0.0
    
    def test_run_benchmark_with_apprentice(self, benchmark, mock_expert, mock_apprentice):
        """Test running benchmark with both models."""
        report = benchmark.run_benchmark(
            expert_model=mock_expert,
            apprentice_model=mock_apprentice,
            max_complexity=1,
        )
        
        assert len(report.expert_results) > 0
        assert len(report.apprentice_results) > 0
        assert report.improvement_ratio >= 0.0
    
    def test_save_report(self, mock_expert):
        """Test saving benchmark report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = EquationDiscoveryBenchmark(output_dir=tmpdir)
            
            report = benchmark.run_benchmark(
                expert_model=mock_expert,
                max_complexity=1,
            )
            
            # Check file was created
            files = list(Path(tmpdir).glob("benchmark_*.json"))
            assert len(files) > 0
    
    def test_check_exact_match(self, benchmark):
        """Test exact match checking."""
        assert benchmark._check_exact_match("x + y", "x+y")
        assert benchmark._check_exact_match("X + Y", "x + y")
        assert not benchmark._check_exact_match("x + y", "x + z")
    
    def test_calculate_errors(self, benchmark):
        """Test error calculation."""
        problem = EquationProblem(
            name="test",
            category="math",
            variables=["x", "y"],
            equation="x + y",
            data_points=[
                {"x": 1.0, "y": 2.0},
                {"x": 2.0, "y": 3.0},
            ],
            complexity=1,
        )
        
        mae, rmse = benchmark._calculate_errors("x + y", problem)
        
        # Exact match should have zero error
        assert mae < 0.0001
        assert rmse < 0.0001


class TestMockModels:
    """Tests for mock model factories."""
    
    def test_create_mock_expert(self):
        """Test mock expert creation."""
        expert = create_mock_expert()
        
        result = expert([{"x": 1, "y": 2}], ["x", "y"])
        
        assert "equation" in result
        assert isinstance(result["equation"], str)
    
    def test_create_mock_apprentice(self):
        """Test mock apprentice creation."""
        apprentice = create_mock_apprentice()
        
        result = apprentice([{"x": 1, "y": 2}], ["x", "y"])
        
        assert "equation" in result


class TestQuickBenchmark:
    """Tests for quick benchmark function."""
    
    def test_run_quick_benchmark(self):
        """Test quick benchmark execution."""
        report = run_quick_benchmark()
        
        assert report.total_problems > 0
        assert report.expert_accuracy >= 0.0
        assert report.apprentice_accuracy >= 0.0


class TestBenchmarkIntegration:
    """Integration tests for benchmark."""
    
    def test_feynman_equations_valid(self):
        """Test that all Feynman equations can be evaluated."""
        dataset = FeynmanDataset()
        
        for eq in dataset.equations:
            # Should be able to evaluate with sample data
            point = eq.data_points[0]
            try:
                result = eq.evaluate(point)
                # Result should be a number (or nan for complex cases)
                assert isinstance(result, (int, float))
            except Exception as e:
                pytest.fail(f"Failed to evaluate {eq.name}: {e}")
    
    def test_benchmark_with_real_equations(self):
        """Test benchmark with actual Feynman equations."""
        benchmark = EquationDiscoveryBenchmark()
        
        # Create a model that tries to find the equation
        def simple_model(data_points, variables):
            # Just return a simple equation for testing
            return {"equation": variables[0] + " + " + variables[1] if len(variables) >= 2 else variables[0]}
        
        report = benchmark.run_benchmark(
            expert_model=simple_model,
            max_complexity=2,
        )
        
        # Should complete without errors
        assert report.total_problems > 0
        assert all(r.time_seconds >= 0 for r in report.expert_results)