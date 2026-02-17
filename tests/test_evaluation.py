"""Tests for evaluation framework."""

import pytest
from src.evaluation import (
    Task, TaskResult, BenchmarkResult, TaskType,
    MetricsCalculator, BenchmarkRunner, AblationStudyRunner,
    create_default_tasks, save_results
)


class TestTaskType:
    """Tests for TaskType enum."""
    
    def test_all_task_types(self):
        """Test all task types are defined."""
        assert TaskType.NAVIGATION.value == "navigation"
        assert TaskType.SEARCH.value == "search"
        assert TaskType.FORM_FILLING.value == "form_filling"
        assert TaskType.MULTI_HOP.value == "multi_hop"
        assert TaskType.EXTRACTION.value == "extraction"


class TestTask:
    """Tests for Task dataclass."""
    
    def test_task_creation(self):
        """Test task creation with required fields."""
        task = Task(
            id="test_1",
            name="Test Task",
            task_type=TaskType.NAVIGATION,
            url="https://example.com",
            goal="Navigate to example"
        )
        
        assert task.id == "test_1"
        assert task.name == "Test Task"
        assert task.task_type == TaskType.NAVIGATION
        assert task.url == "https://example.com"
        assert task.goal == "Navigate to example"
        assert task.timeout == 120  # default
    
    def test_task_with_options(self):
        """Test task with optional fields."""
        task = Task(
            id="test_2",
            name="Complex Task",
            task_type=TaskType.SEARCH,
            url="https://google.com",
            goal="Search for something",
            expected_actions=["navigate", "type", "click"],
            success_criteria={"has_results": True},
            timeout=60
        )
        
        assert task.expected_actions == ["navigate", "type", "click"]
        assert task.success_criteria == {"has_results": True}
        assert task.timeout == 60


class TestTaskResult:
    """Tests for TaskResult dataclass."""
    
    def test_task_result_creation(self):
        """Test task result creation."""
        result = TaskResult(
            task_id="test_1",
            success=True,
            steps_taken=5,
            tool_calls=5,
            execution_time=10.5,
            checklist_completion=0.8,
            uncertainty_avg=0.3
        )
        
        assert result.task_id == "test_1"
        assert result.success is True
        assert result.steps_taken == 5
        assert result.error is None
    
    def test_task_result_with_error(self):
        """Test task result with error."""
        result = TaskResult(
            task_id="test_1",
            success=False,
            steps_taken=0,
            tool_calls=0,
            execution_time=1.0,
            checklist_completion=0.0,
            uncertainty_avg=0.0,
            error="Timeout"
        )
        
        assert result.success is False
        assert result.error == "Timeout"


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        task_results = [
            TaskResult("t1", True, 5, 5, 10.0, 0.8, 0.3),
            TaskResult("t2", True, 7, 7, 12.0, 0.9, 0.2),
        ]
        
        result = BenchmarkResult(
            total_tasks=2,
            successful_tasks=2,
            success_rate=1.0,
            avg_steps=6.0,
            avg_tool_calls=6.0,
            avg_execution_time=11.0,
            avg_checklist_completion=0.85,
            avg_uncertainty=0.25,
            task_results=task_results
        )
        
        assert result.total_tasks == 2
        assert result.successful_tasks == 2
        assert result.success_rate == 1.0


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""
    
    def test_success_rate(self):
        """Test success rate calculation."""
        results = [
            TaskResult("t1", True, 5, 5, 10.0, 0.8, 0.3),
            TaskResult("t2", False, 5, 5, 10.0, 0.8, 0.3),
            TaskResult("t3", True, 5, 5, 10.0, 0.8, 0.3),
        ]
        
        rate = MetricsCalculator.calculate_success_rate(results)
        assert rate == pytest.approx(2/3)
    
    def test_success_rate_empty(self):
        """Test success rate with empty results."""
        rate = MetricsCalculator.calculate_success_rate([])
        assert rate == 0.0
    
    def test_avg_steps(self):
        """Test average steps calculation."""
        results = [
            TaskResult("t1", True, 5, 5, 10.0, 0.8, 0.3),
            TaskResult("t2", True, 10, 5, 10.0, 0.8, 0.3),
        ]
        
        avg = MetricsCalculator.calculate_avg_steps(results)
        assert avg == 7.5
    
    def test_avg_tool_calls(self):
        """Test average tool calls calculation."""
        results = [
            TaskResult("t1", True, 5, 3, 10.0, 0.8, 0.3),
            TaskResult("t2", True, 5, 7, 10.0, 0.8, 0.3),
        ]
        
        avg = MetricsCalculator.calculate_avg_tool_calls(results)
        assert avg == 5.0
    
    def test_avg_execution_time(self):
        """Test average execution time calculation."""
        results = [
            TaskResult("t1", True, 5, 5, 5.0, 0.8, 0.3),
            TaskResult("t2", True, 5, 5, 15.0, 0.8, 0.3),
        ]
        
        avg = MetricsCalculator.calculate_avg_execution_time(results)
        assert avg == 10.0
    
    def test_avg_checklist_completion(self):
        """Test average checklist completion calculation."""
        results = [
            TaskResult("t1", True, 5, 5, 10.0, 0.6, 0.3),
            TaskResult("t2", True, 5, 5, 10.0, 1.0, 0.3),
        ]
        
        avg = MetricsCalculator.calculate_avg_checklist_completion(results)
        assert avg == 0.8
    
    def test_avg_uncertainty(self):
        """Test average uncertainty calculation."""
        results = [
            TaskResult("t1", True, 5, 5, 10.0, 0.8, 0.1),
            TaskResult("t2", True, 5, 5, 10.0, 0.8, 0.3),
        ]
        
        avg = MetricsCalculator.calculate_avg_uncertainty(results)
        assert avg == 0.2


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""
    
    def test_runner_initialization(self):
        """Test runner initializes correctly."""
        runner = BenchmarkRunner()
        
        assert runner.agent is None
        assert runner.tasks == []
        assert runner.results == []
    
    def test_add_task(self):
        """Test adding a task."""
        runner = BenchmarkRunner()
        task = Task("t1", "Test", TaskType.NAVIGATION, "https://example.com", "Goal")
        
        runner.add_task(task)
        
        assert len(runner.tasks) == 1
    
    def test_add_tasks(self):
        """Test adding multiple tasks."""
        runner = BenchmarkRunner()
        tasks = [
            Task("t1", "Test1", TaskType.NAVIGATION, "https://a.com", "Goal1"),
            Task("t2", "Test2", TaskType.SEARCH, "https://b.com", "Goal2"),
        ]
        
        runner.add_tasks(tasks)
        
        assert len(runner.tasks) == 2
    
    def test_run_task_mock(self):
        """Test running a task with mock agent."""
        runner = BenchmarkRunner()
        task = Task("t1", "Test", TaskType.NAVIGATION, "https://example.com", "Goal")
        
        result = runner.run_task(task)
        
        assert result.task_id == "t1"
        assert result.success is True
        assert result.steps_taken == 5
    
    def test_run_benchmark_mock(self):
        """Test running benchmark with mock agent."""
        runner = BenchmarkRunner()
        tasks = [
            Task("t1", "Test1", TaskType.NAVIGATION, "https://a.com", "Goal1"),
            Task("t2", "Test2", TaskType.SEARCH, "https://b.com", "Goal2"),
        ]
        runner.add_tasks(tasks)
        
        result = runner.run_benchmark()
        
        assert result.total_tasks == 2
        assert len(result.task_results) == 2
    
    def test_clear_results(self):
        """Test clearing results."""
        runner = BenchmarkRunner()
        task = Task("t1", "Test", TaskType.NAVIGATION, "https://example.com", "Goal")
        runner.add_task(task)
        runner.run_task(task)
        
        runner.clear_results()
        
        assert runner.results == []


class TestAblationStudyRunner:
    """Tests for AblationStudyRunner."""
    
    def test_ablation_initialization(self):
        """Test ablation runner initialization."""
        def factory(config):
            return None
        
        runner = AblationStudyRunner(factory)
        
        assert runner.base_agent_factory is not None
        assert runner.configurations == []
    
    def test_add_configuration(self):
        """Test adding configuration."""
        def factory(config):
            return None
        
        runner = AblationStudyRunner(factory)
        runner.add_configuration("no_pruning", {"use_pruning": False})
        
        assert len(runner.configurations) == 1
        assert runner.configurations[0]["name"] == "no_pruning"
    
    def test_compare_results(self):
        """Test comparing ablation results."""
        def factory(config):
            return None
        
        runner = AblationStudyRunner(factory)
        
        results = {
            "baseline": BenchmarkResult(
                2, 2, 1.0, 5.0, 5.0, 10.0, 0.8, 0.3, []
            ),
            "no_pruning": BenchmarkResult(
                2, 1, 0.5, 7.0, 7.0, 12.0, 0.6, 0.4, []
            ),
        }
        
        comparison = runner.compare_results(results)
        
        assert comparison["baseline"] == "baseline"
        assert "baseline" in comparison["configurations"]
        assert "no_pruning" in comparison["configurations"]
    
    def test_compare_empty_results(self):
        """Test comparing empty results."""
        def factory(config):
            return None
        
        runner = AblationStudyRunner(factory)
        comparison = runner.compare_results({})
        
        assert comparison == {}


class TestCreateDefaultTasks:
    """Tests for create_default_tasks."""
    
    def test_create_default_tasks(self):
        """Test creating default tasks."""
        tasks = create_default_tasks()
        
        assert len(tasks) == 5
        assert all(isinstance(t, Task) for t in tasks)
        
        # Check task types
        types = {t.task_type for t in tasks}
        assert TaskType.NAVIGATION in types
        assert TaskType.SEARCH in types
        assert TaskType.FORM_FILLING in types
        assert TaskType.MULTI_HOP in types
        assert TaskType.EXTRACTION in types
    
    def test_default_task_fields(self):
        """Test default tasks have required fields."""
        tasks = create_default_tasks()
        
        for task in tasks:
            assert task.id
            assert task.name
            assert task.url
            assert task.goal
            assert task.timeout > 0


class TestSaveResults:
    """Tests for save_results."""
    
    def test_save_results(self, tmp_path):
        """Test saving results to file."""
        result = BenchmarkResult(
            total_tasks=1,
            successful_tasks=1,
            success_rate=1.0,
            avg_steps=5.0,
            avg_tool_calls=5.0,
            avg_execution_time=10.0,
            avg_checklist_completion=0.8,
            avg_uncertainty=0.3,
            task_results=[
                TaskResult("t1", True, 5, 5, 10.0, 0.8, 0.3)
            ]
        )
        
        filepath = tmp_path / "results.json"
        save_results(result, str(filepath))
        
        assert filepath.exists()
        
        import json
        with open(filepath) as f:
            data = json.load(f)
        
        assert data["total_tasks"] == 1
        assert data["success_rate"] == 1.0
        assert len(data["task_results"]) == 1
