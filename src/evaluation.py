"""Evaluation framework for agent benchmarking."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import json


class TaskType(Enum):
    """Types of benchmark tasks."""
    NAVIGATION = "navigation"
    SEARCH = "search"
    FORM_FILLING = "form_filling"
    MULTI_HOP = "multi_hop"
    EXTRACTION = "extraction"


@dataclass
class Task:
    """A benchmark task definition."""
    id: str
    name: str
    task_type: TaskType
    url: str
    goal: str
    expected_actions: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 120
    
    
@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    success: bool
    steps_taken: int
    tool_calls: int
    execution_time: float
    checklist_completion: float
    uncertainty_avg: float
    error: Optional[str] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a full benchmark run."""
    total_tasks: int
    successful_tasks: int
    success_rate: float
    avg_steps: float
    avg_tool_calls: float
    avg_execution_time: float
    avg_checklist_completion: float
    avg_uncertainty: float
    task_results: List[TaskResult]
    timestamp: float = field(default_factory=time.time)


class MetricsCalculator:
    """Calculate metrics from task results."""
    
    @staticmethod
    def calculate_success_rate(results: List[TaskResult]) -> float:
        """Calculate success rate."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)
    
    @staticmethod
    def calculate_avg_steps(results: List[TaskResult]) -> float:
        """Calculate average steps per task."""
        if not results:
            return 0.0
        return sum(r.steps_taken for r in results) / len(results)
    
    @staticmethod
    def calculate_avg_tool_calls(results: List[TaskResult]) -> float:
        """Calculate average tool calls per task."""
        if not results:
            return 0.0
        return sum(r.tool_calls for r in results) / len(results)
    
    @staticmethod
    def calculate_avg_execution_time(results: List[TaskResult]) -> float:
        """Calculate average execution time."""
        if not results:
            return 0.0
        return sum(r.execution_time for r in results) / len(results)
    
    @staticmethod
    def calculate_avg_checklist_completion(results: List[TaskResult]) -> float:
        """Calculate average checklist completion."""
        if not results:
            return 0.0
        return sum(r.checklist_completion for r in results) / len(results)
    
    @staticmethod
    def calculate_avg_uncertainty(results: List[TaskResult]) -> float:
        """Calculate average uncertainty."""
        if not results:
            return 0.0
        return sum(r.uncertainty_avg for r in results) / len(results)


class BenchmarkRunner:
    """Run benchmarks on agent tasks."""
    
    def __init__(self, agent=None):
        """Initialize benchmark runner.
        
        Args:
            agent: The agent to benchmark. If None, uses mock execution.
        """
        self.agent = agent
        self.tasks: List[Task] = []
        self.results: List[TaskResult] = []
        
    def add_task(self, task: Task) -> None:
        """Add a task to the benchmark."""
        self.tasks.append(task)
    
    def add_tasks(self, tasks: List[Task]) -> None:
        """Add multiple tasks."""
        self.tasks.extend(tasks)
    
    def run_task(self, task: Task) -> TaskResult:
        """Run a single task.
        
        Args:
            task: Task to run.
            
        Returns:
            TaskResult from execution.
        """
        start_time = time.time()
        
        if self.agent is None:
            # Mock execution for testing
            return TaskResult(
                task_id=task.id,
                success=True,
                steps_taken=5,
                tool_calls=5,
                execution_time=time.time() - start_time,
                checklist_completion=0.8,
                uncertainty_avg=0.3,
                trajectory=[]
            )
        
        try:
            # Real execution would call agent.run_task()
            result = self.agent.run_task({
                "url": task.url,
                "goal": task.goal,
                "timeout": task.timeout
            })
            
            return TaskResult(
                task_id=task.id,
                success=result.get("success", False),
                steps_taken=result.get("steps", 0),
                tool_calls=result.get("tool_calls", 0),
                execution_time=time.time() - start_time,
                checklist_completion=result.get("checklist_completion", 0.0),
                uncertainty_avg=result.get("uncertainty_avg", 0.0),
                trajectory=result.get("trajectory", [])
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                steps_taken=0,
                tool_calls=0,
                execution_time=time.time() - start_time,
                checklist_completion=0.0,
                uncertainty_avg=0.0,
                error=str(e)
            )
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run all tasks in the benchmark.
        
        Returns:
            BenchmarkResult with aggregated metrics.
        """
        self.results = []
        
        for task in self.tasks:
            result = self.run_task(task)
            self.results.append(result)
        
        return BenchmarkResult(
            total_tasks=len(self.tasks),
            successful_tasks=sum(1 for r in self.results if r.success),
            success_rate=MetricsCalculator.calculate_success_rate(self.results),
            avg_steps=MetricsCalculator.calculate_avg_steps(self.results),
            avg_tool_calls=MetricsCalculator.calculate_avg_tool_calls(self.results),
            avg_execution_time=MetricsCalculator.calculate_avg_execution_time(self.results),
            avg_checklist_completion=MetricsCalculator.calculate_avg_checklist_completion(self.results),
            avg_uncertainty=MetricsCalculator.calculate_avg_uncertainty(self.results),
            task_results=self.results
        )
    
    def get_results(self) -> List[TaskResult]:
        """Get task results."""
        return self.results
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []


class AblationStudyRunner:
    """Run ablation studies comparing agent configurations."""
    
    def __init__(self, base_agent_factory: Callable):
        """Initialize ablation study runner.
        
        Args:
            base_agent_factory: Function that creates agent instances with config.
        """
        self.base_agent_factory = base_agent_factory
        self.configurations: List[Dict[str, Any]] = []
        
    def add_configuration(self, name: str, config: Dict[str, Any]) -> None:
        """Add a configuration to test.
        
        Args:
            name: Name of configuration.
            config: Agent configuration dict.
        """
        self.configurations.append({"name": name, "config": config})
    
    def run_ablation(self, tasks: List[Task]) -> Dict[str, BenchmarkResult]:
        """Run ablation study across configurations.
        
        Args:
            tasks: Tasks to run.
            
        Returns:
            Dict mapping config name to BenchmarkResult.
        """
        results = {}
        
        for cfg in self.configurations:
            agent = self.base_agent_factory(cfg["config"])
            runner = BenchmarkRunner(agent)
            runner.add_tasks(tasks)
            result = runner.run_benchmark()
            results[cfg["name"]] = result
        
        return results
    
    def compare_results(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Compare ablation results.
        
        Args:
            results: Dict of config name to results.
            
        Returns:
            Comparison summary.
        """
        if not results:
            return {}
        
        # Find baseline (first config)
        baseline_name = list(results.keys())[0]
        baseline = results[baseline_name]
        
        comparison = {
            "baseline": baseline_name,
            "configurations": {}
        }
        
        for name, result in results.items():
            comparison["configurations"][name] = {
                "success_rate": result.success_rate,
                "avg_steps": result.avg_steps,
                "avg_tool_calls": result.avg_tool_calls,
                "avg_execution_time": result.avg_execution_time,
                "vs_baseline": {
                    "success_rate_delta": result.success_rate - baseline.success_rate,
                    "steps_delta": result.avg_steps - baseline.avg_steps,
                    "tool_calls_delta": result.avg_tool_calls - baseline.avg_tool_calls,
                } if name != baseline_name else None
            }
        
        return comparison


def create_default_tasks() -> List[Task]:
    """Create default benchmark tasks.
    
    Returns:
        List of default tasks.
    """
    return [
        Task(
            id="nav_1",
            name="Navigate to Wikipedia",
            task_type=TaskType.NAVIGATION,
            url="https://www.google.com",
            goal="Navigate to Wikipedia and verify the page loaded",
            expected_actions=["navigate", "extract"],
            success_criteria={"url_contains": "wikipedia.org"}
        ),
        Task(
            id="search_1",
            name="Search on Google",
            task_type=TaskType.SEARCH,
            url="https://www.google.com",
            goal="Search for 'Python programming language' and verify results",
            expected_actions=["navigate", "type", "click", "extract"],
            success_criteria={"has_results": True}
        ),
        Task(
            id="form_1",
            name="Fill a contact form",
            task_type=TaskType.FORM_FILLING,
            url="https://httpbin.org/forms/post",
            goal="Fill in name and email fields and submit",
            expected_actions=["navigate", "type", "click"],
            success_criteria={"form_submitted": True}
        ),
        Task(
            id="multi_1",
            name="Multi-hop: Find CEO of company",
            task_type=TaskType.MULTI_HOP,
            url="https://www.google.com",
            goal="Find the CEO of a tech company by searching",
            expected_actions=["navigate", "type", "click", "extract", "click", "extract"],
            success_criteria={"found_name": True}
        ),
        Task(
            id="extract_1",
            name="Extract article content",
            task_type=TaskType.EXTRACTION,
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            goal="Extract the first paragraph of the article",
            expected_actions=["navigate", "extract"],
            success_criteria={"has_content": True, "min_length": 50}
        ),
    ]


def save_results(result: BenchmarkResult, filepath: str) -> None:
    """Save benchmark results to JSON file.
    
    Args:
        result: BenchmarkResult to save.
        filepath: Path to save to.
    """
    data = {
        "total_tasks": result.total_tasks,
        "successful_tasks": result.successful_tasks,
        "success_rate": result.success_rate,
        "avg_steps": result.avg_steps,
        "avg_tool_calls": result.avg_tool_calls,
        "avg_execution_time": result.avg_execution_time,
        "avg_checklist_completion": result.avg_checklist_completion,
        "avg_uncertainty": result.avg_uncertainty,
        "timestamp": result.timestamp,
        "task_results": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "steps_taken": r.steps_taken,
                "tool_calls": r.tool_calls,
                "execution_time": r.execution_time,
                "checklist_completion": r.checklist_completion,
                "uncertainty_avg": r.uncertainty_avg,
                "error": r.error,
            }
            for r in result.task_results
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
