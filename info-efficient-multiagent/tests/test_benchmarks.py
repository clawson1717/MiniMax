"""Tests for benchmark tasks module."""

import pytest
from src.benchmarks import (
    BenchmarkTask,
    BenchmarkType,
    Difficulty,
    EvaluationResult,
    BenchmarkResults,
    ExactMatchEvaluator,
    NumericEvaluator,
    SemanticEvaluator,
    TaskDataset,
    ReasoningBenchmark,
    QABenchmark,
    MathBenchmark,
    MultiAgentBenchmarkRunner,
)


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass."""

    def test_create_task(self):
        """Test creating a basic benchmark task."""
        task = BenchmarkTask(
            task_id="test_001",
            prompt="What is 2 + 2?",
            expected_answer="4",
            benchmark_type=BenchmarkType.MATH,
        )
        assert task.task_id == "test_001"
        assert task.prompt == "What is 2 + 2?"
        assert task.expected_answer == "4"
        assert task.benchmark_type == BenchmarkType.MATH
        assert task.difficulty == Difficulty.MEDIUM

    def test_task_to_dict(self):
        """Test serializing task to dictionary."""
        task = BenchmarkTask(
            task_id="test_002",
            prompt="Test question",
            expected_answer="yes",
            benchmark_type=BenchmarkType.REASONING,
            difficulty=Difficulty.HARD,
            category="logic",
        )
        d = task.to_dict()
        assert d["task_id"] == "test_002"
        assert d["benchmark_type"] == "reasoning"
        assert d["difficulty"] == 3
        assert d["category"] == "logic"

    def test_task_from_dict(self):
        """Test deserializing task from dictionary."""
        data = {
            "task_id": "test_003",
            "prompt": "Test prompt",
            "expected_answer": "answer",
            "benchmark_type": "qa",
            "difficulty": 1,
            "category": "test",
        }
        task = BenchmarkTask.from_dict(data)
        assert task.task_id == "test_003"
        assert task.benchmark_type == BenchmarkType.QA
        assert task.difficulty == Difficulty.EASY


class TestExactMatchEvaluator:
    """Tests for exact match evaluator."""

    def test_exact_match_correct(self):
        """Test correct exact match."""
        evaluator = ExactMatchEvaluator()
        task = BenchmarkTask(
            task_id="test_exact",
            prompt="What is the capital of France?",
            expected_answer="Paris",
            benchmark_type=BenchmarkType.QA,
        )
        result = evaluator.evaluate(task, "Paris", "agent1")
        assert result.is_correct
        assert result.score == 1.0

    def test_exact_match_incorrect(self):
        """Test incorrect exact match."""
        evaluator = ExactMatchEvaluator()
        task = BenchmarkTask(
            task_id="test_exact_wrong",
            prompt="What is the capital of France?",
            expected_answer="Paris",
            benchmark_type=BenchmarkType.QA,
        )
        result = evaluator.evaluate(task, "London", "agent1")
        assert not result.is_correct
        assert result.score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching."""
        evaluator = ExactMatchEvaluator()
        task = BenchmarkTask(
            task_id="test_case",
            prompt="What is the capital of France?",
            expected_answer="Paris",
            benchmark_type=BenchmarkType.QA,
        )
        result = evaluator.evaluate(task, "paris", "agent1")
        assert result.is_correct

    def test_extract_mc_answer(self):
        """Test multiple choice answer extraction."""
        evaluator = ExactMatchEvaluator()
        task = BenchmarkTask(
            task_id="test_mc",
            prompt="Pick one:",
            expected_answer="B",
            benchmark_type=BenchmarkType.QA,
            options=["Option A", "Option B", "Option C"],
        )
        result = evaluator.evaluate(task, "The answer is B", "agent1")
        assert result.is_correct


class TestNumericEvaluator:
    """Tests for numeric evaluator."""

    def test_numeric_exact_match(self):
        """Test exact numeric match."""
        evaluator = NumericEvaluator()
        task = BenchmarkTask(
            task_id="test_num",
            prompt="What is 15 + 27?",
            expected_answer=42,
            benchmark_type=BenchmarkType.MATH,
        )
        result = evaluator.evaluate(task, "The answer is 42", "agent1")
        assert result.is_correct

    def test_numeric_tolerance(self):
        """Test numeric comparison with tolerance."""
        evaluator = NumericEvaluator()
        task = BenchmarkTask(
            task_id="test_num_tol",
            prompt="What is the value?",
            expected_answer=100.0,
            benchmark_type=BenchmarkType.MATH,
        )
        result = evaluator.evaluate(task, "The result is 99.5", "agent1")
        assert result.is_correct  # Within 1% tolerance

    def test_numeric_extract_from_response(self):
        """Test extracting numeric answer from response."""
        evaluator = NumericEvaluator()
        task = BenchmarkTask(
            task_id="test_num_extract",
            prompt="Calculate:",
            expected_answer=96,
            benchmark_type=BenchmarkType.MATH,
        )
        extracted = evaluator.extract_answer("The answer equals 96", task)
        assert extracted == 96.0


class TestSemanticEvaluator:
    """Tests for semantic evaluator."""

    def test_semantic_yes_match(self):
        """Test semantic matching for yes answers."""
        evaluator = SemanticEvaluator()
        task = BenchmarkTask(
            task_id="test_sem_yes",
            prompt="Is this valid?",
            expected_answer="yes",
            benchmark_type=BenchmarkType.REASONING,
        )
        result = evaluator.evaluate(task, "Yes, this is correct", "agent1")
        assert result.is_correct

    def test_semantic_no_match(self):
        """Test semantic matching for no answers."""
        evaluator = SemanticEvaluator()
        task = BenchmarkTask(
            task_id="test_sem_no",
            prompt="Is this valid?",
            expected_answer="no",
            benchmark_type=BenchmarkType.REASONING,
        )
        result = evaluator.evaluate(task, "No, this is incorrect", "agent1")
        assert result.is_correct

    def test_semantic_partial_score(self):
        """Test partial score for semantic overlap."""
        evaluator = SemanticEvaluator()
        task = BenchmarkTask(
            task_id="test_sem_partial",
            prompt="Complete the analogy:",
            expected_answer="dark",
            benchmark_type=BenchmarkType.REASONING,
        )
        result = evaluator.evaluate(task, "The answer is darkness", "agent1")
        assert result.score > 0


class TestTaskDataset:
    """Tests for TaskDataset class."""

    def test_create_dataset(self):
        """Test creating a task dataset."""
        tasks = [
            BenchmarkTask("t1", "Q1", "A1", BenchmarkType.QA),
            BenchmarkTask("t2", "Q2", "A2", BenchmarkType.QA),
        ]
        dataset = TaskDataset(tasks)
        assert len(dataset) == 2

    def test_filter_by_type(self):
        """Test filtering by benchmark type."""
        tasks = [
            BenchmarkTask("t1", "Q1", "A1", BenchmarkType.QA),
            BenchmarkTask("t2", "Q2", "A2", BenchmarkType.MATH),
            BenchmarkTask("t3", "Q3", "A3", BenchmarkType.QA),
        ]
        dataset = TaskDataset(tasks)
        filtered = dataset.filter_by_type(BenchmarkType.QA)
        assert len(filtered) == 2

    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        tasks = [
            BenchmarkTask("t1", "Q1", "A1", BenchmarkType.QA, Difficulty.EASY),
            BenchmarkTask("t2", "Q2", "A2", BenchmarkType.QA, Difficulty.HARD),
        ]
        dataset = TaskDataset(tasks)
        filtered = dataset.filter_by_difficulty(Difficulty.EASY)
        assert len(filtered) == 1

    def test_sample(self):
        """Test sampling from dataset."""
        tasks = [BenchmarkTask(f"t{i}", f"Q{i}", f"A{i}", BenchmarkType.QA) for i in range(10)]
        dataset = TaskDataset(tasks)
        sampled = dataset.sample(5, seed=42)
        assert len(sampled) == 5

    def test_get_task(self):
        """Test getting task by ID."""
        tasks = [BenchmarkTask("t1", "Q1", "A1", BenchmarkType.QA)]
        dataset = TaskDataset(tasks)
        task = dataset.get_task("t1")
        assert task is not None
        assert task.task_id == "t1"

    def test_iteration(self):
        """Test iterating over dataset."""
        tasks = [
            BenchmarkTask("t1", "Q1", "A1", BenchmarkType.QA),
            BenchmarkTask("t2", "Q2", "A2", BenchmarkType.QA),
        ]
        dataset = TaskDataset(tasks)
        ids = [t.task_id for t in dataset]
        assert ids == ["t1", "t2"]


class TestReasoningBenchmark:
    """Tests for ReasoningBenchmark class."""

    def test_default_tasks(self):
        """Test default reasoning tasks are loaded."""
        benchmark = ReasoningBenchmark()
        assert len(benchmark.tasks) == 10
        assert all(t.benchmark_type == BenchmarkType.REASONING for t in benchmark.tasks)

    def test_get_dataset(self):
        """Test getting task dataset."""
        benchmark = ReasoningBenchmark()
        dataset = benchmark.get_dataset()
        assert len(dataset) == 10

    def test_evaluator_type(self):
        """Test evaluator is semantic evaluator."""
        benchmark = ReasoningBenchmark()
        assert isinstance(benchmark.evaluator, SemanticEvaluator)


class TestQABenchmark:
    """Tests for QABenchmark class."""

    def test_default_tasks(self):
        """Test default QA tasks are loaded."""
        benchmark = QABenchmark()
        assert len(benchmark.tasks) == 10
        assert all(t.benchmark_type == BenchmarkType.QA for t in benchmark.tasks)

    def test_get_dataset(self):
        """Test getting task dataset."""
        benchmark = QABenchmark()
        dataset = benchmark.get_dataset()
        assert len(dataset) == 10


class TestMathBenchmark:
    """Tests for MathBenchmark class."""

    def test_default_tasks(self):
        """Test default math tasks are loaded."""
        benchmark = MathBenchmark()
        assert len(benchmark.tasks) == 10
        assert all(t.benchmark_type == BenchmarkType.MATH for t in benchmark.tasks)

    def test_numeric_evaluator(self):
        """Test evaluator is numeric evaluator."""
        benchmark = MathBenchmark()
        assert isinstance(benchmark.evaluator, NumericEvaluator)


class TestMultiAgentBenchmarkRunner:
    """Tests for MultiAgentBenchmarkRunner class."""

    def test_get_all_tasks(self):
        """Test getting all tasks combined."""
        runner = MultiAgentBenchmarkRunner()
        dataset = runner.get_all_tasks()
        assert len(dataset) == 30  # 10 reasoning + 10 QA + 10 math

    def test_get_evaluator_by_type(self):
        """Test getting correct evaluator for task type."""
        runner = MultiAgentBenchmarkRunner()

        math_task = BenchmarkTask("m1", "Q", 1, BenchmarkType.MATH)
        qa_task = BenchmarkTask("q1", "Q", "A", BenchmarkType.QA)
        reason_task = BenchmarkTask("r1", "Q", "yes", BenchmarkType.REASONING)

        assert isinstance(runner.get_evaluator(math_task), NumericEvaluator)
        assert isinstance(runner.get_evaluator(qa_task), ExactMatchEvaluator)
        assert isinstance(runner.get_evaluator(reason_task), SemanticEvaluator)

    def test_evaluate_response(self):
        """Test evaluating a single response."""
        runner = MultiAgentBenchmarkRunner()
        task = BenchmarkTask("t1", "What is 2 + 2?", 4, BenchmarkType.MATH)
        result = runner.evaluate_response(task, "The answer is 4", "agent1")
        assert result.is_correct

    def test_run_benchmark(self):
        """Test running a full benchmark."""
        runner = MultiAgentBenchmarkRunner()
        tasks = TaskDataset([
            BenchmarkTask("t1", "Q1", "A1", BenchmarkType.QA),
            BenchmarkTask("t2", "Q2", "A2", BenchmarkType.QA),
        ])

        def mock_agent(prompt):
            if "Q1" in prompt:
                return "A1"
            return "Wrong"

        results = runner.run_benchmark(tasks, mock_agent, "test_agent")
        assert results.total_tasks == 2
        assert results.correct == 1
        assert results.accuracy == 0.5


class TestBenchmarkResults:
    """Tests for BenchmarkResults dataclass."""

    def test_results_creation(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            benchmark_name="test",
            total_tasks=10,
            correct=8,
            total_score=8.0,
            accuracy=0.8,
            avg_response_time_ms=100.0,
            total_tokens_used=1000,
            total_capacity_used=50.0,
        )
        assert results.benchmark_name == "test"
        assert results.accuracy == 0.8

    def test_results_to_dict(self):
        """Test serializing results to dictionary."""
        results = BenchmarkResults(
            benchmark_name="test",
            total_tasks=5,
            correct=3,
            total_score=3.0,
            accuracy=0.6,
            avg_response_time_ms=50.0,
            total_tokens_used=500,
            total_capacity_used=25.0,
        )
        d = results.to_dict()
        assert d["benchmark_name"] == "test"
        assert d["accuracy"] == 0.6


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            task_id="test_task",
            agent_id="agent1",
            predicted_answer="Paris",
            expected_answer="Paris",
            is_correct=True,
            score=1.0,
        )
        assert result.is_correct
        assert result.score == 1.0

    def test_result_to_dict(self):
        """Test serializing result to dictionary."""
        result = EvaluationResult(
            task_id="t1",
            agent_id="a1",
            predicted_answer="yes",
            expected_answer="yes",
            is_correct=True,
            score=1.0,
            response_time_ms=150.0,
        )
        d = result.to_dict()
        assert d["task_id"] == "t1"
        assert d["is_correct"] is True
