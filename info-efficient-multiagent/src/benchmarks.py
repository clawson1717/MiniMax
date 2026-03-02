"""
Benchmark Tasks for Multi-Agent Reasoning Systems

Provides standardized benchmark tasks to evaluate multi-agent reasoning systems
against baselines. Includes reasoning, question-answering, and math tasks.
"""

import random
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json


class BenchmarkType(Enum):
    """Types of benchmark tasks."""
    REASONING = "reasoning"
    QA = "qa"
    MATH = "math"
    MIXED = "mixed"


class Difficulty(Enum):
    """Task difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class BenchmarkTask:
    """A single benchmark task with prompt, expected answer, and metadata."""
    task_id: str
    prompt: str
    expected_answer: Any
    benchmark_type: BenchmarkType
    difficulty: Difficulty = Difficulty.MEDIUM
    category: str = "general"
    options: Optional[List[str]] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "benchmark_type": self.benchmark_type.value,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "options": self.options,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            expected_answer=data["expected_answer"],
            benchmark_type=BenchmarkType(data["benchmark_type"]),
            difficulty=Difficulty(data.get("difficulty", 2)),
            category=data.get("category", "general"),
            options=data.get("options"),
            explanation=data.get("explanation"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationResult:
    """Result of evaluating an agent's response to a benchmark task."""
    task_id: str
    agent_id: str
    predicted_answer: Any
    expected_answer: Any
    is_correct: bool
    score: float
    confidence: float = 0.0
    response_time_ms: float = 0.0
    tokens_used: int = 0
    capacity_used: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "predicted_answer": self.predicted_answer,
            "expected_answer": self.expected_answer,
            "is_correct": self.is_correct,
            "score": self.score,
            "confidence": self.confidence,
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "capacity_used": self.capacity_used,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResults:
    """Aggregated results for a benchmark run."""
    benchmark_name: str
    total_tasks: int
    correct: int
    total_score: float
    accuracy: float
    avg_response_time_ms: float
    total_tokens_used: int
    total_capacity_used: float
    results_by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    results_by_difficulty: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_task_results: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "total_tasks": self.total_tasks,
            "correct": self.correct,
            "total_score": self.total_score,
            "accuracy": self.accuracy,
            "avg_response_time_ms": self.avg_response_time_ms,
            "total_tokens_used": self.total_tokens_used,
            "total_capacity_used": self.total_capacity_used,
            "results_by_type": self.results_by_type,
            "results_by_difficulty": self.results_by_difficulty,
            "per_task_results": [r.to_dict() for r in self.per_task_results],
            "metadata": self.metadata,
        }


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""

    @abstractmethod
    def evaluate(self, task: BenchmarkTask, response: str, agent_id: str) -> EvaluationResult:
        pass

    @abstractmethod
    def extract_answer(self, response: str, task: BenchmarkTask) -> Any:
        pass


class ExactMatchEvaluator(BenchmarkEvaluator):
    """Evaluator for exact match answers."""

    def evaluate(self, task: BenchmarkTask, response: str, agent_id: str) -> EvaluationResult:
        predicted = self.extract_answer(response, task)
        expected = self.normalize_answer(task.expected_answer)
        # Normalize predicted for comparison
        predicted_normalized = self.normalize_answer(predicted) if predicted else ""
        is_correct = predicted_normalized == expected
        score = 1.0 if is_correct else 0.0
        return EvaluationResult(
            task_id=task.task_id,
            agent_id=agent_id,
            predicted_answer=predicted,
            expected_answer=task.expected_answer,
            is_correct=is_correct,
            score=score,
        )

    def extract_answer(self, response: str, task: BenchmarkTask) -> Any:
        if task.options:
            return self._extract_mc_answer(response, task.options)
        return self.normalize_answer(response)

    def _extract_mc_answer(self, response: str, options: List[str]) -> str:
        response_upper = response.upper()
        for letter in ["A", "B", "C", "D", "E", "F"]:
            patterns = [rf"\b{letter}\b", rf"\b{letter}[.)]", rf"option {letter}", rf"answer is {letter}"]
            for pattern in patterns:
                if re.search(pattern, response_upper):
                    return letter
        for i, option in enumerate(options):
            if option.lower() in response.lower():
                return chr(65 + i)
        return ""

    def normalize_answer(self, answer: Any) -> str:
        if answer is None:
            return ""
        answer_str = str(answer).strip().lower()
        # Don't strip too much for single letters/short answers
        if len(answer_str) <= 2:
            return answer_str
        answer_str = re.sub(r'[^\w\s-]', '', answer_str)
        return ' '.join(answer_str.split())


class NumericEvaluator(BenchmarkEvaluator):
    """Evaluator for numeric/math answers."""

    def evaluate(self, task: BenchmarkTask, response: str, agent_id: str) -> EvaluationResult:
        predicted = self.extract_answer(response, task)
        expected = task.expected_answer
        is_correct = False
        score = 0.0

        if predicted is not None:
            try:
                pred_num = float(predicted)
                exp_num = float(expected)
                if exp_num != 0:
                    rel_error = abs(pred_num - exp_num) / abs(exp_num)
                    is_correct = rel_error < 0.01
                    score = max(0.0, 1.0 - rel_error)
                else:
                    is_correct = abs(pred_num) < 0.001
                    score = 1.0 if is_correct else 0.0
            except (ValueError, TypeError):
                is_correct = str(predicted).strip() == str(expected).strip()
                score = 1.0 if is_correct else 0.0

        return EvaluationResult(
            task_id=task.task_id,
            agent_id=agent_id,
            predicted_answer=predicted,
            expected_answer=expected,
            is_correct=is_correct,
            score=score,
        )

    def extract_answer(self, response: str, task: BenchmarkTask) -> Optional[float]:
        patterns = [r"answer[:\s]+(-?[\d.]+)", r"result[:\s]+(-?[\d.]+)", r"equals?[:\s]+(-?[\d.]+)", r"=\s*(-?[\d.]+)"]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None


class SemanticEvaluator(BenchmarkEvaluator):
    """Evaluator for semantic/reasoning answers."""

    def evaluate(self, task: BenchmarkTask, response: str, agent_id: str) -> EvaluationResult:
        predicted = self.extract_answer(response, task)
        expected = str(task.expected_answer).lower()
        is_correct = self._check_semantic_match(predicted, expected)
        score = self._calculate_semantic_score(predicted, expected)
        return EvaluationResult(
            task_id=task.task_id,
            agent_id=agent_id,
            predicted_answer=predicted,
            expected_answer=task.expected_answer,
            is_correct=is_correct,
            score=score,
        )

    def extract_answer(self, response: str, task: BenchmarkTask) -> str:
        markers = ["answer:", "conclusion:", "therefore:", "thus:"]
        response_lower = response.lower()
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.index(marker)
                return response[idx + len(marker):].strip()
        sentences = response.split(". ")
        return sentences[-1].strip() if sentences else response.strip()

    def _check_semantic_match(self, predicted: str, expected: str) -> bool:
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        if exp_lower in pred_lower:
            return True
        if expected in ["yes", "true", "correct"]:
            return any(kw in pred_lower for kw in ["correct", "true", "yes", "valid"])
        if expected in ["no", "false", "incorrect"]:
            return any(kw in pred_lower for kw in ["incorrect", "false", "no", "invalid"])
        return False

    def _calculate_semantic_score(self, predicted: str, expected: str) -> float:
        pred_words = set(predicted.lower().split())
        exp_words = set(expected.lower().split())
        if not exp_words:
            return 0.0
        overlap = len(pred_words & exp_words)
        base_score = min(1.0, overlap / len(exp_words))
        # Bonus for substring match
        if expected.lower() in predicted.lower():
            return max(base_score, 0.5)
        return base_score


class TaskDataset:
    """Collection of benchmark tasks with filtering and sampling utilities."""

    def __init__(self, tasks: Optional[List[BenchmarkTask]] = None):
        self.tasks: List[BenchmarkTask] = tasks or []
        self._task_index: Dict[str, BenchmarkTask] = {}
        self._build_index()

    def _build_index(self) -> None:
        self._task_index = {t.task_id: t for t in self.tasks}

    def add_task(self, task: BenchmarkTask) -> None:
        self.tasks.append(task)
        self._task_index[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[BenchmarkTask]:
        return self._task_index.get(task_id)

    def filter_by_type(self, benchmark_type: BenchmarkType) -> "TaskDataset":
        return TaskDataset([t for t in self.tasks if t.benchmark_type == benchmark_type])

    def filter_by_difficulty(self, difficulty: Difficulty) -> "TaskDataset":
        return TaskDataset([t for t in self.tasks if t.difficulty == difficulty])

    def sample(self, n: int, seed: Optional[int] = None) -> "TaskDataset":
        if seed is not None:
            random.seed(seed)
        n = min(n, len(self.tasks))
        return TaskDataset(random.sample(self.tasks, n))

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, idx: int) -> BenchmarkTask:
        return self.tasks[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {"tasks": [t.to_dict() for t in self.tasks]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDataset":
        return cls([BenchmarkTask.from_dict(t) for t in data["tasks"]])


# =============================================================================
# Specific Benchmark Classes
# =============================================================================

class ReasoningBenchmark:
    """Benchmark for logical reasoning tasks: syllogisms, analogies, logic puzzles."""

    DEFAULT_TASKS = [
        BenchmarkTask("reason_001", "All cats are mammals. All mammals are animals. Therefore, all cats are animals. Is this reasoning valid?", "yes", BenchmarkType.REASONING, Difficulty.EASY, "syllogism"),
        BenchmarkTask("reason_002", "All birds can fly. Penguins are birds. Therefore, penguins can fly. Is this reasoning valid?", "no", BenchmarkType.REASONING, Difficulty.EASY, "syllogism"),
        BenchmarkTask("reason_003", "If it rains, the ground gets wet. The ground is wet. Therefore, it rained. Is this reasoning valid?", "no", BenchmarkType.REASONING, Difficulty.MEDIUM, "conditional"),
        BenchmarkTask("reason_004", "If you study hard, you will pass the test. You did not pass the test. Therefore, you did not study hard. Is this reasoning valid?", "yes", BenchmarkType.REASONING, Difficulty.MEDIUM, "conditional"),
        BenchmarkTask("reason_005", "Hot is to cold as light is to what?", "dark", BenchmarkType.REASONING, Difficulty.EASY, "analogy"),
        BenchmarkTask("reason_006", "Doctor is to hospital as teacher is to what?", "school", BenchmarkType.REASONING, Difficulty.EASY, "analogy"),
        BenchmarkTask("reason_007", "Book is to reading as fork is to what?", "eating", BenchmarkType.REASONING, Difficulty.EASY, "analogy"),
        BenchmarkTask("reason_008", "Planet is to solar system as electron is to what?", "atom", BenchmarkType.REASONING, Difficulty.MEDIUM, "analogy"),
        BenchmarkTask("reason_009", "Alice is taller than Bob. Bob is taller than Carol. Is Alice taller than Carol?", "yes", BenchmarkType.REASONING, Difficulty.EASY, "transitive"),
        BenchmarkTask("reason_010", "If all A are B, and some B are C, can we conclude that some A are C?", "no", BenchmarkType.REASONING, Difficulty.HARD, "syllogism"),
    ]

    def __init__(self, tasks: Optional[List[BenchmarkTask]] = None):
        self.tasks = tasks or self.DEFAULT_TASKS.copy()
        self.evaluator = SemanticEvaluator()

    def get_dataset(self) -> TaskDataset:
        return TaskDataset(self.tasks)


class QABenchmark:
    """Benchmark for question-answering tasks."""

    DEFAULT_TASKS = [
        BenchmarkTask("qa_001", "What is the capital of France?", "Paris", BenchmarkType.QA, Difficulty.EASY, "geography"),
        BenchmarkTask("qa_002", "Who wrote 'Romeo and Juliet'?", "Shakespeare", BenchmarkType.QA, Difficulty.EASY, "literature"),
        BenchmarkTask("qa_003", "What is the largest planet in our solar system?", "Jupiter", BenchmarkType.QA, Difficulty.EASY, "science"),
        BenchmarkTask("qa_004", "In what year did World War II end?", "1945", BenchmarkType.QA, Difficulty.EASY, "history"),
        BenchmarkTask("qa_005", "What is the chemical symbol for gold?", "Au", BenchmarkType.QA, Difficulty.EASY, "science"),
        BenchmarkTask("qa_006", "What is the tallest mountain in the world?", "Mount Everest", BenchmarkType.QA, Difficulty.EASY, "geography"),
        BenchmarkTask("qa_007", "Who painted the Mona Lisa?", "Leonardo da Vinci", BenchmarkType.QA, Difficulty.EASY, "art"),
        BenchmarkTask("qa_008", "What is the speed of light in a vacuum (approximately)?", "299792458", BenchmarkType.QA, Difficulty.MEDIUM, "physics"),
        BenchmarkTask("qa_009", "What is the capital of Australia?", "Canberra", BenchmarkType.QA, Difficulty.MEDIUM, "geography"),
        BenchmarkTask("qa_010", "Who developed the theory of general relativity?", "Einstein", BenchmarkType.QA, Difficulty.EASY, "science"),
    ]

    def __init__(self, tasks: Optional[List[BenchmarkTask]] = None):
        self.tasks = tasks or self.DEFAULT_TASKS.copy()
        self.evaluator = ExactMatchEvaluator()

    def get_dataset(self) -> TaskDataset:
        return TaskDataset(self.tasks)


class MathBenchmark:
    """Benchmark for math tasks: arithmetic, algebra."""

    DEFAULT_TASKS = [
        BenchmarkTask("math_001", "What is 15 + 27?", 42, BenchmarkType.MATH, Difficulty.EASY, "arithmetic"),
        BenchmarkTask("math_002", "What is 100 - 37?", 63, BenchmarkType.MATH, Difficulty.EASY, "arithmetic"),
        BenchmarkTask("math_003", "What is 12 × 8?", 96, BenchmarkType.MATH, Difficulty.EASY, "arithmetic"),
        BenchmarkTask("math_004", "What is 144 ÷ 12?", 12, BenchmarkType.MATH, Difficulty.EASY, "arithmetic"),
        BenchmarkTask("math_005", "What is 7²?", 49, BenchmarkType.MATH, Difficulty.EASY, "arithmetic"),
        BenchmarkTask("math_006", "What is the square root of 81?", 9, BenchmarkType.MATH, Difficulty.EASY, "arithmetic"),
        BenchmarkTask("math_007", "If x + 5 = 12, what is x?", 7, BenchmarkType.MATH, Difficulty.EASY, "algebra"),
        BenchmarkTask("math_008", "If 3x = 21, what is x?", 7, BenchmarkType.MATH, Difficulty.EASY, "algebra"),
        BenchmarkTask("math_009", "What is 15% of 200?", 30, BenchmarkType.MATH, Difficulty.MEDIUM, "arithmetic"),
        BenchmarkTask("math_010", "If 2x + 3 = 11, what is x?", 4, BenchmarkType.MATH, Difficulty.MEDIUM, "algebra"),
    ]

    def __init__(self, tasks: Optional[List[BenchmarkTask]] = None):
        self.tasks = tasks or self.DEFAULT_TASKS.copy()
        self.evaluator = NumericEvaluator()

    def get_dataset(self) -> TaskDataset:
        return TaskDataset(self.tasks)


class MultiAgentBenchmarkRunner:
    """Runner for evaluating multi-agent systems on benchmarks."""

    def __init__(self):
        self.reasoning = ReasoningBenchmark()
        self.qa = QABenchmark()
        self.math = MathBenchmark()

    def get_all_tasks(self) -> TaskDataset:
        all_tasks = self.reasoning.tasks + self.qa.tasks + self.math.tasks
        return TaskDataset(all_tasks)

    def get_evaluator(self, task: BenchmarkTask) -> BenchmarkEvaluator:
        if task.benchmark_type == BenchmarkType.MATH:
            return NumericEvaluator()
        elif task.benchmark_type == BenchmarkType.QA:
            return ExactMatchEvaluator()
        else:
            return SemanticEvaluator()

    def evaluate_response(self, task: BenchmarkTask, response: str, agent_id: str) -> EvaluationResult:
        evaluator = self.get_evaluator(task)
        return evaluator.evaluate(task, response, agent_id)

    def run_benchmark(
        self,
        tasks: TaskDataset,
        agent_response_fn,
        agent_id: str = "agent",
    ) -> BenchmarkResults:
        """Run benchmark with a callable that takes a prompt and returns a response."""
        results = []
        total_score = 0.0
        total_time = 0.0
        total_tokens = 0

        for task in tasks:
            import time
            start = time.time()
            response = agent_response_fn(task.prompt)
            elapsed_ms = (time.time() - start) * 1000

            result = self.evaluate_response(task, response, agent_id)
            result.response_time_ms = elapsed_ms
            results.append(result)
            total_score += result.score
            total_time += elapsed_ms

        correct = sum(1 for r in results if r.is_correct)
        avg_time = total_time / len(results) if results else 0

        return BenchmarkResults(
            benchmark_name="multi_agent_benchmark",
            total_tasks=len(tasks),
            correct=correct,
            total_score=total_score,
            accuracy=correct / len(tasks) if tasks else 0,
            avg_response_time_ms=avg_time,
            total_tokens_used=total_tokens,
            total_capacity_used=0.0,
            per_task_results=results,
        )
