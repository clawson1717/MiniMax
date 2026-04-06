"""Experiments & Evaluation — benchmark harness, synthetic data generators, and reporting.

Provides:
  - BenchmarkTask, EvalResult, ExperimentConfig — data structures for evaluation
  - BaselineMode enum: naive, self_consistency, discouq_only, raptor_full
  - run_experiment() — evaluation loop over tasks with pluggable baseline modes
  - compute_metrics() — accuracy, ECE, avg rerolls, avg steps, total cost
  - sweep_configs() — generate configs for threshold/weight sweeps
  - ExperimentReport — aggregated results with markdown/JSON comparison table
  - generate_gsm8k_synthetic(), generate_math_synthetic(), generate_hotpotqa_synthetic()
    — synthetic benchmark data generators for testing without real datasets

Usage::

    from raptor.experiments import (
        generate_gsm8k_synthetic, run_experiment, compute_metrics,
        ExperimentConfig, ExperimentReport, BaselineMode,
    )

    tasks = generate_gsm8k_synthetic(50)
    agents = [MockReasoningAgent(...) for _ in range(5)]
    config = ExperimentConfig(baseline_mode=BaselineMode.RAPTOR_FULL)
    results = run_experiment(config, tasks, agents)
    metrics = compute_metrics(results)
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from math import comb
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from raptor.agents import ReasoningAgent, poll_agents, reroll as agent_reroll
from raptor.config import Config, OrchestrationAction
from raptor.disagreement import AgentResponse, DisagreementMonitor
from raptor.integration import run_with_raptor
from raptor.orchestrator import RAPTOROrchestrator


# ══════════════════════════════════════════════════════════════════════
# Enums & Data Structures
# ══════════════════════════════════════════════════════════════════════


class BaselineMode(Enum):
    """Evaluation baseline modes for comparison."""

    NAIVE = "naive"  # Single agent, no RAPTOR
    SELF_CONSISTENCY = "self_consistency"  # Majority vote over N agents
    DISCOUQ_ONLY = "discouq_only"  # Disagreement monitoring, no entropy/utility
    RAPTOR_FULL = "raptor_full"  # Full RAPTOR pipeline


@dataclass
class BenchmarkTask:
    """A single benchmark evaluation task.

    Attributes:
        question: The question or prompt.
        ground_truth: The known correct answer.
        dataset_name: Which benchmark this task belongs to.
        task_id: Unique identifier for this task.
        metadata: Arbitrary extra data (difficulty, category, etc.).
    """

    question: str
    ground_truth: str
    dataset_name: str
    task_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single benchmark task.

    Attributes:
        task_id: Identifier linking back to the BenchmarkTask.
        predicted: The model's predicted answer.
        correct: The ground truth answer.
        is_correct: Whether predicted matches ground truth.
        signals_history: Signal vectors from RAPTOR orchestration steps.
        n_rerolls: Number of reroll actions taken.
        n_steps: Number of orchestration steps.
        cost_estimate: Estimated cost in arbitrary units.
        confidence: Calibrated confidence score for this prediction.
    """

    task_id: str
    predicted: str
    correct: str
    is_correct: bool
    signals_history: list[dict[str, Any]] = field(default_factory=list)
    n_rerolls: int = 0
    n_steps: int = 1
    cost_estimate: float = 0.0
    confidence: float = 0.5


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes:
        raptor_config: RAPTOR system configuration.
        baseline_mode: Which baseline to run.
        n_agents: Number of agents to poll.
        dataset: Dataset name (for labeling).
        n_samples: Max tasks to evaluate (0 = all).
        label: Human-readable label for this config.
    """

    raptor_config: Config = field(default_factory=Config)
    baseline_mode: BaselineMode = BaselineMode.RAPTOR_FULL
    n_agents: int = 5
    dataset: str = "gsm8k"
    n_samples: int = 100
    label: str = ""


# ══════════════════════════════════════════════════════════════════════
# Answer Extraction & Comparison
# ══════════════════════════════════════════════════════════════════════


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, lowercases, removes common answer prefixes
    (``"the answer is"``, ``"answer:"``, etc.) and trailing punctuation.
    """
    answer = answer.strip().lower()
    for prefix in ("the answer is ", "answer: ", "answer:", "result: ", "result:", "= "):
        if answer.startswith(prefix):
            answer = answer[len(prefix) :].strip()
    answer = answer.rstrip(".")
    return answer


def _answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if *predicted* matches *ground_truth* (flexible comparison).

    Tries exact match, numeric equality (within 1e-6), and substring containment.
    """
    norm_pred = _normalize_answer(predicted)
    norm_gt = _normalize_answer(ground_truth)

    if norm_pred == norm_gt:
        return True

    # Numeric comparison
    try:
        if abs(float(norm_pred) - float(norm_gt)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Substring containment (ground truth in prediction)
    if norm_gt and norm_gt in norm_pred:
        return True

    return False


def _majority_vote(answers: list[str]) -> str:
    """Return the most common answer via majority vote."""
    if not answers:
        return ""
    normalized = [_normalize_answer(a) for a in answers]
    counter = Counter(normalized)
    winner = counter.most_common(1)[0][0]
    # Return the original (non-normalized) answer matching the winner
    for ans, norm in zip(answers, normalized):
        if norm == winner:
            return ans
    return answers[0]


# ══════════════════════════════════════════════════════════════════════
# Cost Estimation
# ══════════════════════════════════════════════════════════════════════

# Rough token cost estimates per action (arbitrary units)
_ACTION_COST = {
    "poll": 1.0,
    "reroll": 0.5,
    "verify": 0.3,
    "retrieve": 0.1,
    "escalate": 2.0,
}


def _estimate_cost(
    n_agents: int, n_steps: int, n_rerolls: int, mode: BaselineMode
) -> float:
    """Estimate cost based on mode and actions taken."""
    poll_cost = n_agents * _ACTION_COST["poll"]

    if mode == BaselineMode.NAIVE:
        return _ACTION_COST["poll"]
    elif mode == BaselineMode.SELF_CONSISTENCY:
        return poll_cost
    elif mode == BaselineMode.DISCOUQ_ONLY:
        return poll_cost + n_rerolls * _ACTION_COST["reroll"]
    else:  # RAPTOR_FULL
        return poll_cost * n_steps + n_rerolls * n_agents * _ACTION_COST["reroll"]


# ══════════════════════════════════════════════════════════════════════
# Baseline Runners
# ══════════════════════════════════════════════════════════════════════


def _run_naive(
    task: BenchmarkTask, agents: list[ReasoningAgent], config: Config
) -> EvalResult:
    """Naive baseline: single agent, direct answer, no RAPTOR."""
    response = agents[0].generate(task.question)
    predicted = response.final_answer
    return EvalResult(
        task_id=task.task_id,
        predicted=predicted,
        correct=task.ground_truth,
        is_correct=_answers_match(predicted, task.ground_truth),
        n_rerolls=0,
        n_steps=1,
        cost_estimate=_estimate_cost(1, 1, 0, BaselineMode.NAIVE),
        confidence=0.5,
    )


def _run_self_consistency(
    task: BenchmarkTask, agents: list[ReasoningAgent], config: Config
) -> EvalResult:
    """Self-consistency baseline: majority vote over N agents."""
    poll_result = poll_agents(agents, task.question)
    answers = [r.final_answer for r in poll_result.responses]
    predicted = _majority_vote(answers) if answers else ""
    is_correct = _answers_match(predicted, task.ground_truth)

    # Confidence from agreement fraction
    if answers:
        normalized = [_normalize_answer(a) for a in answers]
        counter = Counter(normalized)
        top_count = counter.most_common(1)[0][1]
        confidence = top_count / len(answers)
    else:
        confidence = 0.0

    return EvalResult(
        task_id=task.task_id,
        predicted=predicted,
        correct=task.ground_truth,
        is_correct=is_correct,
        n_rerolls=0,
        n_steps=1,
        cost_estimate=_estimate_cost(len(agents), 1, 0, BaselineMode.SELF_CONSISTENCY),
        confidence=confidence,
    )


def _run_discouq_only(
    task: BenchmarkTask, agents: list[ReasoningAgent], config: Config
) -> EvalResult:
    """DiscoUQ-only baseline: disagreement monitoring, no entropy or utility."""
    poll_result = poll_agents(agents, task.question)
    responses = poll_result.responses

    if not responses:
        return EvalResult(
            task_id=task.task_id,
            predicted="",
            correct=task.ground_truth,
            is_correct=False,
            cost_estimate=_estimate_cost(len(agents), 1, 0, BaselineMode.DISCOUQ_ONLY),
            confidence=0.0,
        )

    monitor = DisagreementMonitor(config.disagreement)
    signal = monitor.compute_signal(responses)

    n_rerolls = 0
    # If weak agreement tier, attempt one reroll round
    if signal.disagreement_tier == "weak" and agents:
        try:
            extras = agent_reroll(agents[0], task.question, n_candidates=3)
            responses.extend(extras)
            n_rerolls = 1
        except Exception:
            pass

    answers = [r.final_answer for r in responses]
    predicted = _majority_vote(answers)
    is_correct = _answers_match(predicted, task.ground_truth)

    return EvalResult(
        task_id=task.task_id,
        predicted=predicted,
        correct=task.ground_truth,
        is_correct=is_correct,
        signals_history=[
            {
                "confidence_score": signal.confidence_score,
                "disagreement_tier": signal.disagreement_tier,
                "evidence_overlap": signal.evidence_overlap,
            }
        ],
        n_rerolls=n_rerolls,
        n_steps=1,
        cost_estimate=_estimate_cost(
            len(agents), 1, n_rerolls, BaselineMode.DISCOUQ_ONLY
        ),
        confidence=signal.confidence_score,
    )


def _run_raptor_full(
    task: BenchmarkTask, agents: list[ReasoningAgent], config: Config
) -> EvalResult:
    """Full RAPTOR pipeline: entropy + disagreement + utility orchestration."""
    result = run_with_raptor(agents=agents, prompt=task.question, config=config)

    predicted = result.final_answer
    is_correct = _answers_match(predicted, task.ground_truth)
    signals_history = result.context.signal_history
    n_rerolls = result.context.action_history.count(OrchestrationAction.REROLL.value)

    # Derive confidence from the last signal vector
    if signals_history:
        last = signals_history[-1]
        confidence = 1.0 - last.get("disagreement_score", 0.5)
    else:
        confidence = 0.5

    return EvalResult(
        task_id=task.task_id,
        predicted=predicted,
        correct=task.ground_truth,
        is_correct=is_correct,
        signals_history=signals_history,
        n_rerolls=n_rerolls,
        n_steps=result.steps_taken,
        cost_estimate=_estimate_cost(
            len(agents), result.steps_taken, n_rerolls, BaselineMode.RAPTOR_FULL
        ),
        confidence=confidence,
    )


_RUNNERS: dict[BaselineMode, Callable[..., EvalResult]] = {
    BaselineMode.NAIVE: _run_naive,
    BaselineMode.SELF_CONSISTENCY: _run_self_consistency,
    BaselineMode.DISCOUQ_ONLY: _run_discouq_only,
    BaselineMode.RAPTOR_FULL: _run_raptor_full,
}


# ══════════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ══════════════════════════════════════════════════════════════════════


def run_experiment(
    config: ExperimentConfig,
    tasks: list[BenchmarkTask],
    agents: list[ReasoningAgent],
    on_result: Optional[Callable[[EvalResult, int], None]] = None,
) -> list[EvalResult]:
    """Run evaluation loop over tasks using the specified baseline mode.

    Args:
        config: Experiment configuration (mode, n_samples, etc.).
        tasks: List of BenchmarkTask to evaluate.
        agents: List of ReasoningAgent instances.
        on_result: Optional per-result callback ``(result, index)``.

    Returns:
        List of EvalResult, one per task (up to *n_samples*).

    Raises:
        ValueError: If baseline_mode is unknown.
    """
    runner = _RUNNERS.get(config.baseline_mode)
    if runner is None:
        raise ValueError(f"Unknown baseline mode: {config.baseline_mode}")

    sample_tasks = tasks[: config.n_samples] if config.n_samples > 0 else tasks
    results: list[EvalResult] = []

    for i, task in enumerate(sample_tasks):
        try:
            result = runner(task, agents, config.raptor_config)
            results.append(result)
            if on_result:
                on_result(result, i)
        except Exception as exc:
            logger.warning(
                "Task {task_id} failed: {err}",
                task_id=task.task_id,
                err=str(exc),
            )
            results.append(
                EvalResult(
                    task_id=task.task_id,
                    predicted="",
                    correct=task.ground_truth,
                    is_correct=False,
                    cost_estimate=0.0,
                )
            )

    return results


# ══════════════════════════════════════════════════════════════════════
# Metrics Computation
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment run.

    Attributes:
        accuracy: Fraction of tasks answered correctly.
        ece: Expected Calibration Error.
        avg_rerolls: Mean rerolls per task.
        avg_steps: Mean orchestration steps per task.
        total_cost: Sum of cost estimates across all tasks.
        n_tasks: Number of tasks evaluated.
        n_correct: Number of correct predictions.
    """

    accuracy: float
    ece: float
    avg_rerolls: float
    avg_steps: float
    total_cost: float
    n_tasks: int
    n_correct: int


def compute_metrics(results: list[EvalResult], n_bins: int = 10) -> ExperimentMetrics:
    """Compute evaluation metrics from experiment results.

    Args:
        results: List of EvalResult from :func:`run_experiment`.
        n_bins: Number of bins for ECE computation.

    Returns:
        :class:`ExperimentMetrics` with accuracy, ECE, and cost stats.
    """
    if not results:
        return ExperimentMetrics(
            accuracy=0.0,
            ece=0.0,
            avg_rerolls=0.0,
            avg_steps=0.0,
            total_cost=0.0,
            n_tasks=0,
            n_correct=0,
        )

    n_tasks = len(results)
    n_correct = sum(1 for r in results if r.is_correct)
    accuracy = n_correct / n_tasks
    ece = _compute_ece(results, n_bins)
    avg_rerolls = sum(r.n_rerolls for r in results) / n_tasks
    avg_steps = sum(r.n_steps for r in results) / n_tasks
    total_cost = sum(r.cost_estimate for r in results)

    return ExperimentMetrics(
        accuracy=accuracy,
        ece=ece,
        avg_rerolls=avg_rerolls,
        avg_steps=avg_steps,
        total_cost=total_cost,
        n_tasks=n_tasks,
        n_correct=n_correct,
    )


def _compute_ece(results: list[EvalResult], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    ECE = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|

    Bins predictions by confidence, measures the gap between accuracy
    and confidence in each bin, weighted by bin size.
    """
    if not results:
        return 0.0

    bin_correct = [0] * n_bins
    bin_confidence = [0.0] * n_bins
    bin_count = [0] * n_bins

    for r in results:
        conf = max(0.0, min(1.0, r.confidence))
        # Map confidence to bin index (0-indexed)
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_count[bin_idx] += 1
        bin_correct[bin_idx] += int(r.is_correct)
        bin_confidence[bin_idx] += conf

    n = len(results)
    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] > 0:
            avg_acc = bin_correct[b] / bin_count[b]
            avg_conf = bin_confidence[b] / bin_count[b]
            ece += (bin_count[b] / n) * abs(avg_acc - avg_conf)

    return ece


# ══════════════════════════════════════════════════════════════════════
# Config Sweep
# ══════════════════════════════════════════════════════════════════════


def sweep_configs(
    base_config: ExperimentConfig,
    param_grid: dict[str, list[Any]],
) -> list[ExperimentConfig]:
    """Generate ExperimentConfigs for threshold/weight sweeps.

    Takes a Cartesian product of all values in *param_grid* and produces
    one :class:`ExperimentConfig` per combination.

    Supported param_grid keys:
      - ``baseline_mode``: list of :class:`BaselineMode`
      - ``n_agents``: list of int
      - ``max_rerolls``: list of int → ``raptor_config.max_rerolls``
      - ``switch_threshold``: list of float → ``raptor_config.utility.switch_threshold``
      - ``gain_weight``: list of float → ``raptor_config.utility.weights["gain"]``
      - ``confidence_weight``: list of float
      - ``cost_penalty_weight``: list of float
      - ``redundancy_penalty_weight``: list of float
      - ``severity_weight``: list of float
      - ``monotonicity_threshold``: list of float → ``raptor_config.entropy``

    Returns:
        List of :class:`ExperimentConfig`, one per combination.
    """
    if not param_grid:
        return [base_config]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs: list[ExperimentConfig] = []
    for combo in product(*values):
        params = dict(zip(keys, combo))

        # Deep-copy the base raptor config to avoid cross-mutation
        rc = copy.deepcopy(base_config.raptor_config)

        new_exp = ExperimentConfig(
            raptor_config=rc,
            baseline_mode=params.get("baseline_mode", base_config.baseline_mode),
            n_agents=params.get("n_agents", base_config.n_agents),
            dataset=base_config.dataset,
            n_samples=base_config.n_samples,
        )

        # Apply raptor_config-level overrides
        if "max_rerolls" in params:
            new_exp.raptor_config.max_rerolls = params["max_rerolls"]
        if "switch_threshold" in params:
            new_exp.raptor_config.utility.switch_threshold = params["switch_threshold"]
        if "monotonicity_threshold" in params:
            new_exp.raptor_config.entropy.monotonicity_threshold = params[
                "monotonicity_threshold"
            ]

        # Weight overrides
        _WEIGHT_MAP = {
            "gain_weight": "gain",
            "confidence_weight": "confidence",
            "cost_penalty_weight": "cost_penalty",
            "redundancy_penalty_weight": "redundancy_penalty",
            "severity_weight": "severity",
        }
        for param_key, weight_key in _WEIGHT_MAP.items():
            if param_key in params:
                new_exp.raptor_config.utility.weights[weight_key] = params[param_key]

        # Build descriptive label
        label_parts = [f"{k}={v}" for k, v in params.items()]
        new_exp.label = ", ".join(label_parts)

        configs.append(new_exp)

    return configs


# ══════════════════════════════════════════════════════════════════════
# Experiment Report
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentReport:
    """Aggregated experiment results with comparison across configurations.

    Collects results from multiple experiment runs and generates
    markdown comparison tables or JSON exports.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add_result(
        self,
        config: ExperimentConfig,
        metrics: ExperimentMetrics,
        results: Optional[list[EvalResult]] = None,
    ) -> None:
        """Add one experiment's results to the report.

        Args:
            config: The experiment configuration.
            metrics: Computed metrics for this run.
            results: Optional raw results (not stored, only used for
                per-task detail in future extensions).
        """
        self.entries.append(
            {
                "label": config.label or config.baseline_mode.value,
                "baseline_mode": config.baseline_mode.value,
                "n_agents": config.n_agents,
                "dataset": config.dataset,
                "n_samples": config.n_samples,
                "accuracy": metrics.accuracy,
                "ece": metrics.ece,
                "avg_rerolls": metrics.avg_rerolls,
                "avg_steps": metrics.avg_steps,
                "total_cost": metrics.total_cost,
                "n_tasks": metrics.n_tasks,
                "n_correct": metrics.n_correct,
            }
        )

    def to_markdown(self) -> str:
        """Generate a markdown comparison table."""
        if not self.entries:
            return "No experiment results to report."

        lines = [
            "# RAPTOR Experiment Report\n",
            "| Config | Mode | Agents | Accuracy | ECE | Avg Rerolls | Avg Steps | Total Cost |",
            "|--------|------|--------|----------|-----|-------------|-----------|------------|",
        ]

        for e in self.entries:
            lines.append(
                f"| {e['label']} | {e['baseline_mode']} | {e['n_agents']} | "
                f"{e['accuracy']:.4f} | {e['ece']:.4f} | "
                f"{e['avg_rerolls']:.2f} | {e['avg_steps']:.2f} | "
                f"{e['total_cost']:.2f} |"
            )

        if len(self.entries) > 1:
            best = max(self.entries, key=lambda e: e["accuracy"])
            cheapest = min(self.entries, key=lambda e: e["total_cost"])
            best_cal = min(self.entries, key=lambda e: e["ece"])

            lines.extend(
                [
                    "",
                    "## Summary",
                    f"- **Best accuracy:** {best['label']} ({best['accuracy']:.4f})",
                    f"- **Lowest cost:** {cheapest['label']} ({cheapest['total_cost']:.2f})",
                    f"- **Best calibration (lowest ECE):** {best_cal['label']} ({best_cal['ece']:.4f})",
                ]
            )

        return "\n".join(lines)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize report to JSON string."""
        return json.dumps({"experiments": self.entries}, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return report as a dictionary."""
        return {"experiments": list(self.entries)}


# ══════════════════════════════════════════════════════════════════════
# Synthetic Benchmark Data Generators
# ══════════════════════════════════════════════════════════════════════


# --------------------------------------------------------------------------
# GSM8K — grade-school math word problems
# --------------------------------------------------------------------------


def generate_gsm8k_synthetic(n: int, seed: int = 42) -> list[BenchmarkTask]:
    """Generate *n* synthetic GSM8K-style math word problems with known answers.

    Produces simple multi-step arithmetic word problems covering:
    shopping, distance, time, sharing, and savings scenarios.

    Args:
        n: Number of tasks to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of :class:`BenchmarkTask` with ``dataset_name="gsm8k_synthetic"``.
    """
    rng = random.Random(seed)
    _templates = [_gsm8k_shopping, _gsm8k_distance, _gsm8k_time, _gsm8k_sharing, _gsm8k_savings]
    tasks: list[BenchmarkTask] = []

    for i in range(n):
        fn = _templates[i % len(_templates)]
        question, answer = fn(rng)
        tasks.append(
            BenchmarkTask(
                question=question,
                ground_truth=str(answer),
                dataset_name="gsm8k_synthetic",
                task_id=f"gsm8k_synth_{i:04d}",
            )
        )
    return tasks


def _gsm8k_shopping(rng: random.Random) -> tuple[str, int]:
    items = ["apples", "oranges", "books", "pencils", "cookies", "toys"]
    item = rng.choice(items)
    n_items = rng.randint(2, 15)
    price = rng.randint(1, 10)
    discount = rng.randint(0, n_items * price // 3)
    total = n_items * price - discount
    q = (
        f"Sarah goes to the store and buys {n_items} {item} at ${price} each. "
        f"She has a coupon for ${discount} off. How much does she pay in total?"
    )
    return q, total


def _gsm8k_distance(rng: random.Random) -> tuple[str, int]:
    s1, t1 = rng.randint(20, 60), rng.randint(1, 5)
    s2, t2 = rng.randint(20, 60), rng.randint(1, 5)
    total = s1 * t1 + s2 * t2
    q = (
        f"A car drives at {s1} mph for {t1} hours, then at {s2} mph "
        f"for {t2} hours. What is the total distance traveled in miles?"
    )
    return q, total


def _gsm8k_time(rng: random.Random) -> tuple[str, int]:
    n_tasks = rng.randint(3, 8)
    mins = rng.randint(5, 30)
    brk = rng.randint(5, 15)
    total = n_tasks * mins + (n_tasks - 1) * brk
    q = (
        f"Tom has {n_tasks} homework assignments. Each takes {mins} minutes. "
        f"He takes a {brk}-minute break between each assignment. "
        f"How many minutes does it take him to finish all assignments including breaks?"
    )
    return q, total


def _gsm8k_sharing(rng: random.Random) -> tuple[str, int]:
    total_items = rng.randint(20, 100)
    n_people = rng.randint(2, 8)
    extra = rng.randint(1, 10)
    each = total_items // n_people
    q = (
        f"There are {total_items} stickers to share equally among {n_people} friends. "
        f"Then each friend gets {extra} more stickers as a bonus. "
        f"How many stickers does each friend have?"
    )
    return q, each + extra


def _gsm8k_savings(rng: random.Random) -> tuple[str, int]:
    weekly = rng.randint(5, 50)
    weeks = rng.randint(2, 12)
    initial = rng.randint(0, 100)
    spent = rng.randint(0, initial + weekly * weeks // 2)
    total = initial + weekly * weeks - spent
    q = (
        f"Emma saves ${weekly} per week for {weeks} weeks. She started with ${initial} "
        f"in savings and spent ${spent} on a gift. How much money does she have now?"
    )
    return q, total


# --------------------------------------------------------------------------
# MATH — harder math (algebra, geometry, number theory, combinatorics, sequences)
# --------------------------------------------------------------------------


def generate_math_synthetic(n: int, seed: int = 42) -> list[BenchmarkTask]:
    """Generate *n* synthetic MATH-style problems (harder than GSM8K).

    Covers algebra, geometry, number theory, combinatorics, and sequences.

    Args:
        n: Number of tasks to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of :class:`BenchmarkTask` with ``dataset_name="math_synthetic"``.
    """
    rng = random.Random(seed)
    _templates = [_math_algebra, _math_geometry, _math_number_theory, _math_combinatorics, _math_sequences]
    tasks: list[BenchmarkTask] = []

    for i in range(n):
        fn = _templates[i % len(_templates)]
        question, answer = fn(rng)
        tasks.append(
            BenchmarkTask(
                question=question,
                ground_truth=str(answer),
                dataset_name="math_synthetic",
                task_id=f"math_synth_{i:04d}",
            )
        )
    return tasks


def _math_algebra(rng: random.Random) -> tuple[str, int]:
    a = rng.randint(2, 10)
    x = rng.randint(1, 15)
    b = rng.randint(1, 20)
    result = a * x + b
    return f"Solve for x: {a}x + {b} = {result}", x


def _math_geometry(rng: random.Random) -> tuple[str, int]:
    base, height = rng.randint(3, 20), rng.randint(3, 20)
    return f"A rectangle has base {base} and height {height}. What is its area?", base * height


def _math_number_theory(rng: random.Random) -> tuple[str, int]:
    n = rng.randint(10, 100)
    answer = sum(i for i in range(1, n + 1) if n % i == 0)
    return f"What is the sum of all positive divisors of {n}?", answer


def _math_combinatorics(rng: random.Random) -> tuple[str, int]:
    n = rng.randint(3, 8)
    r = rng.randint(1, min(n, 4))
    return f"How many ways can you choose {r} items from {n} distinct items?", comb(n, r)


def _math_sequences(rng: random.Random) -> tuple[str, int]:
    a1 = rng.randint(1, 10)
    d = rng.randint(1, 5)
    n = rng.randint(5, 15)
    total = n * (2 * a1 + (n - 1) * d) // 2
    return (
        f"Find the sum of the first {n} terms of an arithmetic sequence "
        f"with first term {a1} and common difference {d}."
    ), total


# --------------------------------------------------------------------------
# HotpotQA — multi-hop reasoning questions
# --------------------------------------------------------------------------


def generate_hotpotqa_synthetic(n: int, seed: int = 42) -> list[BenchmarkTask]:
    """Generate *n* synthetic HotpotQA-style multi-hop QA tasks.

    Covers comparison, bridge, intersection, and temporal question types.

    Args:
        n: Number of tasks to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of :class:`BenchmarkTask` with ``dataset_name="hotpotqa_synthetic"``.
    """
    rng = random.Random(seed)
    _templates = [_hotpot_comparison, _hotpot_bridge, _hotpot_intersection, _hotpot_temporal]
    tasks: list[BenchmarkTask] = []

    for i in range(n):
        fn = _templates[i % len(_templates)]
        question, answer = fn(rng)
        tasks.append(
            BenchmarkTask(
                question=question,
                ground_truth=answer,
                dataset_name="hotpotqa_synthetic",
                task_id=f"hotpotqa_synth_{i:04d}",
            )
        )
    return tasks


def _hotpot_comparison(rng: random.Random) -> tuple[str, str]:
    cities = [
        ("New York", 8_336_817),
        ("Los Angeles", 3_979_576),
        ("Chicago", 2_693_976),
        ("Houston", 2_320_268),
        ("Phoenix", 1_680_992),
        ("Philadelphia", 1_584_064),
        ("San Antonio", 1_547_253),
        ("San Diego", 1_423_851),
        ("Dallas", 1_343_573),
        ("San Jose", 1_021_795),
    ]
    i1, i2 = rng.sample(range(len(cities)), 2)
    c1, p1 = cities[i1]
    c2, p2 = cities[i2]
    answer = c1 if p1 > p2 else c2
    q = (
        f"{c1} has a population of {p1:,}. {c2} has a population of {p2:,}. "
        f"Which city has a larger population?"
    )
    return q, answer


def _hotpot_bridge(rng: random.Random) -> tuple[str, str]:
    subjects = [
        ("Albert Einstein", "physics", "Germany"),
        ("Marie Curie", "chemistry", "Poland"),
        ("Isaac Newton", "physics", "England"),
        ("Charles Darwin", "biology", "England"),
        ("Nikola Tesla", "engineering", "Serbia"),
        ("Ada Lovelace", "computing", "England"),
    ]
    name, field_, country = rng.choice(subjects)
    q = (
        f"{name} was born in {country} and made groundbreaking contributions "
        f"to {field_}. What country was {name} born in?"
    )
    return q, country


def _hotpot_intersection(rng: random.Random) -> tuple[str, str]:
    pairs = [
        ("Python", "Java", "programming languages"),
        ("Mars", "Venus", "planets in our solar system"),
        ("Beethoven", "Mozart", "classical music composers"),
        ("Shakespeare", "Dickens", "English authors"),
        ("Tesla", "Edison", "inventors who worked with electricity"),
    ]
    a, b, common = rng.choice(pairs)
    return f"What do {a} and {b} have in common?", common


def _hotpot_temporal(rng: random.Random) -> tuple[str, str]:
    events = [
        ("World War I started", 1914),
        ("World War II ended", 1945),
        ("Moon landing", 1969),
        ("Berlin Wall fell", 1989),
        ("Internet created", 1969),
    ]
    i1, i2 = rng.sample(range(len(events)), 2)
    e1, y1 = events[i1]
    e2, y2 = events[i2]
    diff = abs(y1 - y2)
    q = (
        f"The {e1} in {y1} and the {e2} in {y2}. "
        f"How many years apart were these events?"
    )
    return q, str(diff)
