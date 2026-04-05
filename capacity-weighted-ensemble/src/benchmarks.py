"""
Benchmark Suite for Capacity-Weighted Test-Time Ensemble (CWTTE).

Provides synthetic, deterministic benchmark tasks across three kinds
(multi-hop reasoning, QA, mathematical reasoning) and a harness for
comparing compute-allocation + voting strategies:

  - "uniform"     : equal budget per agent, majority vote
  - "uncertainty" : probe + scale-up-when-uncertain, majority vote
  - "capacity"    : budget allocated by known per-agent capacity,
                    capacity-weighted voting (CapacityWeightedVoter)

The suite uses lightweight ``MockCapacityAgent`` instances with
configurable per-kind competence, so we can show that capacity-aware
allocation beats uniform when agents have varied expertise.

No external API calls, no LLMs -- pure numpy/stdlib. Deterministic
given the same seeds, budgets, agents, and tasks.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .allocator import ComputeAllocator
from .uncertainty import UncertaintyEstimator
from .voting import CapacityWeightedVoter


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

TASK_KINDS: Tuple[str, ...] = ("multihop", "qa", "math")
STRATEGIES: Tuple[str, ...] = ("uniform", "uncertainty", "capacity")


@dataclass
class BenchmarkTask:
    """
    A single benchmark task.

    Attributes:
        id: Stable task identifier (e.g. ``"math_0003"``).
        kind: Task category, one of ``TASK_KINDS``.
        prompt: Human-readable prompt text.
        expected_answer: The canonical correct answer (string).
        distractors: Alternative wrong answers used by mock agents.
        metadata: Freeform metadata dict (hops, operands, ...).
    """

    id: str
    kind: str
    prompt: str
    expected_answer: str
    distractors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("BenchmarkTask.id must be non-empty")
        if self.kind not in TASK_KINDS:
            raise ValueError(
                f"BenchmarkTask.kind must be one of {TASK_KINDS}, got {self.kind!r}"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("BenchmarkTask.prompt must be a non-empty string")
        if not isinstance(self.expected_answer, str) or not self.expected_answer:
            raise ValueError("BenchmarkTask.expected_answer must be a non-empty string")


@dataclass
class BenchmarkResult:
    """
    Aggregated result of running one strategy across a BenchmarkSuite.

    Attributes:
        strategy: Strategy name used.
        accuracy: Fraction of tasks answered correctly (0-1).
        avg_tokens: Average tokens consumed per task.
        compute_efficiency: Accuracy achieved per 1K tokens spent.
            ``accuracy / (avg_tokens / 1000)``. 0.0 if avg_tokens == 0.
        per_task_trace: Per-task details (predicted, expected, correct,
            tokens, num_samples, ...).
        num_tasks: Number of tasks evaluated.
        num_correct: Absolute count of correct answers.
        total_tokens: Sum of tokens across all tasks.
        metadata: Arbitrary extra info.
    """

    strategy: str
    accuracy: float
    avg_tokens: float
    compute_efficiency: float
    per_task_trace: List[Dict[str, Any]]
    num_tasks: int
    num_correct: int
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"BenchmarkResult.strategy must be one of {STRATEGIES}, got {self.strategy!r}"
            )
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(
                f"BenchmarkResult.accuracy must be in [0,1], got {self.accuracy}"
            )
        if self.avg_tokens < 0:
            raise ValueError("avg_tokens must be non-negative")
        if self.compute_efficiency < 0:
            raise ValueError("compute_efficiency must be non-negative")
        if self.num_tasks < 0:
            raise ValueError("num_tasks must be non-negative")
        if self.num_correct < 0 or self.num_correct > self.num_tasks:
            raise ValueError("num_correct must be in [0, num_tasks]")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if len(self.per_task_trace) != self.num_tasks:
            raise ValueError(
                "per_task_trace length must equal num_tasks "
                f"({len(self.per_task_trace)} vs {self.num_tasks})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "accuracy": self.accuracy,
            "avg_tokens": self.avg_tokens,
            "compute_efficiency": self.compute_efficiency,
            "num_tasks": self.num_tasks,
            "num_correct": self.num_correct,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------

def _stable_seed(*parts: Any) -> int:
    """Deterministic 64-bit seed from heterogeneous parts (no PYTHONHASHSEED)."""
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class MockCapacityAgent:
    """
    Lightweight synthetic agent with configurable per-kind competence.

    ``competence[kind]`` is the probability the agent returns the correct
    answer for tasks of that kind. Wrong answers are drawn uniformly from
    the task's ``distractors``.

    Each ``answer()`` call is deterministic given
    ``(agent_id, task.id, sample_idx, seed)``.
    """

    def __init__(
        self,
        agent_id: str,
        competence: Dict[str, float],
        seed: int = 0,
        tokens_per_call: int = 50,
        capacity_scale: float = 10.0,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        for k, v in competence.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"competence[{k!r}] must be in [0,1], got {v}"
                )
        if tokens_per_call < 0:
            raise ValueError("tokens_per_call must be non-negative")
        if capacity_scale <= 0:
            raise ValueError("capacity_scale must be positive")

        self.agent_id = agent_id
        self.competence = dict(competence)
        self.seed = seed
        self.tokens_per_call = tokens_per_call
        self.capacity_scale = capacity_scale

        # Alias commonly checked by coordinator._get_agent_id()
        self.id = agent_id

    def answer(self, task: BenchmarkTask, sample_idx: int = 0) -> Tuple[str, int]:
        """Return ``(response, tokens_used)`` deterministically."""
        rng = random.Random(_stable_seed(self.agent_id, task.id, sample_idx, self.seed))
        p_correct = self.competence.get(task.kind, 0.25)
        if rng.random() < p_correct:
            return task.expected_answer, self.tokens_per_call
        if task.distractors:
            return rng.choice(task.distractors), self.tokens_per_call
        return f"wrong_{rng.randint(0, 999)}", self.tokens_per_call

    def capacity_for(self, kind: str) -> float:
        """Known capacity for a task kind: competence × scale."""
        return self.competence.get(kind, 0.25) * self.capacity_scale

    # Compatibility: generate(prompt) returns a string only. Used by callers
    # that expect an LLM-like interface (capacity estimator sampling).
    def generate(self, prompt: Any) -> str:
        """Very rough ``generate`` fallback (adds per-call diversity)."""
        key = str(prompt)
        rng = random.Random(_stable_seed(self.agent_id, key, self.seed))
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        return " ".join(rng.choice(words) for _ in range(3))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MockCapacityAgent(id={self.agent_id!r}, competence={self.competence})"


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

_MULTIHOP_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve",
    "Frank", "Grace", "Hank", "Ivy", "Jack",
]
_MULTIHOP_PLACES = [
    "Paris", "Tokyo", "Madrid", "Cairo", "Lima",
    "Rome", "Oslo", "Delhi", "Sydney", "Seoul",
]

_QA_BANK: List[Tuple[str, str, List[str]]] = [
    ("What is the capital of France?", "Paris",
     ["London", "Berlin", "Madrid", "Rome"]),
    ("How many days are in a week?", "7",
     ["5", "6", "8", "10"]),
    ("Who wrote Hamlet?", "Shakespeare",
     ["Dickens", "Austen", "Wilde", "Orwell"]),
    ("What color are bananas when ripe?", "yellow",
     ["red", "green", "blue", "purple"]),
    ("What is H2O commonly called?", "water",
     ["salt", "acid", "metal", "gas"]),
    ("How many continents are there?", "7",
     ["5", "6", "8", "9"]),
    ("Which planet is called the Red Planet?", "Mars",
     ["Venus", "Jupiter", "Saturn", "Mercury"]),
    ("What is the largest ocean?", "Pacific",
     ["Atlantic", "Indian", "Arctic", "Southern"]),
    ("What language is primarily spoken in Brazil?", "Portuguese",
     ["Spanish", "English", "French", "Italian"]),
    ("At what Celsius does water boil at sea level?", "100",
     ["50", "75", "90", "120"]),
]


def generate_multihop_tasks(n: int, seed: int = 0) -> List[BenchmarkTask]:
    """Generate ``n`` deterministic 2-hop reasoning tasks."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(_stable_seed("multihop", seed))
    tasks: List[BenchmarkTask] = []
    for i in range(n):
        a, b, c = rng.sample(_MULTIHOP_NAMES, 3)
        place = rng.choice(_MULTIHOP_PLACES)
        prompt = (
            f"{a} traveled with {b}. {b} was born in {place}. "
            f"{c} has never met {b}. Who was born in {place}?"
        )
        tasks.append(
            BenchmarkTask(
                id=f"multihop_{i:04d}",
                kind="multihop",
                prompt=prompt,
                expected_answer=b,
                distractors=[a, c, place],
                metadata={"hops": 2, "entities": [a, b, c], "place": place},
            )
        )
    return tasks


def generate_qa_tasks(n: int, seed: int = 0) -> List[BenchmarkTask]:
    """Generate ``n`` deterministic factual QA tasks."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(_stable_seed("qa", seed))
    indices = list(range(len(_QA_BANK)))
    rng.shuffle(indices)
    tasks: List[BenchmarkTask] = []
    for i in range(n):
        # Cycle through bank with a reshuffle each pass for variety.
        if i > 0 and i % len(_QA_BANK) == 0:
            rng.shuffle(indices)
        q, ans, distractors = _QA_BANK[indices[i % len(_QA_BANK)]]
        tasks.append(
            BenchmarkTask(
                id=f"qa_{i:04d}",
                kind="qa",
                prompt=q,
                expected_answer=ans,
                distractors=list(distractors),
                metadata={"bank_index": indices[i % len(_QA_BANK)]},
            )
        )
    return tasks


def generate_math_tasks(n: int, seed: int = 0) -> List[BenchmarkTask]:
    """Generate ``n`` deterministic arithmetic tasks."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(_stable_seed("math", seed))
    ops = ["+", "-", "*"]
    tasks: List[BenchmarkTask] = []
    for i in range(n):
        op = rng.choice(ops)
        x = rng.randint(2, 50)
        y = rng.randint(2, 50)
        if op == "+":
            ans = x + y
        elif op == "-":
            ans = x - y
        else:
            ans = x * y
        expected = str(ans)
        # Plausible near-miss distractors.
        raw = {ans + 1, ans - 1, ans + 10, ans - 10, ans * 2, ans + x, ans + y}
        raw.discard(ans)
        distractors = [str(v) for v in list(raw)[:4]]
        tasks.append(
            BenchmarkTask(
                id=f"math_{i:04d}",
                kind="math",
                prompt=f"Compute: {x} {op} {y}",
                expected_answer=expected,
                distractors=distractors,
                metadata={"x": x, "y": y, "op": op},
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """
    A container of BenchmarkTask objects with convenience constructors.

    Raises ``ValueError`` on empty task lists: benchmarks require tasks.
    """

    def __init__(self, tasks: List[BenchmarkTask], name: str = "benchmark") -> None:
        if not tasks:
            raise ValueError("BenchmarkSuite requires at least one task")
        for t in tasks:
            if not isinstance(t, BenchmarkTask):
                raise TypeError(f"tasks must contain BenchmarkTask, got {type(t).__name__}")
        self.tasks: List[BenchmarkTask] = list(tasks)
        self.name = name

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def count_by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {k: 0 for k in TASK_KINDS}
        for t in self.tasks:
            counts[t.kind] = counts.get(t.kind, 0) + 1
        return counts

    @classmethod
    def from_mixed(cls, n_per_kind: int = 5, seed: int = 0) -> "BenchmarkSuite":
        """Build a suite with equal tasks per kind."""
        if n_per_kind < 1:
            raise ValueError("n_per_kind must be at least 1")
        tasks: List[BenchmarkTask] = []
        tasks.extend(generate_multihop_tasks(n_per_kind, seed=seed))
        tasks.extend(generate_qa_tasks(n_per_kind, seed=seed))
        tasks.extend(generate_math_tasks(n_per_kind, seed=seed))
        return cls(tasks=tasks, name=f"mixed-{n_per_kind}x3")


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _majority_vote(responses: List[str]) -> str:
    """Unweighted plurality vote; returns first-seen on ties."""
    voter = CapacityWeightedVoter(temperature=1.0)
    result = voter.weighted_vote(responses, [1.0] * len(responses))
    return result.winning_response


def _run_uniform(
    agents: List[MockCapacityAgent],
    task: BenchmarkTask,
    budget: int,
) -> Tuple[str, int, Dict[str, Any]]:
    per_agent = max(1, budget // max(1, len(agents)))
    responses: List[str] = []
    tokens = 0
    per_agent_counts: Dict[str, int] = {}
    for agent in agents:
        for k in range(per_agent):
            resp, tok = agent.answer(task, sample_idx=k)
            responses.append(resp)
            tokens += tok
        per_agent_counts[agent.agent_id] = per_agent
    predicted = _majority_vote(responses)
    return predicted, tokens, {
        "num_samples": len(responses),
        "per_agent_calls": per_agent_counts,
    }


def _run_uncertainty(
    agents: List[MockCapacityAgent],
    task: BenchmarkTask,
    budget: int,
    uncertainty_threshold: float = 0.3,
) -> Tuple[str, int, Dict[str, Any]]:
    # 1. Probe each agent with a single sample.
    probes: List[Tuple[MockCapacityAgent, str]] = []
    tokens = 0
    for agent in agents:
        resp, tok = agent.answer(task, sample_idx=0)
        probes.append((agent, resp))
        tokens += tok

    # 2. Measure global uncertainty among probe responses.
    estimator = UncertaintyEstimator(normalize=True)
    unc = estimator.estimate_uncertainty([r for _, r in probes])

    per_agent_counts: Dict[str, int] = {a.agent_id: 1 for a in agents}

    # 3. If confident, stop here. Otherwise scale up by distributing
    #    the remaining budget equally across agents.
    remaining = max(0, budget - len(agents))
    if unc.uncertainty_score >= uncertainty_threshold and remaining > 0:
        extra = remaining // len(agents)
        leftover = remaining - extra * len(agents)
        responses = [r for _, r in probes]
        for i, (agent, _probe_resp) in enumerate(probes):
            extra_for_agent = extra + (1 if i < leftover else 0)
            for k in range(1, 1 + extra_for_agent):
                resp, tok = agent.answer(task, sample_idx=k)
                responses.append(resp)
                tokens += tok
            per_agent_counts[agent.agent_id] = 1 + extra_for_agent
    else:
        responses = [r for _, r in probes]

    predicted = _majority_vote(responses)
    return predicted, tokens, {
        "num_samples": len(responses),
        "per_agent_calls": per_agent_counts,
        "probe_uncertainty": unc.uncertainty_score,
        "scaled_up": unc.uncertainty_score >= uncertainty_threshold and remaining > 0,
    }


def _run_capacity(
    agents: List[MockCapacityAgent],
    task: BenchmarkTask,
    budget: int,
    capacity_weight: float = 0.8,
) -> Tuple[str, int, Dict[str, Any]]:
    capacities = {a.agent_id: a.capacity_for(task.kind) for a in agents}
    # Neutral uncertainties so allocation tracks capacity.
    uncertainties = {a.agent_id: 0.5 for a in agents}

    allocator = ComputeAllocator(
        total_budget=max(len(agents), budget),
        capacity_weight=capacity_weight,
        uncertainty_weight=1.0 - capacity_weight,
    )
    allocation = allocator.allocate(
        agents=[a.agent_id for a in agents],
        capacities=capacities,
        uncertainties=uncertainties,
    )

    responses: List[str] = []
    weights: List[float] = []
    tokens = 0
    per_agent_counts: Dict[str, int] = {}
    for agent in agents:
        n_calls = allocation.get(agent.agent_id, 0)
        per_agent_counts[agent.agent_id] = n_calls
        for k in range(n_calls):
            resp, tok = agent.answer(task, sample_idx=k)
            responses.append(resp)
            weights.append(capacities[agent.agent_id])
            tokens += tok

    # Safety net: if allocator produced zero calls total (shouldn't happen
    # with budget >= num_agents, but defensive), invoke top-capacity agent.
    if not responses:
        top = max(agents, key=lambda a: capacities[a.agent_id])
        resp, tok = top.answer(task, sample_idx=0)
        responses.append(resp)
        weights.append(capacities[top.agent_id])
        tokens += tok
        per_agent_counts[top.agent_id] = per_agent_counts.get(top.agent_id, 0) + 1

    voter = CapacityWeightedVoter(temperature=1.0)
    vote_result = voter.weighted_vote(responses, weights)
    return vote_result.winning_response, tokens, {
        "num_samples": len(responses),
        "per_agent_calls": per_agent_counts,
        "capacities": capacities,
        "allocation": allocation,
    }


_STRATEGY_FNS = {
    "uniform": _run_uniform,
    "uncertainty": _run_uncertainty,
    "capacity": _run_capacity,
}


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_benchmark(
    suite: BenchmarkSuite,
    strategy: str,
    agents: List[MockCapacityAgent],
    budget: int,
) -> BenchmarkResult:
    """
    Run ``strategy`` across every task in ``suite`` using ``agents``.

    Args:
        suite: BenchmarkSuite of tasks to evaluate.
        strategy: One of ``"uniform"``, ``"uncertainty"``, ``"capacity"``.
        agents: List of MockCapacityAgent (or compatible). Non-empty.
        budget: Total compute units per task (each call costs 1 unit,
            and ``tokens_per_call`` tokens). Must be positive.

    Returns:
        BenchmarkResult with accuracy, efficiency, and per-task trace.

    Raises:
        ValueError: on unknown strategy, empty suite, empty agents,
            or non-positive budget.
    """
    if strategy not in _STRATEGY_FNS:
        raise ValueError(
            f"Unknown strategy {strategy!r}; valid: {sorted(_STRATEGY_FNS)}"
        )
    if not isinstance(suite, BenchmarkSuite) or len(suite) == 0:
        raise ValueError("suite must be a non-empty BenchmarkSuite")
    if not agents:
        raise ValueError("agents list must be non-empty")
    if budget <= 0:
        raise ValueError("budget must be positive")

    fn = _STRATEGY_FNS[strategy]
    trace: List[Dict[str, Any]] = []
    total_tokens = 0
    num_correct = 0

    for task in suite.tasks:
        predicted, tokens, extra = fn(agents, task, budget)
        correct = (predicted == task.expected_answer)
        if correct:
            num_correct += 1
        total_tokens += tokens
        trace.append({
            "task_id": task.id,
            "kind": task.kind,
            "predicted": predicted,
            "expected": task.expected_answer,
            "correct": correct,
            "tokens": tokens,
            **extra,
        })

    n = len(suite)
    accuracy = num_correct / n
    avg_tokens = total_tokens / n
    efficiency = (accuracy / (avg_tokens / 1000.0)) if avg_tokens > 0 else 0.0

    return BenchmarkResult(
        strategy=strategy,
        accuracy=accuracy,
        avg_tokens=avg_tokens,
        compute_efficiency=efficiency,
        per_task_trace=trace,
        num_tasks=n,
        num_correct=num_correct,
        total_tokens=total_tokens,
        metadata={
            "budget": budget,
            "num_agents": len(agents),
            "agent_ids": [a.agent_id for a in agents],
            "suite_name": suite.name,
        },
    )


def compare_strategies(
    suite: BenchmarkSuite,
    agents: List[MockCapacityAgent],
    budget: int,
    strategies: Optional[List[str]] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Run multiple strategies against the same suite and return their results.

    Args:
        suite: BenchmarkSuite to run against.
        agents: Agents to use.
        budget: Compute units per task.
        strategies: Strategy names (defaults to all three).

    Returns:
        Dict mapping strategy name to ``BenchmarkResult``.
    """
    strategies = list(strategies) if strategies is not None else list(STRATEGIES)
    if not strategies:
        raise ValueError("strategies list must be non-empty")
    results: Dict[str, BenchmarkResult] = {}
    for strat in strategies:
        results[strat] = run_benchmark(suite, strat, agents, budget)
    return results


# ---------------------------------------------------------------------------
# Convenience: build a varied-specialist agent roster
# ---------------------------------------------------------------------------

def build_specialist_agents(
    high: float = 0.85,
    low: float = 0.35,
    seed: int = 0,
    tokens_per_call: int = 50,
) -> List[MockCapacityAgent]:
    """
    Three agents, each specialising in a different task kind.
    Useful for demonstrating capacity-weighted gains in tests and demos.
    """
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("require 0 <= low <= high <= 1")
    configs = [
        ("agent_hop",  {"multihop": high, "qa": low, "math": low}),
        ("agent_qa",   {"multihop": low,  "qa": high, "math": low}),
        ("agent_math", {"multihop": low,  "qa": low, "math": high}),
    ]
    return [
        MockCapacityAgent(
            agent_id=aid,
            competence=comp,
            seed=seed,
            tokens_per_call=tokens_per_call,
        )
        for aid, comp in configs
    ]


__all__ = [
    "TASK_KINDS",
    "STRATEGIES",
    "BenchmarkTask",
    "BenchmarkResult",
    "BenchmarkSuite",
    "MockCapacityAgent",
    "generate_multihop_tasks",
    "generate_qa_tasks",
    "generate_math_tasks",
    "run_benchmark",
    "compare_strategies",
    "build_specialist_agents",
]
