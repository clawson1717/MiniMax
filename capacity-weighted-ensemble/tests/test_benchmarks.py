"""Tests for src/benchmarks.py — Step 10 benchmark suite."""

import pytest

from src.benchmarks import (
    STRATEGIES,
    TASK_KINDS,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    MockCapacityAgent,
    build_specialist_agents,
    compare_strategies,
    generate_math_tasks,
    generate_multihop_tasks,
    generate_qa_tasks,
    run_benchmark,
)


# ---- Task generators --------------------------------------------------------


class TestGenerateMultihopTasks:
    def test_returns_expected_count(self):
        tasks = generate_multihop_tasks(7)
        assert len(tasks) == 7

    def test_zero_returns_empty(self):
        assert generate_multihop_tasks(0) == []

    def test_valid_structure(self):
        tasks = generate_multihop_tasks(3, seed=42)
        for t in tasks:
            assert isinstance(t, BenchmarkTask)
            assert t.kind == "multihop"
            assert t.expected_answer
            assert len(t.distractors) == 3
            assert t.metadata.get("hops") == 2

    def test_deterministic(self):
        a = generate_multihop_tasks(5, seed=1)
        b = generate_multihop_tasks(5, seed=1)
        assert [t.expected_answer for t in a] == [t.expected_answer for t in b]

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            generate_multihop_tasks(-1)


class TestGenerateQaTasks:
    def test_returns_expected_count(self):
        assert len(generate_qa_tasks(10)) == 10

    def test_valid_structure(self):
        for t in generate_qa_tasks(4, seed=7):
            assert t.kind == "qa"
            assert t.expected_answer
            assert len(t.distractors) >= 1

    def test_wraps_around_bank(self):
        # More tasks than QA bank size (10) should still work.
        tasks = generate_qa_tasks(15)
        assert len(tasks) == 15

    def test_deterministic(self):
        a = generate_qa_tasks(6, seed=99)
        b = generate_qa_tasks(6, seed=99)
        assert [t.prompt for t in a] == [t.prompt for t in b]


class TestGenerateMathTasks:
    def test_returns_expected_count(self):
        assert len(generate_math_tasks(8)) == 8

    def test_valid_structure(self):
        for t in generate_math_tasks(5, seed=3):
            assert t.kind == "math"
            assert t.metadata["op"] in ("+", "-", "*")
            # expected_answer should be the string of the computed result
            x, y, op = t.metadata["x"], t.metadata["y"], t.metadata["op"]
            expected_val = {"+": x + y, "-": x - y, "*": x * y}[op]
            assert t.expected_answer == str(expected_val)

    def test_deterministic(self):
        a = generate_math_tasks(4, seed=0)
        b = generate_math_tasks(4, seed=0)
        assert [t.expected_answer for t in a] == [t.expected_answer for t in b]


# ---- MockCapacityAgent ------------------------------------------------------


class TestMockCapacityAgent:
    def test_deterministic_answer(self):
        agent = MockCapacityAgent("a1", {"qa": 0.9}, seed=0)
        task = BenchmarkTask(
            id="qa_0000", kind="qa", prompt="Q?",
            expected_answer="yes", distractors=["no"],
        )
        r1 = agent.answer(task, sample_idx=0)
        r2 = agent.answer(task, sample_idx=0)
        assert r1 == r2

    def test_different_sample_idx_can_differ(self):
        agent = MockCapacityAgent("a1", {"qa": 0.5}, seed=0)
        task = BenchmarkTask(
            id="qa_0001", kind="qa", prompt="Q?",
            expected_answer="yes", distractors=["no", "maybe"],
        )
        # Collect answers across many samples; at p=0.5 we should see variation.
        answers = {agent.answer(task, sample_idx=i)[0] for i in range(50)}
        assert len(answers) > 1

    def test_capacity_for(self):
        agent = MockCapacityAgent("a1", {"math": 0.8}, capacity_scale=10.0)
        assert agent.capacity_for("math") == pytest.approx(8.0)
        # Unknown kind falls back to 0.25 * scale
        assert agent.capacity_for("unknown") == pytest.approx(2.5)

    def test_generate_returns_string(self):
        agent = MockCapacityAgent("a1", {"qa": 0.5})
        result = agent.generate("hello")
        assert isinstance(result, str) and len(result) > 0

    def test_invalid_competence_raises(self):
        with pytest.raises(ValueError):
            MockCapacityAgent("a1", {"qa": 1.5})

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            MockCapacityAgent("", {"qa": 0.5})


# ---- BenchmarkSuite ---------------------------------------------------------


class TestBenchmarkSuite:
    def test_from_mixed(self):
        suite = BenchmarkSuite.from_mixed(n_per_kind=3, seed=0)
        assert len(suite) == 9
        counts = suite.count_by_kind()
        assert counts == {"multihop": 3, "qa": 3, "math": 3}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            BenchmarkSuite([])

    def test_iterable(self):
        suite = BenchmarkSuite.from_mixed(n_per_kind=2)
        assert len(list(suite)) == 6


# ---- run_benchmark per strategy ---------------------------------------------


@pytest.fixture
def small_suite():
    return BenchmarkSuite.from_mixed(n_per_kind=3, seed=0)


@pytest.fixture
def agents():
    return build_specialist_agents(high=0.85, low=0.35, seed=0)


class TestRunBenchmark:
    @pytest.mark.parametrize("strategy", list(STRATEGIES))
    def test_strategy_end_to_end(self, small_suite, agents, strategy):
        result = run_benchmark(small_suite, strategy, agents, budget=6)
        assert isinstance(result, BenchmarkResult)
        assert result.strategy == strategy
        assert result.num_tasks == len(small_suite)
        assert 0.0 <= result.accuracy <= 1.0
        assert result.avg_tokens >= 0
        assert result.compute_efficiency >= 0
        assert result.total_tokens >= 0
        assert result.num_correct >= 0
        assert len(result.per_task_trace) == result.num_tasks

    def test_unknown_strategy_raises(self, small_suite, agents):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_benchmark(small_suite, "bogus", agents, budget=6)

    def test_zero_budget_raises(self, small_suite, agents):
        with pytest.raises(ValueError, match="budget must be positive"):
            run_benchmark(small_suite, "uniform", agents, budget=0)

    def test_empty_agents_raises(self, small_suite):
        with pytest.raises(ValueError, match="agents list must be non-empty"):
            run_benchmark(small_suite, "uniform", [], budget=6)

    def test_trace_contains_expected_keys(self, small_suite, agents):
        result = run_benchmark(small_suite, "capacity", agents, budget=6)
        for entry in result.per_task_trace:
            assert "task_id" in entry
            assert "kind" in entry
            assert "predicted" in entry
            assert "expected" in entry
            assert "correct" in entry
            assert "tokens" in entry


# ---- compare_strategies -----------------------------------------------------


class TestCompareStrategies:
    def test_returns_all_three(self, small_suite, agents):
        results = compare_strategies(small_suite, agents, budget=6)
        assert set(results.keys()) == set(STRATEGIES)
        for strat, res in results.items():
            assert res.strategy == strat
            assert isinstance(res, BenchmarkResult)

    def test_subset(self, small_suite, agents):
        results = compare_strategies(
            small_suite, agents, budget=6, strategies=["uniform", "capacity"]
        )
        assert set(results.keys()) == {"uniform", "capacity"}

    def test_empty_strategies_raises(self, small_suite, agents):
        with pytest.raises(ValueError):
            compare_strategies(small_suite, agents, budget=6, strategies=[])


# ---- build_specialist_agents ------------------------------------------------


class TestBuildSpecialistAgents:
    def test_produces_three_agents(self):
        agents = build_specialist_agents()
        assert len(agents) == 3

    def test_varied_capacities(self):
        agents = build_specialist_agents(high=0.9, low=0.1)
        # Each agent should excel at exactly one kind.
        for kind in TASK_KINDS:
            caps = [a.capacity_for(kind) for a in agents]
            assert max(caps) > 2 * min(caps), f"No specialist for {kind}"

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            build_specialist_agents(high=0.3, low=0.8)


# ---- BenchmarkResult.to_dict ------------------------------------------------


class TestBenchmarkResultToDict:
    def test_round_trip_fields(self, small_suite, agents):
        result = run_benchmark(small_suite, "uniform", agents, budget=6)
        d = result.to_dict()
        assert d["strategy"] == "uniform"
        assert d["num_tasks"] == result.num_tasks
        assert d["accuracy"] == result.accuracy
        assert "per_task_trace" not in d  # trace excluded from summary dict
