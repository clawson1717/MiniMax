"""Tests for raptor.experiments — evaluation harness, synthetic generators, and reporting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from raptor.agents import MockReasoningAgent, MockVariableAgent
from raptor.config import Config
from raptor.disagreement import AgentResponse
from raptor.experiments import (
    BaselineMode,
    BenchmarkTask,
    EvalResult,
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentReport,
    _answers_match,
    _compute_ece,
    _majority_vote,
    _normalize_answer,
    compute_metrics,
    generate_gsm8k_synthetic,
    generate_hotpotqa_synthetic,
    generate_math_synthetic,
    run_experiment,
    sweep_configs,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures — reusable mock agents & tasks
# ══════════════════════════════════════════════════════════════════════


def _make_correct_agent(agent_id: str, answer: str) -> MockReasoningAgent:
    """Create a mock agent that always returns *answer*."""
    return MockReasoningAgent(
        agent_id=agent_id,
        reasoning_steps=[
            "Step 1: Read the problem",
            "Step 2: Compute",
            "Step 3: Conclude",
        ],
        final_answer=answer,
    )


def _make_wrong_agent(agent_id: str) -> MockReasoningAgent:
    """Create a mock agent that always returns a wrong answer."""
    return MockReasoningAgent(
        agent_id=agent_id,
        reasoning_steps=["Step 1: I'm guessing"],
        final_answer="completely_incorrect_response",
    )


def _make_tasks(n: int = 5, answer: str = "42") -> list[BenchmarkTask]:
    """Create N tasks whose ground truth is *answer*."""
    return [
        BenchmarkTask(
            question=f"What is {i} + {42 - i}?",
            ground_truth=answer,
            dataset_name="test",
            task_id=f"test_{i:04d}",
        )
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════
# Synthetic Data Generator Tests
# ══════════════════════════════════════════════════════════════════════


class TestSyntheticGSM8K:
    """Tests for generate_gsm8k_synthetic()."""

    def test_generates_correct_count(self):
        tasks = generate_gsm8k_synthetic(10)
        assert len(tasks) == 10

    def test_zero_count(self):
        tasks = generate_gsm8k_synthetic(0)
        assert tasks == []

    def test_all_are_benchmark_tasks(self):
        tasks = generate_gsm8k_synthetic(20)
        for t in tasks:
            assert isinstance(t, BenchmarkTask)

    def test_dataset_name(self):
        tasks = generate_gsm8k_synthetic(5)
        for t in tasks:
            assert t.dataset_name == "gsm8k_synthetic"

    def test_unique_ids(self):
        tasks = generate_gsm8k_synthetic(50)
        ids = [t.task_id for t in tasks]
        assert len(set(ids)) == len(ids)

    def test_answers_are_numeric(self):
        tasks = generate_gsm8k_synthetic(25)
        for t in tasks:
            int(t.ground_truth)  # Should not raise

    def test_deterministic_with_seed(self):
        a = generate_gsm8k_synthetic(10, seed=123)
        b = generate_gsm8k_synthetic(10, seed=123)
        for x, y in zip(a, b):
            assert x.question == y.question
            assert x.ground_truth == y.ground_truth

    def test_different_seeds_differ(self):
        a = generate_gsm8k_synthetic(10, seed=1)
        b = generate_gsm8k_synthetic(10, seed=2)
        # Not all questions should be identical
        questions_a = [t.question for t in a]
        questions_b = [t.question for t in b]
        assert questions_a != questions_b

    def test_questions_nonempty(self):
        tasks = generate_gsm8k_synthetic(15)
        for t in tasks:
            assert len(t.question) > 20  # Reasonable length


class TestSyntheticMATH:
    """Tests for generate_math_synthetic()."""

    def test_generates_correct_count(self):
        tasks = generate_math_synthetic(12)
        assert len(tasks) == 12

    def test_dataset_name(self):
        tasks = generate_math_synthetic(5)
        for t in tasks:
            assert t.dataset_name == "math_synthetic"

    def test_unique_ids(self):
        tasks = generate_math_synthetic(30)
        ids = [t.task_id for t in tasks]
        assert len(set(ids)) == len(ids)

    def test_answers_are_numeric(self):
        tasks = generate_math_synthetic(20)
        for t in tasks:
            int(t.ground_truth)  # Should not raise

    def test_deterministic(self):
        a = generate_math_synthetic(10, seed=99)
        b = generate_math_synthetic(10, seed=99)
        for x, y in zip(a, b):
            assert x.ground_truth == y.ground_truth

    def test_covers_all_types(self):
        # 5 template types, so 5 tasks should cover all
        tasks = generate_math_synthetic(5)
        questions = [t.question.lower() for t in tasks]
        assert any("solve" in q for q in questions)  # algebra
        assert any("rectangle" in q or "area" in q for q in questions)  # geometry
        assert any("divisors" in q for q in questions)  # number theory
        assert any("choose" in q for q in questions)  # combinatorics
        assert any("arithmetic sequence" in q for q in questions)  # sequences


class TestSyntheticHotpotQA:
    """Tests for generate_hotpotqa_synthetic()."""

    def test_generates_correct_count(self):
        tasks = generate_hotpotqa_synthetic(8)
        assert len(tasks) == 8

    def test_dataset_name(self):
        tasks = generate_hotpotqa_synthetic(4)
        for t in tasks:
            assert t.dataset_name == "hotpotqa_synthetic"

    def test_unique_ids(self):
        tasks = generate_hotpotqa_synthetic(20)
        ids = [t.task_id for t in tasks]
        assert len(set(ids)) == len(ids)

    def test_answers_nonempty(self):
        tasks = generate_hotpotqa_synthetic(16)
        for t in tasks:
            assert t.ground_truth.strip() != ""

    def test_deterministic(self):
        a = generate_hotpotqa_synthetic(10, seed=7)
        b = generate_hotpotqa_synthetic(10, seed=7)
        for x, y in zip(a, b):
            assert x.ground_truth == y.ground_truth


# ══════════════════════════════════════════════════════════════════════
# Answer Matching Tests
# ══════════════════════════════════════════════════════════════════════


class TestAnswerMatching:
    """Tests for _normalize_answer, _answers_match, _majority_vote."""

    def test_normalize_strips_whitespace(self):
        assert _normalize_answer("  42  ") == "42"

    def test_normalize_lowercases(self):
        assert _normalize_answer("FoRtY Two") == "forty two"

    def test_normalize_strips_prefix(self):
        assert _normalize_answer("The answer is 42") == "42"
        assert _normalize_answer("Answer: 42") == "42"
        assert _normalize_answer("Result: 42") == "42"
        assert _normalize_answer("= 42") == "42"

    def test_normalize_strips_trailing_dot(self):
        assert _normalize_answer("42.") == "42"

    def test_exact_match(self):
        assert _answers_match("42", "42")

    def test_case_insensitive_match(self):
        assert _answers_match("FORTY TWO", "forty two")

    def test_numeric_match(self):
        assert _answers_match("42.0", "42")
        assert _answers_match("42.000000", "42")

    def test_substring_match(self):
        assert _answers_match("The final result is 42", "42")

    def test_no_match(self):
        assert not _answers_match("100", "42")
        assert not _answers_match("apple", "banana")

    def test_majority_vote_simple(self):
        assert _normalize_answer(_majority_vote(["42", "42", "100"])) == "42"

    def test_majority_vote_single(self):
        assert _majority_vote(["42"]) == "42"

    def test_majority_vote_empty(self):
        assert _majority_vote([]) == ""

    def test_majority_vote_all_different(self):
        result = _majority_vote(["a", "b", "c"])
        assert result in ("a", "b", "c")


# ══════════════════════════════════════════════════════════════════════
# Metrics Computation Tests
# ══════════════════════════════════════════════════════════════════════


class TestComputeMetrics:
    """Tests for compute_metrics() and ECE computation."""

    def test_empty_results(self):
        m = compute_metrics([])
        assert m.accuracy == 0.0
        assert m.ece == 0.0
        assert m.n_tasks == 0

    def test_all_correct(self):
        results = [
            EvalResult(
                task_id=f"t{i}",
                predicted="42",
                correct="42",
                is_correct=True,
                confidence=0.9,
                n_rerolls=0,
                n_steps=1,
                cost_estimate=1.0,
            )
            for i in range(10)
        ]
        m = compute_metrics(results)
        assert m.accuracy == 1.0
        assert m.n_correct == 10
        assert m.avg_rerolls == 0.0
        assert m.avg_steps == 1.0
        assert m.total_cost == 10.0

    def test_all_wrong(self):
        results = [
            EvalResult(
                task_id=f"t{i}",
                predicted="wrong",
                correct="42",
                is_correct=False,
                confidence=0.1,
            )
            for i in range(10)
        ]
        m = compute_metrics(results)
        assert m.accuracy == 0.0
        assert m.n_correct == 0

    def test_mixed_accuracy(self):
        results = [
            EvalResult(task_id="t0", predicted="42", correct="42", is_correct=True),
            EvalResult(task_id="t1", predicted="x", correct="42", is_correct=False),
            EvalResult(task_id="t2", predicted="42", correct="42", is_correct=True),
            EvalResult(task_id="t3", predicted="x", correct="42", is_correct=False),
        ]
        m = compute_metrics(results)
        assert m.accuracy == 0.5
        assert m.n_tasks == 4

    def test_ece_perfect_calibration(self):
        """ECE should be 0 when confidence matches accuracy within each bin."""
        results = [
            EvalResult(task_id="t0", predicted="a", correct="a", is_correct=True, confidence=1.0),
            EvalResult(task_id="t1", predicted="b", correct="c", is_correct=False, confidence=0.0),
        ]
        m = compute_metrics(results, n_bins=10)
        # With confidence=1 correct and confidence=0 wrong, ECE should be ~0
        assert m.ece < 0.1

    def test_ece_bad_calibration(self):
        """ECE should be high when confidence doesn't match accuracy."""
        results = [
            EvalResult(task_id=f"t{i}", predicted="wrong", correct="42",
                       is_correct=False, confidence=0.95)
            for i in range(10)
        ]
        m = compute_metrics(results, n_bins=10)
        # All predictions are wrong but confidence is ~0.95 → high ECE
        assert m.ece > 0.5

    def test_rerolls_and_steps_aggregation(self):
        results = [
            EvalResult(task_id="t0", predicted="a", correct="a", is_correct=True,
                       n_rerolls=2, n_steps=3, cost_estimate=5.0),
            EvalResult(task_id="t1", predicted="b", correct="b", is_correct=True,
                       n_rerolls=4, n_steps=7, cost_estimate=15.0),
        ]
        m = compute_metrics(results)
        assert m.avg_rerolls == 3.0
        assert m.avg_steps == 5.0
        assert m.total_cost == 20.0


class TestECE:
    """Focused tests for the _compute_ece helper."""

    def test_empty(self):
        assert _compute_ece([]) == 0.0

    def test_single_bin_perfect(self):
        results = [
            EvalResult(task_id="t0", predicted="a", correct="a", is_correct=True, confidence=0.55),
        ]
        # Single result: bin 5 (conf ~0.55), acc=1.0, gap=|1.0-0.55|=0.45
        ece = _compute_ece(results, n_bins=10)
        assert abs(ece - 0.45) < 0.01

    def test_ece_bounded_0_1(self):
        """ECE should always be between 0 and 1."""
        results = [
            EvalResult(task_id=f"t{i}", predicted="a", correct="a" if i % 2 == 0 else "b",
                       is_correct=(i % 2 == 0), confidence=i / 10.0)
            for i in range(10)
        ]
        ece = _compute_ece(results)
        assert 0.0 <= ece <= 1.0


# ══════════════════════════════════════════════════════════════════════
# Config Sweep Tests
# ══════════════════════════════════════════════════════════════════════


class TestSweepConfigs:
    """Tests for sweep_configs()."""

    def test_empty_grid(self):
        base = ExperimentConfig()
        configs = sweep_configs(base, {})
        assert len(configs) == 1

    def test_single_param(self):
        base = ExperimentConfig()
        configs = sweep_configs(base, {"n_agents": [3, 5, 7]})
        assert len(configs) == 3
        assert [c.n_agents for c in configs] == [3, 5, 7]

    def test_cartesian_product(self):
        base = ExperimentConfig()
        configs = sweep_configs(
            base,
            {"n_agents": [3, 5], "max_rerolls": [1, 3]},
        )
        assert len(configs) == 4
        combos = [(c.n_agents, c.raptor_config.max_rerolls) for c in configs]
        assert (3, 1) in combos
        assert (3, 3) in combos
        assert (5, 1) in combos
        assert (5, 3) in combos

    def test_baseline_mode_sweep(self):
        base = ExperimentConfig()
        configs = sweep_configs(
            base,
            {"baseline_mode": [BaselineMode.NAIVE, BaselineMode.RAPTOR_FULL]},
        )
        assert len(configs) == 2
        modes = {c.baseline_mode for c in configs}
        assert modes == {BaselineMode.NAIVE, BaselineMode.RAPTOR_FULL}

    def test_weight_sweep(self):
        base = ExperimentConfig()
        configs = sweep_configs(
            base,
            {"gain_weight": [0.1, 0.5], "severity_weight": [0.1, 0.3]},
        )
        assert len(configs) == 4
        for c in configs:
            assert c.raptor_config.utility.weights["gain"] in (0.1, 0.5)
            assert c.raptor_config.utility.weights["severity"] in (0.1, 0.3)

    def test_switch_threshold_sweep(self):
        base = ExperimentConfig()
        configs = sweep_configs(base, {"switch_threshold": [0.01, 0.05, 0.1]})
        thresholds = [c.raptor_config.utility.switch_threshold for c in configs]
        assert thresholds == [0.01, 0.05, 0.1]

    def test_monotonicity_threshold_sweep(self):
        base = ExperimentConfig()
        configs = sweep_configs(base, {"monotonicity_threshold": [0.5, 1.0]})
        vals = [c.raptor_config.entropy.monotonicity_threshold for c in configs]
        assert vals == [0.5, 1.0]

    def test_labels_generated(self):
        base = ExperimentConfig()
        configs = sweep_configs(base, {"n_agents": [3, 5]})
        for c in configs:
            assert "n_agents=" in c.label

    def test_no_cross_mutation(self):
        """Each config should be independent — mutations shouldn't leak."""
        base = ExperimentConfig()
        configs = sweep_configs(base, {"max_rerolls": [1, 5, 10]})
        configs[0].raptor_config.max_rerolls = 999
        assert configs[1].raptor_config.max_rerolls == 5
        assert configs[2].raptor_config.max_rerolls == 10


# ══════════════════════════════════════════════════════════════════════
# Experiment Runner Tests (per baseline mode)
# ══════════════════════════════════════════════════════════════════════


class TestRunExperimentNaive:
    """Test naive baseline mode."""

    def test_single_correct(self):
        agents = [_make_correct_agent("a1", "42")]
        tasks = _make_tasks(3, answer="42")
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=3)
        results = run_experiment(config, tasks, agents)
        assert len(results) == 3
        assert all(r.is_correct for r in results)
        assert all(r.n_rerolls == 0 for r in results)
        assert all(r.n_steps == 1 for r in results)

    def test_single_wrong(self):
        agents = [_make_wrong_agent("a1")]
        tasks = _make_tasks(2, answer="42")
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=2)
        results = run_experiment(config, tasks, agents)
        assert all(not r.is_correct for r in results)

    def test_cost_is_minimal(self):
        agents = [_make_correct_agent("a1", "42")]
        tasks = _make_tasks(1)
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE)
        results = run_experiment(config, tasks, agents)
        assert results[0].cost_estimate > 0
        # Naive should be cheapest — just 1 poll
        assert results[0].cost_estimate <= 2.0


class TestRunExperimentSelfConsistency:
    """Test self-consistency (majority vote) baseline mode."""

    def test_majority_wins(self):
        # 3 agents agree on "42", 2 agents say "wrong"
        agents = [
            _make_correct_agent(f"a{i}", "42") for i in range(3)
        ] + [_make_wrong_agent(f"w{i}") for i in range(2)]
        tasks = _make_tasks(2, answer="42")
        config = ExperimentConfig(
            baseline_mode=BaselineMode.SELF_CONSISTENCY, n_agents=5, n_samples=2
        )
        results = run_experiment(config, tasks, agents)
        assert all(r.is_correct for r in results)

    def test_unanimous_agreement(self):
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(5)]
        tasks = _make_tasks(1)
        config = ExperimentConfig(baseline_mode=BaselineMode.SELF_CONSISTENCY, n_agents=5)
        results = run_experiment(config, tasks, agents)
        assert results[0].is_correct
        assert results[0].confidence == 1.0  # All agree

    def test_all_wrong(self):
        agents = [_make_wrong_agent(f"a{i}") for i in range(5)]
        tasks = _make_tasks(1, answer="42")
        config = ExperimentConfig(baseline_mode=BaselineMode.SELF_CONSISTENCY, n_agents=5)
        results = run_experiment(config, tasks, agents)
        assert not results[0].is_correct


class TestRunExperimentDiscoUQOnly:
    """Test DiscoUQ-only baseline mode."""

    def test_correct_with_agreement(self):
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(5)]
        tasks = _make_tasks(1)
        config = ExperimentConfig(baseline_mode=BaselineMode.DISCOUQ_ONLY, n_agents=5)
        results = run_experiment(config, tasks, agents)
        assert results[0].is_correct

    def test_has_signals_history(self):
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(3)]
        tasks = _make_tasks(1)
        config = ExperimentConfig(baseline_mode=BaselineMode.DISCOUQ_ONLY, n_agents=3)
        results = run_experiment(config, tasks, agents)
        # DiscoUQ should record disagreement signal
        assert len(results[0].signals_history) > 0

    def test_confidence_in_range(self):
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(3)]
        tasks = _make_tasks(1)
        config = ExperimentConfig(baseline_mode=BaselineMode.DISCOUQ_ONLY)
        results = run_experiment(config, tasks, agents)
        assert 0.0 <= results[0].confidence <= 1.0


class TestRunExperimentRaptorFull:
    """Test full RAPTOR pipeline mode."""

    def test_completes_with_agents(self):
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(3)]
        tasks = _make_tasks(1, answer="42")
        cfg = Config(max_steps=3, log_signal_history=False)
        config = ExperimentConfig(
            raptor_config=cfg,
            baseline_mode=BaselineMode.RAPTOR_FULL,
            n_agents=3,
        )
        results = run_experiment(config, tasks, agents)
        assert len(results) == 1
        # Should at least have attempted an answer
        assert results[0].predicted != ""

    def test_records_steps(self):
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(3)]
        tasks = _make_tasks(1)
        cfg = Config(max_steps=5, log_signal_history=False)
        config = ExperimentConfig(
            raptor_config=cfg,
            baseline_mode=BaselineMode.RAPTOR_FULL,
        )
        results = run_experiment(config, tasks, agents)
        assert results[0].n_steps >= 1


class TestRunExperimentInvalid:
    """Test error handling in run_experiment."""

    def test_unknown_mode(self):
        # Manually construct a config with an invalid mode
        config = ExperimentConfig()
        config.baseline_mode = "bogus"  # type: ignore
        tasks = _make_tasks(1)
        agents = [_make_correct_agent("a1", "42")]
        with pytest.raises(ValueError, match="Unknown baseline mode"):
            run_experiment(config, tasks, agents)


class TestRunExperimentSampling:
    """Test n_samples parameter in run_experiment."""

    def test_samples_limits_tasks(self):
        agents = [_make_correct_agent("a1", "42")]
        tasks = _make_tasks(10)
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=3)
        results = run_experiment(config, tasks, agents)
        assert len(results) == 3

    def test_zero_samples_runs_all(self):
        agents = [_make_correct_agent("a1", "42")]
        tasks = _make_tasks(5)
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=0)
        results = run_experiment(config, tasks, agents)
        assert len(results) == 5

    def test_on_result_callback(self):
        called = []
        agents = [_make_correct_agent("a1", "42")]
        tasks = _make_tasks(3)
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=3)
        run_experiment(config, tasks, agents, on_result=lambda r, i: called.append((r.task_id, i)))
        assert len(called) == 3
        assert called[0][1] == 0
        assert called[2][1] == 2


# ══════════════════════════════════════════════════════════════════════
# Experiment Report Tests
# ══════════════════════════════════════════════════════════════════════


class TestExperimentReport:
    """Tests for ExperimentReport generation."""

    def _sample_report(self) -> ExperimentReport:
        report = ExperimentReport()
        for mode in [BaselineMode.NAIVE, BaselineMode.SELF_CONSISTENCY, BaselineMode.RAPTOR_FULL]:
            config = ExperimentConfig(baseline_mode=mode, label=mode.value)
            metrics = ExperimentMetrics(
                accuracy=0.6 + 0.1 * list(BaselineMode).index(mode),
                ece=0.1 - 0.02 * list(BaselineMode).index(mode),
                avg_rerolls=float(list(BaselineMode).index(mode)),
                avg_steps=float(list(BaselineMode).index(mode) + 1),
                total_cost=10.0 * (list(BaselineMode).index(mode) + 1),
                n_tasks=100,
                n_correct=int(100 * (0.6 + 0.1 * list(BaselineMode).index(mode))),
            )
            report.add_result(config, metrics)
        return report

    def test_add_result(self):
        report = ExperimentReport()
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, label="test_naive")
        metrics = ExperimentMetrics(
            accuracy=0.8, ece=0.05, avg_rerolls=0.0,
            avg_steps=1.0, total_cost=10.0, n_tasks=50, n_correct=40,
        )
        report.add_result(config, metrics)
        assert len(report.entries) == 1
        assert report.entries[0]["accuracy"] == 0.8

    def test_empty_markdown(self):
        report = ExperimentReport()
        md = report.to_markdown()
        assert "No experiment results" in md

    def test_markdown_has_table(self):
        report = self._sample_report()
        md = report.to_markdown()
        assert "| Config |" in md
        assert "naive" in md
        assert "raptor_full" in md

    def test_markdown_has_summary(self):
        report = self._sample_report()
        md = report.to_markdown()
        assert "Best accuracy" in md
        assert "Lowest cost" in md
        assert "Best calibration" in md

    def test_to_json_valid(self):
        report = self._sample_report()
        j = report.to_json()
        data = json.loads(j)
        assert "experiments" in data
        assert len(data["experiments"]) == 3

    def test_to_dict(self):
        report = self._sample_report()
        d = report.to_dict()
        assert len(d["experiments"]) == 3

    def test_single_entry_no_summary(self):
        report = ExperimentReport()
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, label="single")
        metrics = ExperimentMetrics(
            accuracy=0.9, ece=0.02, avg_rerolls=0.0,
            avg_steps=1.0, total_cost=5.0, n_tasks=10, n_correct=9,
        )
        report.add_result(config, metrics)
        md = report.to_markdown()
        # With only 1 entry, no summary section
        assert "Best accuracy" not in md


# ══════════════════════════════════════════════════════════════════════
# End-to-end Integration Tests
# ══════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """End-to-end tests combining synthetic data + experiment runner + metrics."""

    def test_gsm8k_naive_pipeline(self):
        """Run naive mode on synthetic GSM8K and compute metrics."""
        tasks = generate_gsm8k_synthetic(5)
        # Agent always says the right answer for the first task
        agents = [_make_correct_agent("solver", tasks[0].ground_truth)]
        config = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=5)
        results = run_experiment(config, tasks, agents)
        metrics = compute_metrics(results)
        assert metrics.n_tasks == 5
        # At least the first task should match (agent returns answer for task 0)
        assert metrics.n_correct >= 1

    def test_full_pipeline_report(self):
        """Run multiple modes, compute metrics, and generate a report."""
        tasks = _make_tasks(3, answer="42")
        agents = [_make_correct_agent(f"a{i}", "42") for i in range(3)]
        report = ExperimentReport()

        for mode in [BaselineMode.NAIVE, BaselineMode.SELF_CONSISTENCY]:
            config = ExperimentConfig(
                baseline_mode=mode, n_agents=3, n_samples=3, label=mode.value
            )
            results = run_experiment(config, tasks, agents)
            metrics = compute_metrics(results)
            report.add_result(config, metrics)

        md = report.to_markdown()
        assert "naive" in md
        assert "self_consistency" in md
        assert len(report.entries) == 2

    def test_sweep_and_run(self):
        """Sweep over configs and run experiments."""
        tasks = _make_tasks(2, answer="42")
        agents = [_make_correct_agent("a1", "42")]
        base = ExperimentConfig(baseline_mode=BaselineMode.NAIVE, n_samples=2)
        configs = sweep_configs(base, {"n_agents": [1, 3]})
        assert len(configs) == 2

        report = ExperimentReport()
        for cfg in configs:
            results = run_experiment(cfg, tasks, agents)
            metrics = compute_metrics(results)
            report.add_result(cfg, metrics)

        assert len(report.entries) == 2

    def test_math_self_consistency(self):
        """Self-consistency on MATH-synthetic with multiple agreeing agents."""
        tasks = generate_math_synthetic(3)
        # Use same answer agent for all tasks (won't be correct for all)
        agents = [_make_correct_agent(f"a{i}", tasks[0].ground_truth) for i in range(5)]
        config = ExperimentConfig(
            baseline_mode=BaselineMode.SELF_CONSISTENCY, n_agents=5, n_samples=3
        )
        results = run_experiment(config, tasks, agents)
        metrics = compute_metrics(results)
        assert metrics.n_tasks == 3
        # First task should be correct since all agents give its answer
        assert results[0].is_correct
