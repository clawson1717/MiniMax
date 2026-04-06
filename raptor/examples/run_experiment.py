#!/usr/bin/env python3
"""Full experiment with synthetic data, multiple baselines, and metrics comparison.

This example demonstrates RAPTOR's experiment harness:
  1. Generate synthetic benchmark tasks (GSM8K, MATH, HotpotQA)
  2. Run all four baseline modes
  3. Compute metrics (accuracy, ECE, cost)
  4. Generate a comparison report
  5. Run a hyperparameter sweep

Everything uses MockReasoningAgent — no API keys needed.

Run with:
    python examples/run_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from raptor.agents import MockReasoningAgent
from raptor.config import Config
from raptor.experiments import (
    BaselineMode,
    BenchmarkTask,
    ExperimentConfig,
    ExperimentReport,
    compute_metrics,
    generate_gsm8k_synthetic,
    generate_math_synthetic,
    generate_hotpotqa_synthetic,
    run_experiment,
    sweep_configs,
)


def create_mock_agents(n: int = 5) -> list[MockReasoningAgent]:
    """Create a set of mock agents with slight answer variation.

    Most agents agree on a common answer pattern, but one agent
    diverges — creating realistic disagreement for RAPTOR to analyze.
    """
    agents = []
    for i in range(n):
        if i < n - 1:
            # Majority agents — give a plausible answer
            agents.append(
                MockReasoningAgent(
                    agent_id=f"agent-{i}",
                    reasoning_steps=[
                        "Step 1: Identify the key quantities in the problem.",
                        "Step 2: Apply the relevant mathematical operation.",
                        "Step 3: Compute the final result.",
                    ],
                    final_answer="42",
                )
            )
        else:
            # Divergent agent — gives a different answer
            agents.append(
                MockReasoningAgent(
                    agent_id=f"agent-{i}-divergent",
                    reasoning_steps=[
                        "Step 1: Read the problem.",
                        "Step 2: I think the answer is different.",
                    ],
                    final_answer="36",
                )
            )
    return agents


def demo_data_generators() -> None:
    """Show the three synthetic data generators."""
    print("=" * 60)
    print("1. Synthetic Data Generators")
    print("=" * 60)

    # GSM8K-style tasks
    gsm_tasks = generate_gsm8k_synthetic(5, seed=42)
    print(f"\nGSM8K synthetic ({len(gsm_tasks)} tasks):")
    for t in gsm_tasks:
        print(f"  [{t.task_id}] Q: {t.question[:70]}...")
        print(f"           A: {t.ground_truth}")

    # MATH-style tasks
    math_tasks = generate_math_synthetic(5, seed=42)
    print(f"\nMATH synthetic ({len(math_tasks)} tasks):")
    for t in math_tasks:
        print(f"  [{t.task_id}] Q: {t.question[:70]}...")
        print(f"           A: {t.ground_truth}")

    # HotpotQA-style tasks
    hotpot_tasks = generate_hotpotqa_synthetic(5, seed=42)
    print(f"\nHotpotQA synthetic ({len(hotpot_tasks)} tasks):")
    for t in hotpot_tasks:
        print(f"  [{t.task_id}] Q: {t.question[:70]}...")
        print(f"           A: {t.ground_truth}")
    print()


def demo_baseline_comparison() -> None:
    """Run all four baselines on the same dataset and compare."""
    print("=" * 60)
    print("2. Baseline Comparison (GSM8K, 20 tasks)")
    print("=" * 60)

    tasks = generate_gsm8k_synthetic(20, seed=42)
    agents = create_mock_agents(5)
    report = ExperimentReport()

    for mode in BaselineMode:
        config = ExperimentConfig(
            baseline_mode=mode,
            n_agents=5,
            n_samples=20,
            dataset="gsm8k_synthetic",
            label=mode.value,
        )

        # Run with progress indicator
        results = run_experiment(config, tasks, agents)
        metrics = compute_metrics(results)
        report.add_result(config, metrics)

        correct = sum(1 for r in results if r.is_correct)
        print(
            f"  {mode.value:20s} │ "
            f"acc={metrics.accuracy:5.1%} │ "
            f"ECE={metrics.ece:.4f} │ "
            f"rerolls={metrics.avg_rerolls:.1f} │ "
            f"steps={metrics.avg_steps:.1f} │ "
            f"cost={metrics.total_cost:7.1f}"
        )

    print()
    print("Markdown report:")
    print("-" * 40)
    print(report.to_markdown())
    print()


def demo_per_task_analysis() -> None:
    """Show per-task details from a RAPTOR_FULL run."""
    print("=" * 60)
    print("3. Per-Task Analysis (RAPTOR_FULL, 10 tasks)")
    print("=" * 60)

    tasks = generate_gsm8k_synthetic(10, seed=99)
    agents = create_mock_agents(5)

    config = ExperimentConfig(
        baseline_mode=BaselineMode.RAPTOR_FULL,
        n_agents=5,
        n_samples=10,
    )

    results = run_experiment(config, tasks, agents)

    for r in results:
        status = "✓" if r.is_correct else "✗"
        print(
            f"  {status} {r.task_id:20s} │ "
            f"pred={r.predicted[:15]:15s} │ "
            f"truth={r.correct[:10]:10s} │ "
            f"conf={r.confidence:.2f} │ "
            f"steps={r.n_steps} │ "
            f"rerolls={r.n_rerolls}"
        )

    metrics = compute_metrics(results)
    print(f"\n  Summary: {metrics.n_correct}/{metrics.n_tasks} correct "
          f"({metrics.accuracy:.0%}), ECE={metrics.ece:.4f}")
    print()


def demo_hyperparameter_sweep() -> None:
    """Run a small hyperparameter sweep over key parameters."""
    print("=" * 60)
    print("4. Hyperparameter Sweep")
    print("=" * 60)

    tasks = generate_gsm8k_synthetic(15, seed=42)
    agents = create_mock_agents(5)

    base = ExperimentConfig(
        baseline_mode=BaselineMode.RAPTOR_FULL,
        n_samples=15,
        dataset="gsm8k_synthetic",
    )

    # Sweep over max_rerolls and n_agents
    configs = sweep_configs(base, {
        "max_rerolls": [1, 3, 5],
        "n_agents": [3, 5],
    })

    print(f"  Generated {len(configs)} sweep configurations\n")

    report = ExperimentReport()
    for cfg in configs:
        results = run_experiment(cfg, tasks, agents[:cfg.n_agents])
        metrics = compute_metrics(results)
        report.add_result(cfg, metrics)

        print(
            f"  {cfg.label:40s} │ "
            f"acc={metrics.accuracy:5.1%} │ "
            f"cost={metrics.total_cost:7.1f}"
        )

    print()

    # Show the best configuration
    best = max(report.entries, key=lambda e: e["accuracy"])
    cheapest = min(report.entries, key=lambda e: e["total_cost"])
    print(f"  Best accuracy:  {best['label']} ({best['accuracy']:.1%})")
    print(f"  Lowest cost:    {cheapest['label']} (cost={cheapest['total_cost']:.1f})")
    print()


def demo_multi_dataset() -> None:
    """Run experiments across multiple datasets."""
    print("=" * 60)
    print("5. Multi-Dataset Experiment")
    print("=" * 60)

    datasets = {
        "gsm8k":    generate_gsm8k_synthetic(15, seed=42),
        "math":     generate_math_synthetic(15, seed=42),
        "hotpotqa": generate_hotpotqa_synthetic(15, seed=42),
    }

    agents = create_mock_agents(5)
    report = ExperimentReport()

    for name, tasks in datasets.items():
        config = ExperimentConfig(
            baseline_mode=BaselineMode.RAPTOR_FULL,
            n_agents=5,
            n_samples=15,
            dataset=name,
            label=f"raptor-{name}",
        )

        results = run_experiment(config, tasks, agents)
        metrics = compute_metrics(results)
        report.add_result(config, metrics)

        print(
            f"  {name:12s} │ "
            f"acc={metrics.accuracy:5.1%} │ "
            f"ECE={metrics.ece:.4f} │ "
            f"cost={metrics.total_cost:7.1f}"
        )

    print()

    # Export as JSON
    json_output = report.to_json(indent=2)
    print(f"  JSON export ({len(json_output)} bytes):")
    # Print first few lines
    for line in json_output.split("\n")[:10]:
        print(f"    {line}")
    print("    ...")
    print()


def main() -> None:
    """Run all experiment demos."""
    print()
    print("RAPTOR Experiment Harness Demo")
    print("=" * 60)
    print("All examples use MockReasoningAgent (no API keys needed).")
    print()

    demo_data_generators()
    demo_baseline_comparison()
    demo_per_task_analysis()
    demo_hyperparameter_sweep()
    demo_multi_dataset()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
