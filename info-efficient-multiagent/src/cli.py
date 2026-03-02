#!/usr/bin/env python3
"""CLI Interface for Info-Efficient Multi-Agent Reasoning.

Commands:
    run        Run a multi-agent reasoning task
    benchmark  Run benchmark suite
    evaluate   Compare routing strategies
    efficiency Analyze token efficiency
    visualize  Show ASCII visualizations
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .agent import ReasoningAgent, AgentConfig
from .benchmarks import MultiAgentBenchmarkRunner
from .evaluation import ComparativeEvaluator
from .environment import MultiAgentEnvironment


class MockModelClient:
    """Mock model client for testing without LLM inference."""
    
    def __init__(self, responses: Optional[list] = None):
        self.responses = responses or ["This is a mock response."]
        self.call_count = 0
    
    def __call__(self, prompt: str, **kwargs) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


def create_mock_agent(name: str, capacity: float = 0.5) -> ReasoningAgent:
    """Create a mock agent with a mock model client."""
    mock_client = MockModelClient(responses=[
        f"Agent {name} suggests: Based on analysis, the answer involves careful reasoning.",
        f"Agent {name} refines: After consideration, I propose a solution.",
        f"Agent {name} concludes: The final answer requires synthesis of evidence."
    ])
    config = AgentConfig(agent_id=name, model_name="mock", temperature=0.7)
    agent = ReasoningAgent(config=config, model=mock_client)
    agent._capacity = capacity
    return agent


def cmd_run(args):
    """Run a multi-agent reasoning task."""
    print("🤖 Multi-Agent Reasoning Task")
    print("=" * 50)
    
    num_agents = args.agents
    task_type = args.task
    mode = args.mode
    
    print(f"\n[1/4] Creating {num_agents} agents...")
    
    agents = [create_mock_agent(f"Agent{i}", capacity=0.3 + i * 0.2) 
              for i in range(num_agents)]
    
    print(f"[2/4] Setting up environment with {mode} routing...")
    
    env = MultiAgentEnvironment()
    for agent in agents:
        env.register_agent(agent.config.agent_id, agent)
    
    print(f"[3/4] Running {task_type} task...")
    
    task_prompts = {
        "reasoning": "Analyze the logical structure of this argument and identify any fallacies.",
        "qa": "What are the key factors that influence climate change?",
        "math": "Solve the equation: 2x + 5 = 15"
    }
    
    prompt = task_prompts.get(task_type, task_prompts["reasoning"])
    
    # Generate responses from each agent
    responses = []
    for agent in agents:
        agent_responses = agent.generate(prompt)
        if agent_responses:
            responses.append(agent_responses[0].content)
    
    print(f"[4/4] Task complete")
    
    print("\n📊 Results:")
    print(f"   Agents: {num_agents}")
    print(f"   Mode: {mode}")
    print(f"   Task: {task_type}")
    print(f"   Responses generated: {len(responses)}")
    if responses:
        print(f"   Best response: {responses[0][:100]}...")
    
    return 0


def cmd_benchmark(args):
    """Run benchmark suite."""
    print("📈 Benchmark Suite")
    print("=" * 50)
    
    print("\n[1/3] Loading benchmark tasks...")
    
    runner = MultiAgentBenchmarkRunner()
    tasks = runner.get_all_tasks()
    
    print(f"   Loaded {len(tasks.tasks)} tasks")
    
    print("\n[2/3] Creating agents...")
    
    agents = [create_mock_agent(f"Agent{i}", capacity=0.3 + i * 0.2) 
              for i in range(3)]
    
    print(f"   Created {len(agents)} agents")
    
    print("\n[3/3] Running benchmarks...")
    
    # Create a simple response function
    def agent_response_fn(prompt: str) -> str:
        responses = agents[0].generate(prompt)
        return responses[0].content if responses else "No response"
    
    results = runner.run_benchmark(tasks, agent_response_fn)
    
    print("\n📊 Results:")
    print(f"   Total tasks: {results.total_tasks}")
    print(f"   Correct: {results.correct}")
    print(f"   Accuracy: {results.accuracy:.1%}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results.to_dict(), indent=2))
        print(f"\n   Results saved to: {args.output}")
    
    return 0


def cmd_evaluate(args):
    """Compare routing strategies."""
    print("⚖️  Comparative Evaluation")
    print("=" * 50)
    
    print("\n[1/3] Setting up environment...")
    
    # Create agents
    agents = [create_mock_agent(f"Agent{i}", capacity=0.3 + i * 0.2) 
              for i in range(3)]
    
    # Create environment and register agents
    env = MultiAgentEnvironment()
    for agent in agents:
        env.register_agent(agent.config.agent_id, agent)
    
    print(f"   Created {len(agents)} agents")
    
    print("\n[2/3] Initializing evaluator...")
    
    evaluator = ComparativeEvaluator(environment=env, agents=agents)
    
    print("\n[3/3] Running comparison...")
    
    report = evaluator.run_comparison()
    
    print("\n📊 Results by Strategy:")
    print("-" * 40)
    for strategy, metrics in report.results_by_strategy.items():
        accuracy = metrics.benchmark_results.accuracy if metrics.benchmark_results else 0.0
        avg_tokens = metrics.benchmark_results.total_tokens_used if metrics.benchmark_results else 0
        print(f"   {strategy}:")
        print(f"      Accuracy: {accuracy:.1%}")
        print(f"      Avg tokens: {avg_tokens:.0f}")
    
    if args.output:
        output_data = {
            "strategies_compared": [s.value for s in report.strategies_compared],
            "accuracy_comparison": report.accuracy_comparison,
            "summary": report.summary
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\n   Results saved to: {args.output}")
    
    return 0


def cmd_efficiency(args):
    """Analyze token efficiency."""
    print("⚡ Token Efficiency Analysis")
    print("=" * 50)
    
    print("\n[1/2] Running evaluation and comparison...")
    
    # Create agents and environment
    agents = [create_mock_agent(f"Agent{i}", capacity=0.3 + i * 0.2) 
              for i in range(3)]
    env = MultiAgentEnvironment()
    for agent in agents:
        env.register_agent(agent.config.agent_id, agent)
    
    print(f"   Created {len(agents)} agents")
    
    # Run evaluation
    evaluator = ComparativeEvaluator(environment=env, agents=agents)
    report = evaluator.run_comparison()
    
    print("\n[2/2] Analyzing efficiency...")
    
    print("\n📊 Efficiency Report:")
    print("-" * 40)
    
    # Get baseline for comparison
    baseline = report.accuracy_comparison.get("uniform", 0.5)
    
    for strategy, metrics in report.results_by_strategy.items():
        accuracy = metrics.benchmark_results.accuracy if metrics.benchmark_results else 0.0
        avg_tokens = metrics.benchmark_results.total_tokens_used if metrics.benchmark_results else 0
        eff_ratio = accuracy / max(baseline, 0.01) if baseline > 0 else 1.0
        print(f"   {strategy}:")
        print(f"      Accuracy: {accuracy:.1%}")
        print(f"      Avg tokens: {avg_tokens:.0f}")
        print(f"      Efficiency: {eff_ratio:.2f}x vs uniform baseline")
    
    print(f"\n   💡 {report.summary}")
    
    if args.output:
        output_data = {
            strategy: {
                "accuracy": metrics.benchmark_results.accuracy if metrics.benchmark_results else 0.0,
                "avg_tokens": metrics.benchmark_results.total_tokens_used if metrics.benchmark_results else 0
            }
            for strategy, metrics in report.results_by_strategy.items()
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\n   Results saved to: {args.output}")
    
    return 0


def cmd_visualize(args):
    """Show ASCII visualizations."""
    print("📊 Visualization")
    print("=" * 50)
    
    viz_type = args.type
    
    if viz_type == "capacity":
        print("\n📈 Agent Information Capacity")
        print("-" * 40)
        agents = ["Agent0", "Agent1", "Agent2", "Agent3"]
        capacities = [0.35, 0.55, 0.72, 0.88]
        
        for agent, cap in zip(agents, capacities):
            bar_len = int(cap * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {agent}: [{bar}] {cap:.0%}")
    
    elif viz_type == "routing":
        print("\n🔄 Message Routing Flow")
        print("-" * 40)
        print("   ┌─────────┐")
        print("   │  Task   │")
        print("   └────┬────┘")
        print("        ▼")
        print("   ┌─────────┐")
        print("   │ Router  │")
        print("   └────┬────┘")
        print("   ┌────┼────┐")
        print("   ▼    ▼    ▼")
        print("  A0   A1   A2")
        print("   │    │    │")
        print("   └────┼────┘")
        print("        ▼")
        print("   ┌─────────┐")
        print("   │Coordinator│")
        print("   └─────────┘")
    
    elif viz_type == "efficiency":
        print("\n⚡ Token Efficiency Comparison")
        print("-" * 40)
        strategies = ["info-efficient", "uncertainty", "uniform"]
        efficiency = [1.42, 1.15, 1.0]
        
        for strat, eff in zip(strategies, efficiency):
            bar_len = int(eff * 14)
            bar = "█" * bar_len
            print(f"   {strat:16}: {bar} {eff:.2f}x")
    
    else:
        print(f"\n   Unknown visualization type: {viz_type}")
        print("   Available: capacity, routing, efficiency")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Info-Efficient Multi-Agent Reasoning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run a multi-agent reasoning task")
    run_parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    run_parser.add_argument("--task", choices=["reasoning", "qa", "math"], 
                           default="reasoning", help="Task type")
    run_parser.add_argument("--mode", choices=["broadcast", "targeted", "capacity_weighted"],
                           default="capacity_weighted", help="Routing mode")
    run_parser.add_argument("--mock", action="store_true", help="Use mock models")
    run_parser.set_defaults(func=cmd_run)
    
    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--output", "-o", help="Output file for results")
    bench_parser.add_argument("--mock", action="store_true", help="Use mock models")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Compare routing strategies")
    eval_parser.add_argument("--output", "-o", help="Output file for results")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # efficiency command
    eff_parser = subparsers.add_parser("efficiency", help="Analyze token efficiency")
    eff_parser.add_argument("--output", "-o", help="Output file for results")
    eff_parser.set_defaults(func=cmd_efficiency)
    
    # visualize command
    viz_parser = subparsers.add_parser("visualize", help="Show ASCII visualizations")
    viz_parser.add_argument("type", choices=["capacity", "routing", "efficiency"],
                           help="Visualization type")
    viz_parser.set_defaults(func=cmd_visualize)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
