#!/usr/bin/env python3
"""
RCF CLI - Robust Continual Flow Command Line Interface

A CLI tool to visualize trajectory graphs, uncertainty scores, and fatigue monitors.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.trajectory import TrajectoryLogger
from src.uncertainty import UncertaintyEstimator
from src.monitor import FailureMonitor
from src.benchmark import run_quick_benchmark
from src.scaling_fatigue_analysis import (
    ScalingFatigueAnalyzer,
    SessionMetrics,
    run_analysis_simulation,
)


class Dashboard:
    """Real-time dashboard for RCF agent monitoring."""
    
    def __init__(self, refresh_rate: float = 0.5):
        """Initialize dashboard."""
        self.refresh_rate = refresh_rate
        self.trajectory = TrajectoryLogger()
        self.monitor = FailureMonitor()
        self.uncertainty_history: List[float] = []
        self.fatigue_history: List[float] = []
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")
    
    def render_header(self, title: str = "RCF Dashboard") -> None:
        """Render dashboard header."""
        width = 60
        print("=" * width)
        print(f"  {title.center(width - 4)}")
        print("=" * width)
        print()
    
    def render_metrics(
        self,
        step: int,
        max_steps: int,
        uncertainty: float,
        fatigue: float,
        compute_budget: float,
    ) -> None:
        """Render current metrics panel."""
        print("┌" + "─" * 58 + "┐")
        print(f"│ {'Step':<15} {step:>5}/{max_steps:<5} {'Progress':>15} {'▓' * int(step/max_steps*20):<20} │")
        print(f"│ {'Uncertainty':<15} {uncertainty:>10.3f} {'Low' if uncertainty < 0.3 else 'Med' if uncertainty < 0.6 else 'High':>15} │")
        print(f"│ {'Fatigue':<15} {fatigue:>10.3f} {'OK' if fatigue < 0.5 else 'Warning' if fatigue < 0.7 else 'Critical':>15} │")
        print(f"│ {'Compute Budget':<15} {compute_budget:>10.1f} {'tokens/step':>15} │")
        print("└" + "─" * 58 + "┘")
        print()
    
    def render_graph(
        self,
        data: List[float],
        title: str,
        width: int = 50,
        height: int = 8,
    ) -> None:
        """Render ASCII graph of data."""
        if not data:
            print(f"  {title}: No data yet")
            return
        
        print(f"  {title}")
        print("  " + "┌" + "─" * width + "┐")
        
        # Normalize data
        max_val = max(data) if max(data) > 0 else 1
        min_val = min(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Create rows
        for row in range(height - 1, -1, -1):
            threshold = min_val + (range_val * row / (height - 1))
            line = "  │"
            for val in data[-width:]:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            line += " " * (width - len(data[-width:]))
            line += "│"
            print(line)
        
        print("  " + "└" + "─" * width + "┘")
        print(f"  Min: {min_val:.3f}  Max: {max_val:.3f}  Current: {data[-1]:.3f}")
        print()
    
    def render_trajectory_summary(self) -> None:
        """Render trajectory graph summary."""
        traj_dict = self.trajectory.to_dict()
        nodes = traj_dict.get("nodes", [])
        edges = traj_dict.get("edges", [])
        
        print("  Trajectory Graph")
        print("  " + "─" * 40)
        print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")
        
        if nodes:
            print(f"  Last node: {nodes[-1].get('id', 'unknown')}")
        
        print()
    
    def render_alerts(self, alerts: List[str]) -> None:
        """Render alert panel."""
        if not alerts:
            return
        
        print("  ⚠️  Alerts")
        print("  " + "─" * 40)
        for alert in alerts[-5:]:  # Show last 5
            print(f"  • {alert}")
        print()
    
    def run_demo(self, steps: int = 20) -> None:
        """Run a demo simulation of the dashboard."""
        import random
        
        self.clear_screen()
        self.render_header("RCF Dashboard - Demo Mode")
        
        print("Running simulation...")
        time.sleep(0.5)
        
        for step in range(1, steps + 1):
            self.clear_screen()
            self.render_header(f"RCF Dashboard - Step {step}/{steps}")
            
            # Simulate metrics
            uncertainty = random.uniform(0.1, 0.8)
            fatigue = min(0.9, 0.1 + step * 0.04 + random.uniform(-0.1, 0.1))
            compute = 3 + (1 - uncertainty) * 4
            
            self.uncertainty_history.append(uncertainty)
            self.fatigue_history.append(fatigue)
            
            # Render panels
            self.render_metrics(step, steps, uncertainty, fatigue, compute)
            self.render_graph(self.uncertainty_history, "Uncertainty History")
            self.render_graph(self.fatigue_history, "Fatigue History")
            self.render_trajectory_summary()
            
            # Generate alerts
            alerts = []
            if fatigue > 0.7:
                alerts.append(f"High fatigue detected: {fatigue:.2f}")
            if uncertainty > 0.6:
                alerts.append(f"High uncertainty: {uncertainty:.2f}")
            self.render_alerts(alerts)
            
            print(f"  Refresh rate: {self.refresh_rate}s | Press Ctrl+C to exit")
            
            time.sleep(self.refresh_rate)
        
        self.clear_screen()
        self.render_header("Demo Complete")
        print(f"  Final fatigue: {self.fatigue_history[-1]:.3f}")
        print(f"  Avg uncertainty: {sum(self.uncertainty_history)/len(self.uncertainty_history):.3f}")
        print()


def cmd_benchmark(args) -> None:
    """Run benchmark command."""
    print("Running RCF Benchmark...")
    result = run_quick_benchmark(model_quality=args.quality, verbose=True)
    print(f"\nSuccess rate: {result['summary']['success_rate']:.1%}")
    print(f"Avg tokens/turn: {result['summary']['avg_tokens_per_turn']:.1f}")


def cmd_analyze(args) -> None:
    """Run analysis command."""
    print(f"Running Scaling vs. Fatigue Analysis ({args.sessions} sessions)...")
    
    analyzer = ScalingFatigueAnalyzer()
    report = run_analysis_simulation(num_sessions=args.sessions)
    
    print(f"\nResults:")
    print(f"  Total sessions: {report['summary']['total_sessions']}")
    print(f"  Fatigue rate: {report['summary']['fatigue_rate']:.1%}")
    
    if report['summary']['avg_fatigue_onset_step']:
        print(f"  Avg fatigue onset: step {report['summary']['avg_fatigue_onset_step']:.1f}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")


def cmd_dashboard(args) -> None:
    """Run dashboard command."""
    dashboard = Dashboard(refresh_rate=args.refresh)
    
    try:
        dashboard.run_demo(steps=args.steps)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


def cmd_evaluate(args) -> None:
    """Run evaluation command."""
    from src.evaluation import Evaluator
    
    print(f"Running evaluation on domain: {args.domain or 'all'}")
    
    evaluator = Evaluator()
    benchmarks = evaluator.load_benchmarks()
    
    if args.domain:
        if args.domain in benchmarks:
            evaluator.run_evaluation(args.domain, benchmarks[args.domain])
        else:
            print(f"Unknown domain: {args.domain}")
            print(f"Available: {list(benchmarks.keys())}")
    else:
        for domain, items in benchmarks.items():
            evaluator.run_evaluation(domain, items)


def main():
    parser = argparse.ArgumentParser(
        description="RCF CLI - Robust Continual Flow Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--quality", type=float, default=0.7, help="Model quality (0-1)")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run scaling/fatigue analysis")
    analyze_parser.add_argument("--sessions", type=int, default=10, help="Number of sessions")
    analyze_parser.add_argument("--output", type=str, help="Output file for report")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Run real-time dashboard")
    dash_parser.add_argument("--steps", type=int, default=20, help="Number of demo steps")
    dash_parser.add_argument("--refresh", type=float, default=0.5, help="Refresh rate (seconds)")
    dash_parser.set_defaults(func=cmd_dashboard)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run domain evaluation")
    eval_parser.add_argument("--domain", type=str, help="Specific domain to evaluate")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
