import json
import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint

from src.agent import TVCAgent, TVCAgentConfig, TVCReport
from src.node import TrajectoryNode, NodeStatus
from src.benchmark import TVCBenchmark

console = Console()

class TVCCLI:
    """Command Line Interface for Trajectory Verification Cascade."""

    def __init__(self):
        self.agent = TVCAgent()
        self.benchmark_suite = TVCBenchmark()

    def run_task(self, task_file: str, output_file: Optional[str] = None):
        """Processes a reasoning task from a JSON file."""
        if not os.path.exists(task_file):
            console.print(f"[bold red]Error:[/] File {task_file} not found.", style="red")
            return

        try:
            with open(task_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Failed to parse JSON: {e}", style="red")
            return

        task_prompt = data.get("task", "Unnamed Task")
        reasoning_steps = data.get("steps", [])

        if not reasoning_steps:
             console.print("[yellow]Warning:[/] No reasoning steps found in JSON.", style="yellow")

        console.print(Panel(f"[bold blue]Processing Task:[/]\n{task_prompt}", title="TVC Run"))

        # Setup config if provided in JSON
        config_data = data.get("config", {})
        config = TVCAgentConfig(**config_data) if config_data else TVCAgentConfig()
        self.agent = TVCAgent(config=config)

        # Run the agent
        report = self.agent.process_task(task_prompt, reasoning_steps)

        # Display results
        self._display_report(report)

        # Export if requested
        if output_file:
            self._export_report(report, output_file)

    def run_benchmark(self, output_file: Optional[str] = None):
        """Runs the built-in benchmark suite."""
        console.print(Panel("[bold magenta]Running TVC Benchmark Suite[/]", title="Benchmark"))
        
        results = self.benchmark_suite.run_benchmark()
        summary = self.benchmark_suite.calculate_summary_metrics(results)

        # Display results table
        table = Table(title="Benchmark Results")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Exp. Success", justify="center")
        table.add_column("Actual", justify="center")
        table.add_column("Status", justify="center")

        for res in results:
            task = next((t for t in self.benchmark_suite.tasks if t.id == res.task_id), None)
            name = task.name if task else "Unknown"
            
            exp_str = "[green]YES[/]" if res.expected_success else "[red]NO[/]"
            act_str = "[green]YES[/]" if res.success else "[red]NO[/]"
            
            status = "[bold green]PASS[/]" if res.success == res.expected_success else "[bold red]FAIL[/]"
            
            table.add_row(res.task_id, name, exp_str, act_str, status)

        console.print(table)

        # Display summary
        summary_panel = Panel(
            f"Total Tasks: {summary['total_tasks']}\n"
            f"Verification Accuracy: {summary['verification_accuracy']:.2%}\n"
            f"Adversarial Detection Rate: {summary['adversarial_failure_detection_rate']:.2%}\n"
            f"Pruning Efficiency: {summary['pruning_efficiency']:.2%}\n"
            f"Avg Steps: {summary['avg_steps']:.1f}",
            title="Summary Metrics",
            expand=False
        )
        console.print(summary_panel)

        if output_file:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "results": [
                    {
                        "task_id": r.task_id,
                        "success": r.success,
                        "expected_success": r.expected_success,
                        "metrics": r.metrics,
                        "failure_reason": r.failure_reason
                    } for r in results
                ]
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[green]Benchmark report exported to {output_file}[/]")

    def visualize_trajectory(self, report_file: str):
        """Visualizes a trajectory from a saved report JSON."""
        if not os.path.exists(report_file):
            console.print(f"[bold red]Error:[/] File {report_file} not found.", style="red")
            return

        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            console.print("[bold red]Error:[/] Failed to parse JSON.", style="red")
            return

        trajectory = data.get("trajectory", [])
        if not trajectory:
            console.print("[yellow]No trajectory found in report.[/]")
            return

        tree = Tree(f"[bold blue]Trajectory: {data.get('task', 'Unknown Task')}[/]")
        
        # Simple linear visualization for now as richness of graph isn't fully exported yet
        # In a real graph, we'd build the tree recursively from parents/children.
        # Since the current agent produces a linear list for the 'verified_path', 
        # we'll use that.
        
        curr = tree
        for i, node in enumerate(trajectory):
            status_color = "green" if node["status"] == "VERIFIED" else "red"
            node_label = f"[bold {status_color}]Node {node['id']}:[/] {node['content'][:50]}..."
            curr = curr.add(node_label)
            
            if node.get("checklist_items"):
                for item in node["checklist_items"]:
                    check_color = "green" if item["passed"] else "red"
                    curr.add(f"[{check_color}]- {item['criterion']}[/]")

        console.print(tree)

    def _display_report(self, report: TVCReport):
        """Displays a TVCReport in the terminal."""
        status_str = "[bold green]SUCCESS[/]" if report.success else "[bold red]FAILED[/]"
        console.print(f"\nStatus: {status_str}")
        
        if report.failure_reason:
            console.print(f"Reason: [yellow]{report.failure_reason}[/]")

        # Trajectory Table
        table = Table(title="Verified Trajectory")
        table.add_column("Node ID", style="cyan")
        table.add_column("Content", style="white", overflow="ellipsis")
        table.add_column("Status", justify="center")

        for node in report.trajectory:
            status = f"[green]{node.status.value}[/]" if node.status == NodeStatus.VERIFIED else f"[red]{node.status.value}[/]"
            table.add_row(node.id, node.content[:60] + "...", status)

        console.print(table)

        # Metrics
        metrics_str = ", ".join([f"{k}: {v}" for k, v in report.metrics.items()])
        console.print(f"[dim]Metrics: {metrics_str}[/]")

    def _export_report(self, report: TVCReport, output_file: str):
        """Exports a TVCReport to JSON."""
        def serialize_node(node: TrajectoryNode) -> Dict[str, Any]:
            return {
                "id": node.id,
                "content": node.content,
                "status": node.status.value,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "checklist_items": [
                    {"criterion": item.criterion, "passed": item.passed, "evidence": item.evidence}
                    for item in node.checklist_items
                ]
            }

        data = {
            "task": report.task,
            "success": report.success,
            "failure_reason": report.failure_reason,
            "metrics": report.metrics,
            "trajectory": [serialize_node(n) for n in report.trajectory],
            "timestamp": datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Report exported to {output_file}[/]")

def main():
    parser = argparse.ArgumentParser(description="TVC - Trajectory Verification Cascade CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Process a task from JSON")
    run_parser.add_argument("file", help="Path to task JSON file")
    run_parser.add_argument("--output", "-o", help="Path to export report JSON")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--output", "-o", help="Path to export benchmark results")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize a trajectory report")
    viz_parser.add_argument("file", help="Path to report JSON file")

    args = parser.parse_args()
    cli = TVCCLI()

    if args.command == "run":
        cli.run_task(args.file, args.output)
    elif args.command == "benchmark":
        cli.run_benchmark(args.output)
    elif args.command == "visualize":
        cli.visualize_trajectory(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
