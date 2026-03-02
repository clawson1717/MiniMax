import logging
import argparse
import sys
import json
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from src.evaluation import Evaluator
from src.integrated_loop import IntegratedAdversarialLoop
from src.visualization import print_terminal_chart, plot_gap_closing, visualize_evaluation

# Set up rich console
if HAS_RICH:
    console = Console()
else:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

def run_loop_with_rich(args):
    """Runs the IntegratedAdversarialLoop with rich progress and display."""
    # Example agent configs - in a real app, these could be loaded from a file
    configs = [
        {"id": "agent_1", "domain": "physics", "morphology": {"expertise": "science"}},
        {"id": "agent_2", "domain": "math", "morphology": {"expertise": "science"}},
        {"id": "agent_3", "domain": "ethics", "morphology": {"expertise": "humanities"}},
    ]
    
    expert_text = args.expert_reference or "The unified theory must account for both quantum effects and gravitational constants."
    query = args.query or "How do we unify physics?"
    
    console.print(f"Initializing Adversarial Loop...")
    loop = IntegratedAdversarialLoop(configs, expert_text, max_iterations=args.iterations)
    
    # Actually run the loop
    final_state = loop.run_iteration(query)
    history = final_state["history"]
    
    # Display results
    if HAS_RICH:
        table = Table(title="Adversarial Loop Results")
        table.add_column("Iteration", justify="right", style="cyan")
        table.add_column("Gap Score", justify="center", style="magenta")
        table.add_column("Consensus Summary (Snippet)", style="green")
        
        for entry in history:
            snippet = entry["consensus_summary"][:50] + "..." if len(entry["consensus_summary"]) > 50 else entry["consensus_summary"]
            table.add_row(str(entry["iteration"] + 1), f"{entry['gap_score']:.4f}", snippet)
        
        console.print(table)
    else:
        print("\nAdversarial Loop Results:")
        for entry in history:
            print(f"Iter {entry['iteration']+1}: Gap Score {entry['gap_score']:.4f}")
    
    if args.visualize:
        print_terminal_chart(history)
        if args.plot_output:
            plot_gap_closing(history, save_path=args.plot_output)

def run_eval_with_rich(args):
    """Runs the Evaluator with rich display."""
    evaluator = Evaluator()
    benchmarks = evaluator.load_benchmarks()
    
    domains_to_run = []
    if args.domain:
        if args.domain in benchmarks:
            domains_to_run = [args.domain]
        else:
            console.print(f"Error: Domain {args.domain} not found.")
            return
    else:
        domains_to_run = list(benchmarks.keys())

    for domain in domains_to_run:
        console.print(f"Evaluating Domain: {domain}")
        report = evaluator.run_evaluation(domain, benchmarks[domain], max_iterations=args.iterations)
        
        # Display evaluation results
        if HAS_RICH:
            table = Table(title=f"Evaluation Results: {domain}")
            table.add_column("Query", style="cyan")
            table.add_column("Avg Improvement", justify="right", style="green")
            table.add_column("Stability (Var)", justify="right", style="magenta")
            
            for res in report["results"]:
                table.add_row(res["query"][:40] + "...", f"{res['avg_accuracy_improvement']:.4f}", f"{res['convergence_stability']:.4e}")
            
            console.print(table)
        else:
            print(f"\nEvaluation Results: {domain}")
            for res in report["results"]:
                print(f"Query: {res['query'][:40]}... | Improv: {res['avg_accuracy_improvement']:.4f}")
        
        if args.visualize:
            visualize_evaluation(report)
            if args.plot_output:
                # Append domain name to plot output if multiple
                base, ext = os.path.splitext(args.plot_output)
                domain_plot_path = f"{base}_{domain}{ext}"
                visualize_evaluation(report, save_path=domain_plot_path)

def main():
    parser = argparse.ArgumentParser(description="Adversarial Domain Diffuser CLI")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run loop command
    loop_parser = subparsers.add_parser("run", help="Run the adversarial loop")
    loop_parser.add_argument("--query", type=str, help="Initial query/problem statement")
    loop_parser.add_argument("--expert-reference", type=str, help="Expert reference text")
    loop_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    loop_parser.add_argument("--visualize", action="store_true", help="Enable terminal visualization")
    loop_parser.add_argument("--plot-output", type=str, help="Path to save plot (e.g., plot.png)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run domain-specific evaluation")
    eval_parser.add_argument("--domain", type=str, help="Specific domain to evaluate")
    eval_parser.add_argument("--iterations", type=int, default=3, help="Iterations per loop")
    eval_parser.add_argument("--visualize", action="store_true", help="Enable terminal visualization")
    eval_parser.add_argument("--plot-output", type=str, help="Path to save plot (e.g., eval_plot.png)")

    args = parser.parse_args()
    
    # Setup logging
    if HAS_RICH:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.command == "run":
        run_loop_with_rich(args)
    elif args.command == "eval":
        run_eval_with_rich(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
