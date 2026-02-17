"""CLI interface for the pruned adaptive agent."""

import sys
import argparse
import json
from typing import Optional

from src.agent import AdaptiveWebAgent, AgentConfig
from src.evaluation import BenchmarkRunner, create_default_tasks, save_results, BenchmarkResult
from src.browser_tool import BrowserTool


def run_task(args):
    """Run a single task with the agent.
    
    Args:
        args: Command line arguments.
    """
    # Create agent config
    config = AgentConfig(
        max_steps=args.max_steps,
        uncertainty_threshold=args.uncertainty_threshold,
        debug=args.debug
    )
    
    # Create agent
    agent = AdaptiveWebAgent(config)
    agent.initialize()
    
    # Create browser tool if URL provided
    if args.url:
        browser = BrowserTool()
        
        # Run the task
        result = agent.run_task({
            "url": args.url,
            "goal": args.goal,
            "browser": browser
        })
        
        # Print result
        if args.debug:
            print(json.dumps(result, indent=2))
        else:
            if result.get("success"):
                print(f"✓ Task completed successfully in {result.get('steps', 0)} steps")
            else:
                print(f"✗ Task failed: {result.get('error', 'Unknown error')}")
    else:
        print("Error: --url required for task execution")
        sys.exit(1)


def run_benchmark(args):
    """Run benchmark tasks.
    
    Args:
        args: Command line arguments.
    """
    # Create agent config
    config = AgentConfig(
        max_steps=args.max_steps,
        uncertainty_threshold=args.uncertainty_threshold,
        debug=args.debug
    )
    
    # Create agent (or None for mock)
    if args.mock:
        agent = None
    else:
        agent = AdaptiveWebAgent(config)
        agent.initialize()
    
    # Create runner
    runner = BenchmarkRunner(agent)
    
    # Add tasks
    if args.tasks_file:
        # Load tasks from file
        with open(args.tasks_file) as f:
            import yaml
            tasks_data = yaml.safe_load(f)
            # TODO: Parse tasks from file
    else:
        # Use default tasks
        tasks = create_default_tasks()
        runner.add_tasks(tasks)
    
    # Run benchmark
    print("Running benchmark...")
    result = runner.run_benchmark()
    
    # Print results
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Total tasks:     {result.total_tasks}")
    print(f"Successful:      {result.successful_tasks}")
    print(f"Success rate:   {result.success_rate*100:.1f}%")
    print(f"Avg steps:      {result.avg_steps:.1f}")
    print(f"Avg tool calls: {result.avg_tool_calls:.1f}")
    print(f"Avg time:       {result.avg_execution_time:.1f}s")
    print(f"Avg checklist:  {result.avg_checklist_completion*100:.1f}%")
    print(f"Avg uncertainty:{result.avg_uncertainty*100:.1f}%")
    print(f"{'='*50}")
    
    # Save if requested
    if args.output:
        save_results(result, args.output)
        print(f"\nResults saved to: {args.output}")


def visualize(args):
    """Visualize trajectory or results.
    
    Args:
        args: Command line arguments.
    """
    if args.trajectory_file:
        # Load and visualize trajectory
        with open(args.trajectory_file) as f:
            data = json.load(f)
        
        print("Trajectory visualization:")
        print(f"Total steps: {len(data.get('steps', []))}")
        
        for i, step in enumerate(data.get("steps", [])):
            print(f"  {i+1}. {step.get('action', 'unknown')}")
    else:
        print("Error: --trajectory-file required for visualization")
        sys.exit(1)


def status(args):
    """Show agent status.
    
    Args:
        args: Command line arguments.
    """
    config = AgentConfig()
    
    print(f"{'='*50}")
    print(f"AGENT STATUS")
    print(f"{'='*50}")
    print(f"Max steps:              {config.max_steps}")
    print(f"Uncertainty threshold: {config.uncertainty_threshold}")
    print(f"Stuck threshold:       {config.stuck_threshold}")
    print(f"Use recovery:          {config.use_recovery}")
    print(f"Use checklist:         {config.use_checklist}")
    print(f"Debug mode:            {config.debug}")
    print(f"{'='*50}")


def interactive(args):
    """Run agent in interactive mode.
    
    Args:
        args: Command line arguments.
    """
    print("Interactive mode - type 'help' for commands, 'quit' to exit")
    
    # Create agent
    config = AgentConfig(debug=True)
    agent = AdaptiveWebAgent(config)
    agent.initialize()
    
    browser = BrowserTool()
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            if cmd == "quit" or cmd == "exit":
                break
            
            if cmd == "help":
                print("Commands:")
                print("  navigate <url>  - Navigate to URL")
                print("  click <selector> - Click element")
                print("  type <selector> <text> - Type into element")
                print("  extract <selector> - Extract content")
                print("  screenshot - Take screenshot")
                print("  status - Show agent status")
                print("  quit - Exit")
                continue
            
            if cmd.startswith("navigate "):
                url = cmd[9:].strip()
                result = browser.navigate(url)
                print(f"→ {result.observation}")
                continue
            
            if cmd.startswith("click "):
                selector = cmd[6:].strip()
                result = browser.click(selector)
                print(f"→ {'OK' if result.success else result.error}")
                continue
            
            if cmd.startswith("type "):
                parts = cmd[4:].strip().split(" ", 1)
                if len(parts) == 2:
                    selector, text = parts
                    result = browser.type(selector, text)
                    print(f"→ {'OK' if result.success else result.error}")
                else:
                    print("Usage: type <selector> <text>")
                continue
            
            if cmd.startswith("extract "):
                selector = cmd[8:].strip()
                result = browser.extract(selector)
                print(f"→ {result.observation[:100]}...")
                continue
            
            if cmd == "screenshot":
                result = browser.screenshot()
                print(f"→ Screenshot captured ({result.metadata.get('size', 0)} bytes)")
                continue
            
            if cmd == "status":
                state = browser.get_page_state()
                print(f"→ URL: {state.get('url', 'unknown')}")
                print(f"→ Title: {state.get('title', 'unknown')}")
                continue
            
            print(f"Unknown command: {cmd}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pruned Adaptive Agent - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run task command
    run_parser = subparsers.add_parser("run", help="Run a single task")
    run_parser.add_argument("--url", required=True, help="Target URL")
    run_parser.add_argument("--goal", required=True, help="Task goal")
    run_parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
    run_parser.add_argument("--uncertainty-threshold", type=float, default=0.7, help="Uncertainty threshold")
    run_parser.add_argument("--debug", action="store_true", help="Debug output")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--max-steps", type=int, default=50, help="Max steps per task")
    bench_parser.add_argument("--uncertainty-threshold", type=float, default=0.7, help="Uncertainty threshold")
    bench_parser.add_argument("--mock", action="store_true", help="Use mock agent")
    bench_parser.add_argument("--tasks-file", help="Path to tasks YAML file")
    bench_parser.add_argument("--output", help="Output JSON file for results")
    bench_parser.add_argument("--debug", action="store_true", help="Debug output")
    
    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize trajectory")
    vis_parser.add_argument("--trajectory-file", required=True, help="Trajectory JSON file")
    
    # Status command
    subparsers.add_parser("status", help="Show agent status")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Parse args
    args = parser.parse_args()
    
    # Run command
    if args.command == "run":
        run_task(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "visualize":
        visualize(args)
    elif args.command == "status":
        status(args)
    elif args.command == "interactive":
        interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
