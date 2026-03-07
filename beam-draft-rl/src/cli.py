import argparse
import sys
import torch
from src.trainer import BDRLTrainer
from src.benchmark import BenchmarkRunner
from src.curriculum import CurriculumManager
from src.model_wrapper import DraftWrapper

def handle_train(args):
    print(f"Starting training for model: {args.model_name}")
    trainer = BDRLTrainer(model_name=args.model_name, learning_rate=args.lr)
    # trainer.setup_model()  # Assume setup_model is needed or handled inside train loop
    # In a real implementation, we would load data and call a train method
    print("Training initiated (simulated).")

def handle_evaluate(args):
    print(f"Starting evaluation of model: {args.model_path}")
    runner = BenchmarkRunner()
    # Mock dataset for evaluation
    dataset = [{"id": 1, "prompt": "Calculate the reaction force at the fixed end of a 5m cantilever beam with a 10kN load at the free end.", "target": "10kN"}]
    # trajectories = runner.run_benchmark(model=args.model_path, dataset=dataset) # runner needs a model object not path usually
    print("Evaluation completed (simulated).")

def handle_visualize(args):
    wrapper = DraftWrapper()
    raw_reasoning = "To find the reaction force, we sum the vertical forces. Ry - 10kN = 0, so Ry = 10kN. The moment at the support is M = 10kN * 5m = 50kNm."
    drafted = wrapper.wrap_prompt("Problem: 10kN at 5m cantilever.")
    example_output = f"{wrapper.DRAFT_START}Ry=10, M=50{wrapper.DRAFT_END} Final Answer: Ry=10kN, M=50kNm."
    
    print("--- RAW REASONING ---")
    print(raw_reasoning)
    print("\n--- DRAFTED REASONING ---")
    print(example_output)
    print("\n--- EXTRACTED DRAFT ---")
    print(wrapper.extract_draft(example_output))

def handle_solve(args):
    print(f"Solving problem: {args.prompt}")
    # In a real implementation:
    # model = load_model(args.model_path)
    # output = model.generate(args.prompt)
    # print(f"Solution: {output}")
    print("Solution generated (simulated).")

def main():
    parser = argparse.ArgumentParser(description="BeamDraft-RL CLI Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run the training loop")
    train_parser.add_argument("--model-name", type=str, required=True, help="Base model name or path")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.set_defaults(func=handle_train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmarks")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Path to the model to evaluate")
    eval_parser.set_defaults(func=handle_evaluate)

    # Visualize-drafts command
    viz_parser = subparsers.add_parser("visualize-drafts", help="Show examples of raw vs. drafted reasoning")
    viz_parser.set_defaults(func=handle_visualize)

    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a single beam problem")
    solve_parser.add_argument("prompt", type=str, help="The beam problem to solve")
    solve_parser.add_argument("--model-path", type=str, help="Path to the model")
    solve_parser.set_defaults(func=handle_solve)

    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
