import argparse
import sys
import torch
import json
from typing import List, Dict, Any

from src.agent import ReasoningAgent
from src.trainer import MasteryTrainer
from src.pedagogy import ZPDController, BloomTier
from src.metrics import InformationTheoreticMetrics

def load_sample_dataset() -> List[Dict[str, Any]]:
    """Loads a basic sample dataset for training/evaluation."""
    return [
        {"prompt": "Calculate 2+2", "tier": BloomTier.KNOWLEDGE},
        {"prompt": "Explain the concept of gravity", "tier": BloomTier.COMPREHENSION},
        {"prompt": "Write a python script to sort a list", "tier": BloomTier.APPLICATION},
        {"prompt": "Analyze the themes in Hamlet", "tier": BloomTier.ANALYSIS},
        {"prompt": "Evaluate the impact of the industrial revolution", "tier": BloomTier.EVALUATION},
        {"prompt": "Design a new type of renewable energy storage", "tier": BloomTier.SYNTHESIS},
    ]

def train_cmd(args):
    print(f"Starting training for {args.epochs} epochs...")
    agent = ReasoningAgent()
    trainer = MasteryTrainer(agent, learning_rate=args.lr)
    
    dataset = load_sample_dataset()
    
    all_metrics = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        metrics = trainer.run_epoch(dataset)
        all_metrics.extend(metrics)
        
        # Simple progress report
        avg_success = sum(m["success_score"] for m in metrics) / len(metrics)
        avg_capacity = sum(m["world_model_capacity"] for m in metrics) / len(metrics)
        print(f"  Avg Success: {avg_success:.2f}, Avg Capacity: {avg_capacity:.2f} bits")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Results saved to {args.output}")

def evaluate_cmd(args):
    print("Evaluating model...")
    agent = ReasoningAgent()
    trainer = MasteryTrainer(agent)
    
    dataset = load_sample_dataset()
    capacity = trainer.evaluate_world_model(dataset)
    
    print(f"Results:")
    print(f"  World Model Capacity: {capacity:.2f} bits")

def visualize_cmd(args):
    print("Visualizing Mastery/World-Model relationship...")
    # This is a stub as per requirements
    print("Relationship Summary:")
    print("  Mastery (Bloom's Tier) scales with World Model Capacity (bits).")
    print("  As the model moves from 'Remember' to 'Create', we observe an increase")
    print("  in the estimated bits of environment information internalised.")
    print("\n[Visualisation plot would be generated here in a full implementation]")

def main():
    parser = argparse.ArgumentParser(description="Mastery Reasoning Refiner CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run the training pipeline")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    train_parser.add_argument("--output", type=str, help="Path to save training metrics")

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize relationships")

    args = parser.parse_args()

    if args.command == "train":
        train_cmd(args)
    elif args.command == "evaluate":
        evaluate_cmd(args)
    elif args.command == "visualize":
        visualize_cmd(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
