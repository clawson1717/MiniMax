try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import os
import json
from typing import List, Dict, Any

def plot_gap_closing(history: List[Dict[str, Any]], title: str = "Gap Closing Progress", save_path: str = None):
    """
    Generates a simple plot showing the gap score reduction over iterations.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed. Skipping plot generation.")
        return

    iterations = [item["iteration"] for item in history]
    gap_scores = [item["gap_score"] for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, gap_scores, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Gap Score")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def print_terminal_chart(history: List[Dict[str, Any]]):
    """
    Prints a simple ASCII/Terminal chart of the gap closing progress.
    """
    if not history:
        return

    print("\nGap Closing Progress (Terminal Visualization):")
    print("-" * 50)
    max_score = max(item["gap_score"] for item in history)
    
    # Scale scores for terminal display (30 chars wide)
    width = 30
    
    for item in history:
        score = item["gap_score"]
        bar_len = int((score / max_score) * width) if max_score > 0 else 0
        bar = "█" * bar_len
        print(f"Iter {item['iteration'] + 1:2d} | {score:.4f} | {bar}")
    print("-" * 50)

def visualize_evaluation(report: Dict[str, Any], save_path: str = None):
    """
    Visualizes evaluation results across runs.
    """
    domain = report.get("domain", "Unknown")
    results = report.get("results", [])
    
    if not results:
        print("No results to visualize.")
        return

    # Print summary table-like structure to terminal
    print(f"\nEvaluation Summary for {domain}")
    print("=" * 60)
    print(f"{'Query (Truncated)':<30} | {'Avg Improv':<10} | {'Stability':<10}")
    print("-" * 60)
    
    for res in results:
        query = (res["query"][:27] + '...') if len(res["query"]) > 30 else res["query"]
        print(f"{query:<30} | {res['avg_accuracy_improvement']:<10.4f} | {res['convergence_stability']:<10.4f}")
    print("=" * 60)

    # If matplotlib is available and save_path is provided, plot improvement
    if HAS_MATPLOTLIB and save_path:
        queries = [f"Q{i+1}" for i in range(len(results))]
        improvements = [res["avg_accuracy_improvement"] for res in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(queries, improvements, color='green')
        plt.title(f"Average Accuracy Improvement - {domain}")
        plt.xlabel("Query ID")
        plt.ylabel("Avg Improvement")
        plt.savefig(save_path)
        print(f"Evaluation plot saved to {save_path}")
