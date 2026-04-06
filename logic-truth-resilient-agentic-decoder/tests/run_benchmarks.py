import asyncio
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from src.agent import LTRADAgent
from src.benchmark import TruthResilienceSuite

async def run_benchmarks():
    agent = LTRADAgent(use_mock=True)
    suite = TruthResilienceSuite(agent)
    
    # 1. Setup LOGIGEN-inspired tasks
    test_tasks = [
        {
            "goal": "Open vault and aggregate data",
            "initial_state": {"authorized": False},
            "min_expected_steps": 2
        },
        {
            "goal": "Synthesis high-entropy nodes",
            "initial_state": {"empty": True},
            "min_expected_steps": 2
        },
        {
            "goal": "Generic research task",
            "initial_state": {},
            "min_expected_steps": 1
        }
    ]
    
    print(f"Running Deterministic Benchmark Suite ({len(test_tasks)} tasks)...")
    results = await suite.run_suite(test_tasks)
    
    print("-" * 25)
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Avg TR Score: {results['avg_tr_score']:.2f}")
    print(f"Denoising Freq: {results['denoising_frequency']*100:.1f}%")
    print(f"Total Time: {results['total_time_ms']}ms")
    print("-" * 25)
    
    assert results["success_rate"] == 1.0
    assert results["avg_tr_score"] > 0
    print("✓ Benchmark suite verified.")

if __name__ == "__main__":
    asyncio.run(run_benchmarks())
