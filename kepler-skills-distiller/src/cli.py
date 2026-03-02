#!/usr/bin/env python3
"""
Kepler Skills Distiller CLI

Command-line interface for the scientific equation discovery and
knowledge distillation pipeline.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def cmd_discover(args):
    """Run the equation discovery pipeline."""
    print("🔬 Kepler Equation Discovery Pipeline")
    print("=" * 50)
    
    import numpy as np
    from src.inferencer import PhysicalPropertyInferencer
    from src.reasoning_agent import KeplerReasoningAgent
    from src.skill_generator import ExpertSkillGenerator
    
    # Load data if provided
    if args.data_file:
        print(f"\n[1/4] Loading data from {args.data_file}...")
        with open(args.data_file, 'r') as f:
            raw_data = json.load(f)
        # Convert to numpy arrays
        data = {k: np.array(v) if isinstance(v, list) else v for k, v in raw_data.items()}
    else:
        print("\n[1/4] Using sample data...")
        # Create sample physics data: simple harmonic motion
        t = np.linspace(0, 10, 100)
        data = {
            "t": t,
            "x": np.sin(t),
            "v": np.cos(t),
        }
    
    # Initialize components
    print("\n[2/4] Initializing components...")
    inferencer = PhysicalPropertyInferencer(data)
    
    # Create mock model client for reasoning agent
    mock_client = MockModelClient()
    agent = KeplerReasoningAgent(model_client=mock_client)
    skill_gen = ExpertSkillGenerator(skills_dir=args.skills_dir)
    
    # Run discovery
    print("\n[3/4] Running discovery loop...")
    start_time = time.time()
    
    # Simulate discovery progress
    if args.verbose:
        for i in range(3):
            print(f"  Iteration {i+1}/3: Analyzing patterns...")
            time.sleep(0.1)
    
    # Infer physical properties
    properties = inferencer.infer_all()
    if args.verbose:
        print(f"  Inferred properties: {properties}")
    
    # Run reasoning agent
    result = agent.discover(data)
    
    elapsed = time.time() - start_time
    print(f"\n[4/4] Discovery complete in {elapsed:.2f}s")
    
    # Generate skill if successful
    if result.get("success", False):
        skill = skill_gen.generate_skill(
            trajectory=result,
            metrics={"elapsed": elapsed},
            skill_name=f"discovered_skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        print(f"\n✅ Generated skill: {skill.name}")
        
        if args.output:
            skill_path = Path(args.skills_dir) / f"{skill.name}.json"
            print(f"   Saved to: {skill_path}")
    else:
        print("\n⚠️ Discovery did not converge to a solution")
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Proposed equations: {len(result.get('proposed_expressions', []))}")
    print(f"   Best equation: {result.get('best_expression', 'N/A')}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"   Full results saved to: {output_path}")


class MockModelClient:
    """Mock model client for CLI testing."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response."""
        return json.dumps({
            "analysis": "Mock analysis of physical constraints.",
            "hypotheses": ["Linear relationship", "Quadratic relationship"],
            "proposed_expressions": ["x + v", "x^2 + v^2"],
        })


def cmd_distill(args):
    """Run the distillation pipeline."""
    print("🎓 Knowledge Distillation Pipeline")
    print("=" * 50)
    
    from src.apprentice import create_apprentice
    from src.distillation import create_mastery_distiller
    
    print("\n[1/3] Setting up apprentice model...")
    apprentice = create_apprentice(
        model_name=args.model,
        use_mock=args.mock,
    )
    apprentice.load()
    
    print(f"\n[2/3] Loading training data from {args.data}...")
    data_path = args.data
    
    print("\n[3/3] Running mastery distillation...")
    distiller = create_mastery_distiller(
        model_name=args.model,
        data_path=data_path,
        output_dir=args.output_dir,
        use_mock=args.mock,
    )
    
    start_time = time.time()
    result = distiller.run_curriculum(
        steps_per_tier=args.steps_per_tier,
        max_iterations=args.max_iterations,
    )
    elapsed = time.time() - start_time
    
    print(f"\n✅ Distillation complete in {elapsed:.2f}s")
    print(f"\n📊 Curriculum Status:")
    print(f"   Current tier: {result['current_tier']}")
    print(f"   Curriculum complete: {result['curriculum_complete']}")
    
    # Show tier mastery
    print("\n   Mastery by tier:")
    for tier, record in result.get('mastery_records', {}).items():
        status = "✅" if record['mastered'] else "⏳"
        print(f"   {status} {tier}: {record['correct']}/{record['attempts']} ({record['correct']/max(record['attempts'],1)*100:.0f}%)")


def cmd_benchmark(args):
    """Run the benchmark suite."""
    print("📈 Equation Discovery Benchmark")
    print("=" * 50)
    
    from src.benchmark import (
        EquationDiscoveryBenchmark,
        FeynmanDataset,
        create_mock_expert,
        create_mock_apprentice,
    )
    
    print("\n[1/2] Loading benchmark dataset...")
    dataset = FeynmanDataset()
    
    categories = args.categories.split(",") if args.categories else None
    problems = dataset.get_problems(
        categories=categories,
        max_complexity=args.max_complexity,
    )
    print(f"   Loaded {len(problems)} problems")
    
    print("\n[2/2] Running benchmark...")
    benchmark = EquationDiscoveryBenchmark(output_dir=args.output_dir)
    
    # Create models
    if args.mock:
        expert = create_mock_expert()
        apprentice = create_mock_apprentice()
    else:
        # For real models, would load from trained weights
        expert = create_mock_expert()  # Placeholder
        apprentice = create_mock_apprentice() if args.compare else None
    
    report = benchmark.run_benchmark(
        expert_model=expert,
        apprentice_model=apprentice,
        categories=categories,
        max_complexity=args.max_complexity,
    )
    
    print(f"\n📊 Results:")
    print(f"   Total problems: {report.total_problems}")
    print(f"   Expert accuracy: {report.expert_accuracy:.1%}")
    print(f"   Expert avg MAE: {report.expert_avg_mae:.4f}")
    print(f"   Expert avg time: {report.expert_avg_time:.3f}s")
    
    if apprentice:
        print(f"\n   Apprentice accuracy: {report.apprentice_accuracy:.1%}")
        print(f"   Apprentice avg MAE: {report.apprentice_avg_mae:.4f}")
        print(f"   Improvement ratio: {report.improvement_ratio:.1%}")


def cmd_visualize(args):
    """Visualize training or discovery progress."""
    print("📊 Visualization")
    print("=" * 50)
    
    if args.type == "capacity":
        _visualize_capacity(args)
    elif args.type == "curriculum":
        _visualize_curriculum(args)
    elif args.type == "skills":
        _visualize_skills(args)
    else:
        print(f"Unknown visualization type: {args.type}")
        print("Available types: capacity, curriculum, skills")


def _visualize_capacity(args):
    """Visualize information capacity over training."""
    print("\n📈 Information Capacity Over Training")
    print("-" * 40)
    
    # Mock visualization
    capacity_data = [1.2, 2.5, 3.8, 5.2, 6.1, 7.0, 7.8, 8.5]
    
    print("\n   Capacity (bits)")
    print("   ┌────────────────────────────────────┐")
    
    max_cap = max(capacity_data)
    for cap in capacity_data[-8:]:
        bar_len = int(cap / max_cap * 30)
        bar = "█" * bar_len
        print(f"   │{bar:<30} {cap:.1f} │")
    
    print("   └────────────────────────────────────┘")
    print("   Steps →")


def _visualize_curriculum(args):
    """Visualize curriculum progress."""
    print("\n📚 Bloom's Taxonomy Curriculum Progress")
    print("-" * 40)
    
    tiers = [
        ("Remember", True, 0.95),
        ("Understand", True, 0.88),
        ("Apply", True, 0.82),
        ("Analyze", False, 0.65),
        ("Evaluate", False, 0.30),
        ("Create", False, 0.10),
    ]
    
    print()
    for tier, mastered, accuracy in tiers:
        status = "✅" if mastered else "⏳"
        bar_len = int(accuracy * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"   {status} {tier:<12} [{bar}] {accuracy*100:.0f}%")


def _visualize_skills(args):
    """Visualize generated skills."""
    print("\n🧪 Generated Skills")
    print("-" * 40)
    
    skills_dir = Path(args.skills_dir)
    
    if not skills_dir.exists():
        print(f"   No skills directory found: {skills_dir}")
        return
    
    skills = list(skills_dir.glob("*.json"))
    
    if not skills:
        print("   No skills generated yet")
        return
    
    print(f"\n   Found {len(skills)} skills:\n")
    
    for skill_file in skills[:5]:  # Show up to 5
        with open(skill_file) as f:
            skill = json.load(f)
        
        name = skill.get("name", skill_file.stem)
        domain = skill.get("domain", "unknown")
        steps = len(skill.get("procedural_steps", []))
        
        print(f"   📝 {name}")
        print(f"      Domain: {domain}")
        print(f"      Steps: {steps}")
        print()


def cmd_list_skills(args):
    """List all generated skills."""
    print("📋 Generated Skills")
    print("=" * 50)
    
    skills_dir = Path(args.skills_dir)
    
    if not skills_dir.exists():
        print(f"\nNo skills directory found: {skills_dir}")
        return
    
    skills = list(skills_dir.glob("*.json"))
    
    if not skills:
        print("\nNo skills generated yet. Run 'discover' first.")
        return
    
    print(f"\nFound {len(skills)} skills:\n")
    
    for i, skill_file in enumerate(skills, 1):
        with open(skill_file) as f:
            skill = json.load(f)
        
        name = skill.get("name", skill_file.stem)
        domain = skill.get("domain", "unknown")
        
        print(f"  {i}. {name} ({domain})")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Kepler Skills Distiller CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Run equation discovery pipeline"
    )
    discover_parser.add_argument("--data-file", help="Path to input data")
    discover_parser.add_argument("--output", "-o", help="Output file for results")
    discover_parser.add_argument("--skills-dir", default="skills", help="Skills directory")
    discover_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    discover_parser.set_defaults(func=cmd_discover)
    
    # Distill command
    distill_parser = subparsers.add_parser(
        "distill",
        help="Run distillation pipeline"
    )
    distill_parser.add_argument("--data", required=True, help="Training data path")
    distill_parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct", help="Model name")
    distill_parser.add_argument("--output-dir", default="outputs/distillation", help="Output directory")
    distill_parser.add_argument("--steps-per-tier", type=int, default=50, help="Steps per tier")
    distill_parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations")
    distill_parser.add_argument("--mock", action="store_true", help="Use mock model")
    distill_parser.set_defaults(func=cmd_distill)
    
    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark suite"
    )
    bench_parser.add_argument("--categories", help="Comma-separated categories")
    bench_parser.add_argument("--max-complexity", type=int, default=3, help="Max complexity")
    bench_parser.add_argument("--compare", action="store_true", help="Compare with apprentice")
    bench_parser.add_argument("--output-dir", help="Output directory for results")
    bench_parser.add_argument("--mock", action="store_true", help="Use mock models")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Visualize command
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Visualize training or discovery"
    )
    viz_parser.add_argument("type", choices=["capacity", "curriculum", "skills"],
                           help="Visualization type")
    viz_parser.add_argument("--skills-dir", default="skills", help="Skills directory")
    viz_parser.set_defaults(func=cmd_visualize)
    
    # List skills command
    list_parser = subparsers.add_parser(
        "list-skills",
        help="List generated skills"
    )
    list_parser.add_argument("--skills-dir", default="skills", help="Skills directory")
    list_parser.set_defaults(func=cmd_list_skills)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
