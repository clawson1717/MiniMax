"""CLI entry point for SNTR."""

import argparse
import json
import structlog

from sntr.agent import SNTRAgent
from sntr.data_structures import FailureTrajectory

logger = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(
        prog="sntr",
        description="SNTR: Self-Healing via Neuro-Symbolic Trajectory Repair"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # diagnose
    diagnose = subparsers.add_parser("diagnose", help="Diagnose failures in a trajectory")
    diagnose.add_argument("trajectory", help="Path to trajectory JSON file")

    # repair
    repair = subparsers.add_parser("repair", help="Repair a single failed step")
    repair.add_argument("step", help="Failed reasoning step text")
    repair.add_argument("--context", help="Additional context as JSON string")

    # replay
    replay = subparsers.add_parser("replay", help="Run full self-healing loop on a trajectory")
    replay.add_argument("trajectory", help="Path to trajectory JSON file")
    replay.add_argument("--timeout", type=int, default=30, help="Isabelle proof timeout (seconds)")

    # experience-stats
    subparsers.add_parser("experience-stats", help="Show hindsight experience library statistics")

    # prove
    prove = subparsers.add_parser("prove", help="Standalone Isabelle/HOL proof query")
    prove.add_argument("claim", help="Claim to prove")
    prove.add_argument("--timeout", type=int, default=30, help="Proof timeout (seconds)")

    args = parser.parse_args()

    if args.command == "diagnose":
        _diagnose(args.trajectory)
    elif args.command == "repair":
        _repair(args.step, args.context)
    elif args.command == "replay":
        _replay(args.trajectory, args.timeout)
    elif args.command == "experience-stats":
        _experience_stats()
    elif args.command == "prove":
        _prove(args.claim, args.timeout)
    else:
        parser.print_help()


def _diagnose(trajectory_path: str):
    agent = SNTRAgent()
    trajectory = FailureTrajectory.from_file(trajectory_path)
    result = agent.self_heal(trajectory)
    print(json.dumps({
        "status": result.status,
        "failure_type": result.failure_type.value if result.failure_type else None,
    }, indent=2))


def _repair(step: str, context_json: str | None):
    agent = SNTRAgent()
    result = agent.repair_step(step)
    print(json.dumps({
        "status": result.status,
        "original": result.original_step,
        "corrected": result.corrected_step,
    }, indent=2))


def _replay(trajectory_path: str, timeout: int):
    agent = SNTRAgent(isabelle_timeout=timeout)
    trajectory = FailureTrajectory.from_file(trajectory_path)
    result = agent.self_heal(trajectory)
    print(json.dumps({
        "status": result.status,
        "original": result.original_step,
        "corrected": result.corrected_step,
        "experience_hits": result.experience_hits,
    }, indent=2))


def _experience_stats():
    agent = SNTRAgent()
    stats = agent.loop.experience_lib.stats()
    print(json.dumps(stats, indent=2))


def _prove(claim: str, timeout: int):
    agent = SNTRAgent(isabelle_timeout=timeout)
    result = agent.prove(claim)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
