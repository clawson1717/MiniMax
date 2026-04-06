"""CLI commands for SNTR."""

from __future__ import annotations
import json
import structlog

from .agent import SNTRAgent
from .data_structures import FailureTrajectory

logger = structlog.get_logger()


def diagnose_command(trajectory_path: str):
    """Diagnose failures in a trajectory file."""
    agent = SNTRAgent()
    trajectory = FailureTrajectory.from_file(trajectory_path)
    result = agent.self_heal(trajectory)
    print(json.dumps({"status": result.status, "failure_type": result.failure_type.value if result.failure_type else None}, indent=2))


def repair_command(step: str, context_json: str | None):
    """Repair a single failed reasoning step."""
    agent = SNTRAgent()
    context = json.loads(context_json) if context_json else {}
    result = agent.repair_step(step)
    print(json.dumps({
        "status": result.status,
        "original": result.original_step,
        "corrected": result.corrected_step,
    }, indent=2))


def replay_command(trajectory_path: str, timeout: int):
    """Run full self-healing loop on a trajectory."""
    agent = SNTRAgent(isabelle_timeout=timeout)
    trajectory = FailureTrajectory.from_file(trajectory_path)
    result = agent.self_heal(trajectory)
    print(json.dumps({
        "status": result.status,
        "original": result.original_step,
        "corrected": result.corrected_step,
        "experience_hits": result.experience_hits,
    }, indent=2))


def experience_stats_command():
    """Show hindsight experience library statistics."""
    agent = SNTRAgent()
    stats = agent.loop.experience_lib.stats()
    print(json.dumps(stats, indent=2))


def prove_command(claim: str, timeout: int):
    """Standalone Isabelle/HOL proof query."""
    agent = SNTRAgent(isabelle_timeout=timeout)
    result = agent.prove(claim)
    print(json.dumps(result.to_dict(), indent=2))
