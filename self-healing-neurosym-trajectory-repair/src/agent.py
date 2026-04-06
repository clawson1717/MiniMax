"""SNTRAgent — high-level Python API for SNTR."""

from __future__ import annotations
import structlog

from .sntr_loop import SNTRSelfHealingLoop
from .data_structures import FailureTrajectory, SNTRResult
from .isabelle_engine import IsabelleProofEngine

logger = structlog.get_logger()


class SNTRAgent:
    """
    High-level SNTR agent API.
    
    Example:
        agent = SNTRAgent(isabelle_timeout=30)
        result = agent.self_heal(trajectory)
    """

    def __init__(
        self,
        isabelle_timeout: int = 30,
        repair_threshold: float = 0.5,
        experience_cache_size: int = 1000,
    ):
        self.loop = SNTRSelfHealingLoop(
            isabelle_timeout=isabelle_timeout,
            repair_threshold=repair_threshold,
            experience_cache_size=experience_cache_size,
        )
        logger.info("sntr.agent.init", timeout=isabelle_timeout, threshold=repair_threshold)

    def self_heal(self, trajectory: FailureTrajectory) -> SNTRResult:
        """Run full self-healing loop on a failed trajectory."""
        return self.loop.self_heal(trajectory)

    def repair_step(self, step_content: str, confidence: float = 0.3) -> SNTRResult:
        """Repair a single reasoning step directly."""
        step = {
            "id": "direct",
            "content": step_content,
            "confidence": confidence,
        }
        trajectory = FailureTrajectory(task="direct_repair", steps=[step])
        return self.loop.self_heal(trajectory)

    def prove(self, claim: str, context: list[str] | None = None):
        """Standalone Isabelle/HOL proof query."""
        return self.loop.isabelle.prove(claim, context)

    def provision_isabelle(self) -> str:
        """Auto-provision Isabelle2024 Docker container."""
        return self.loop.isabelle.provision_isabelle_docker()
