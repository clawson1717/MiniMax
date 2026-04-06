"""Milestone Guardian integration — SNTR as a drop-in repair engine for Milestone Guardian."""

from __future__ import annotations
import structlog
from typing import Optional

from .agent import SNTRAgent
from .data_structures import FailureTrajectory, SNTRResult

logger = structlog.get_logger()


class MilestoneIntegration:
    """
    SNTR as a repair engine for Milestone Guardian.
    
    When Milestone Guardian's critic rejects a milestone evidence chain,
    this integration invokes SNTR to formally repair the reasoning.
    
    Usage:
        mg = MilestoneGuardianAgent(...)
        sntr = SNTRAgent(...)
        integration = MilestoneIntegration(sntr, mg)
        
        # In MG's recovery loop:
        if mg.critic.review(evidence) == REJECTED:
            repaired = integration.repair_milestone(milestone, evidence, critic_feedback)
    """

    def __init__(self, sntr_agent: SNTRAgent, milestone_guardian=None):
        self.sntr = sntr_agent
        self.mg = milestone_guardian

    def repair_milestone(
        self,
        milestone_id: str,
        failed_step: str,
        critic_feedback: str,
        confidence: float = 0.3,
    ) -> SNTRResult:
        """
        Repair a rejected milestone using SNTR.
        
        1. Extract the failed reasoning from the rejected milestone
        2. Run SNTR self-healing loop
        3. If repair succeeds → patch milestone and continue to next
        4. If repair fails → return to Milestone Guardian for standard recovery
        """
        logger.info(
            "sntr.milestone.repair",
            milestone_id=milestone_id,
            feedback=critic_feedback[:100],
        )

        step = {
            "id": milestone_id,
            "content": failed_step,
            "confidence": confidence,
            "critic_feedback": critic_feedback,
        }

        trajectory = FailureTrajectory(
            task=f"milestone_repair:{milestone_id}",
            steps=[step],
        )

        result = self.sntr.self_heal(trajectory)

        logger.info(
            "sntr.milestone.result",
            milestone_id=milestone_id,
            status=result.status,
        )

        return result

    def should_escalate(self, result: SNTRResult) -> bool:
        """Determine if repaired milestone should still be escalated."""
        if result.status == "REPAIR_SUCCEEDED":
            return result.patch and result.patch.confidence_after < 0.7
        return True
