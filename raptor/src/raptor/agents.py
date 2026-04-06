"""Agent protocol and multi-agent polling utilities.

Provides:
  - ReasoningAgent protocol (abstract interface)
  - poll_agents() for collecting multi-agent responses
  - reroll() helper for rerolling with candidate selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from raptor.disagreement import AgentResponse


@runtime_checkable
class ReasoningAgent(Protocol):
    """Abstract interface for a reasoning-capable LLM agent."""

    def generate(self, prompt: str) -> AgentResponse:
        """Generate a response to the given prompt.

        Returns:
            AgentResponse with reasoning_steps and final_answer.
        """
        ...


def poll_agents(
    agents: list[ReasoningAgent],
    prompt: str,
    return_embeddings: bool = False,
) -> list[AgentResponse]:
    """Poll multiple agents and collect their responses.

    Args:
        agents: List of ReasoningAgent instances
        prompt: The prompt to send to each agent
        return_embeddings: If True, compute and attach embeddings to each response

    Returns:
        List of AgentResponse, one per agent
    """
    # TODO: implement Step 6
    # - Dispatch generate() to each agent in parallel (concurrent.futures)
    # - If return_embeddings: compute sentence embeddings for each response
    # - Return list of AgentResponse
    raise NotImplementedError("Step 6 — Agent polling not yet implemented")


def reroll(
    agent: ReasoningAgent,
    prompt: str,
    n_candidates: int = 3,
) -> list[AgentResponse]:
    """Generate multiple candidate responses and return all of them.

    The caller (typically RAPTOROrchestrator) selects the best via disagreement monitoring.

    Args:
        agent: The agent to use for rerolling
        prompt: The prompt
        n_candidates: Number of candidates to generate

    Returns:
        List of n_candidates AgentResponses
    """
    # TODO: implement Step 6
    raise NotImplementedError("Step 6 — reroll not yet implemented")


# ----------------------------------------------------------------------
# Mock agents for testing (use in tests/)
# ----------------------------------------------------------------------
@dataclass
class MockReasoningAgent:
    """A deterministic mock agent for testing."""

    agent_id: str
    reasoning_steps: list[str]
    final_answer: str

    def generate(self, prompt: str) -> AgentResponse:
        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=self.reasoning_steps,
            final_answer=self.final_answer,
        )
