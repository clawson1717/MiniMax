"""Agent protocol and multi-agent polling utilities.

Provides:
  - ReasoningAgent protocol (abstract interface for sync generation)
  - StreamingReasoningAgent protocol (yields tokens/steps for real-time entropy)
  - AgentStreamEvent for streaming token/step/done events
  - TokenLogProbs for optional per-token log-probability data
  - poll_agents() for concurrent multi-agent polling with timeout/error handling
  - poll_agents_streaming() for streaming variant with real-time callbacks
  - reroll() helper for generating n_candidates from a single agent
  - AgentError for wrapping agent failures
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np
from loguru import logger

from raptor.disagreement import AgentResponse


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class TokenLogProbs:
    """Per-token log-probability data from a single generation step.

    Attributes:
        token: The token text produced.
        log_prob: Log probability of the chosen token.
        top_k_tokens: Optional top-k alternative tokens with their log probs.
    """

    token: str
    log_prob: float
    top_k_tokens: Optional[dict[str, float]] = None


class StreamEventType(Enum):
    """Types of streaming events an agent can emit."""

    TOKEN = "token"  # Single token produced
    STEP_COMPLETE = "step_complete"  # A full reasoning step is finished
    DONE = "done"  # Generation complete


@dataclass
class AgentStreamEvent:
    """Event emitted by a streaming agent during incremental generation.

    Attributes:
        event_type: Type of event (token, step_complete, done).
        token: The token text (for TOKEN events).
        token_log_probs: Optional log-probability data for this token.
        step_text: Full text of a completed reasoning step (for STEP_COMPLETE).
        step_index: Index of the completed step (0-based).
        final_answer: The final answer (for DONE events).
        agent_id: Identifier of the agent that produced this event.
    """

    event_type: StreamEventType
    token: Optional[str] = None
    token_log_probs: Optional[TokenLogProbs] = None
    step_text: Optional[str] = None
    step_index: Optional[int] = None
    final_answer: Optional[str] = None
    agent_id: Optional[str] = None


class AgentError(Exception):
    """Raised when an agent fails to generate a response.

    Attributes:
        agent_id: Identifier of the failed agent (if available).
        original: The original exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        original: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.agent_id = agent_id
        self.original = original


@dataclass
class PollResult:
    """Result of a poll_agents() call, including successes and failures.

    Attributes:
        responses: Successfully collected AgentResponse objects.
        errors: List of (agent_index, AgentError) for agents that failed.
    """

    responses: list[AgentResponse] = field(default_factory=list)
    errors: list[tuple[int, AgentError]] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return len(self.responses)

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @property
    def all_succeeded(self) -> bool:
        return len(self.errors) == 0


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


@runtime_checkable
class ReasoningAgent(Protocol):
    """Abstract interface for a reasoning-capable LLM agent.

    Any object with a ``generate(prompt: str) -> AgentResponse`` method
    satisfies this protocol.  The protocol is runtime-checkable, so
    ``isinstance(obj, ReasoningAgent)`` works.
    """

    def generate(self, prompt: str) -> AgentResponse:
        """Generate a response to the given prompt.

        Returns:
            AgentResponse with agent_id, reasoning_steps, final_answer,
            and optionally an embedding.
        """
        ...


@runtime_checkable
class StreamingReasoningAgent(Protocol):
    """Extended agent protocol that supports streaming token-by-token output.

    Agents implementing this protocol yield :class:`AgentStreamEvent` objects
    incrementally, enabling real-time entropy computation as tokens arrive.
    """

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        ...

    def stream(self, prompt: str) -> Iterator[AgentStreamEvent]:
        """Stream a response to the given prompt, yielding events incrementally.

        Event sequence: TOKEN* (STEP_COMPLETE TOKEN*)* DONE

        Yields:
            AgentStreamEvent objects — TOKEN events for each token, optional
            STEP_COMPLETE events when a reasoning step finishes, and a final
            DONE event with the complete answer.
        """
        ...


# --------------------------------------------------------------------------
# Polling — concurrent multi-agent dispatch
# --------------------------------------------------------------------------


def poll_agents(
    agents: list[ReasoningAgent],
    prompt: str,
    timeout: Optional[float] = None,
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
) -> PollResult:
    """Poll multiple agents concurrently and collect their responses.

    Uses :class:`~concurrent.futures.ThreadPoolExecutor` to dispatch
    ``generate()`` calls in parallel.  Agents that fail or time out are
    recorded in :attr:`PollResult.errors` rather than crashing the batch
    (unless *fail_fast* is True).

    Args:
        agents: List of ReasoningAgent instances to poll.
        prompt: The prompt to send to each agent.
        timeout: Maximum seconds to wait for all agents.  ``None`` for no limit.
        max_workers: Thread pool size.  Defaults to ``len(agents)`` (one thread
            per agent).
        fail_fast: If True, raise :class:`AgentError` on the first failure
            instead of collecting errors.

    Returns:
        :class:`PollResult` containing successful responses and any errors.

    Raises:
        ValueError: If *agents* is empty.
        AgentError: If *fail_fast* is True and an agent fails.
    """
    if not agents:
        raise ValueError("agents list must be non-empty")

    n_workers = max_workers or len(agents)
    result = PollResult()

    def _call_agent(index: int, agent: ReasoningAgent) -> tuple[int, AgentResponse]:
        return index, agent.generate(prompt)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_index: dict[Future, int] = {}
        for i, agent in enumerate(agents):
            future = executor.submit(_call_agent, i, agent)
            future_to_index[future] = i

        try:
            completed = as_completed(future_to_index, timeout=timeout)
            for future in completed:
                idx = future_to_index[future]
                try:
                    _, response = future.result()
                    result.responses.append(response)
                except Exception as exc:
                    agent_id = getattr(agents[idx], "agent_id", f"agent-{idx}")
                    error = AgentError(
                        f"Agent {agent_id} failed: {exc}",
                        agent_id=str(agent_id),
                        original=exc if isinstance(exc, Exception) else None,
                    )
                    if fail_fast:
                        raise error from exc
                    result.errors.append((idx, error))
                    logger.warning(
                        "Agent {agent_id} failed during polling: {err}",
                        agent_id=agent_id,
                        err=str(exc),
                    )

        except TimeoutError:
            # Collect any remaining futures as timeout errors
            for future, idx in future_to_index.items():
                if not future.done():
                    future.cancel()
                    agent_id = getattr(agents[idx], "agent_id", f"agent-{idx}")
                    error = AgentError(
                        f"Agent {agent_id} timed out after {timeout}s",
                        agent_id=str(agent_id),
                    )
                    result.errors.append((idx, error))
                    logger.warning(
                        "Agent {agent_id} timed out after {timeout}s",
                        agent_id=agent_id,
                        timeout=timeout,
                    )

    logger.info(
        "poll_agents: {n_ok}/{n_total} agents responded successfully",
        n_ok=result.n_success,
        n_total=len(agents),
    )
    return result


# --------------------------------------------------------------------------
# Streaming poll
# --------------------------------------------------------------------------


def poll_agents_streaming(
    agents: list[StreamingReasoningAgent],
    prompt: str,
    on_event: Optional[Callable[[AgentStreamEvent], None]] = None,
    timeout: Optional[float] = None,
    max_workers: Optional[int] = None,
) -> PollResult:
    """Poll streaming agents concurrently, invoking *on_event* for each token.

    Each agent's ``stream()`` method is called in its own thread.  Events are
    forwarded to *on_event* as they arrive (from any agent, in arrival order).
    After all agents finish, their accumulated token streams are assembled
    into :class:`AgentResponse` objects.

    Args:
        agents: List of StreamingReasoningAgent instances.
        prompt: The prompt to send.
        on_event: Optional callback invoked for each :class:`AgentStreamEvent`.
            Useful for real-time entropy tracking.
        timeout: Maximum seconds to wait for all agents.
        max_workers: Thread pool size.

    Returns:
        :class:`PollResult` with assembled responses and any errors.
    """
    if not agents:
        raise ValueError("agents list must be non-empty")

    n_workers = max_workers or len(agents)
    result = PollResult()

    def _stream_agent(agent: StreamingReasoningAgent) -> AgentResponse:
        """Consume a streaming agent's full output and build AgentResponse."""
        reasoning_steps: list[str] = []
        current_tokens: list[str] = []
        final_answer: str = ""
        aid = agent.agent_id

        for event in agent.stream(prompt):
            event.agent_id = aid
            if on_event is not None:
                on_event(event)

            if event.event_type == StreamEventType.TOKEN:
                if event.token is not None:
                    current_tokens.append(event.token)

            elif event.event_type == StreamEventType.STEP_COMPLETE:
                step_text = event.step_text or "".join(current_tokens)
                reasoning_steps.append(step_text)
                current_tokens = []

            elif event.event_type == StreamEventType.DONE:
                # Flush any remaining tokens as a final step
                if current_tokens:
                    reasoning_steps.append("".join(current_tokens))
                    current_tokens = []
                final_answer = event.final_answer or ""

        return AgentResponse(
            agent_id=aid,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_index: dict[Future, int] = {}
        for i, agent in enumerate(agents):
            future = executor.submit(_stream_agent, agent)
            future_to_index[future] = i

        try:
            completed = as_completed(future_to_index, timeout=timeout)
            for future in completed:
                idx = future_to_index[future]
                try:
                    response = future.result()
                    result.responses.append(response)
                except Exception as exc:
                    aid = agents[idx].agent_id
                    error = AgentError(
                        f"Streaming agent {aid} failed: {exc}",
                        agent_id=aid,
                        original=exc if isinstance(exc, Exception) else None,
                    )
                    result.errors.append((idx, error))
                    logger.warning(
                        "Streaming agent {aid} failed: {err}",
                        aid=aid,
                        err=str(exc),
                    )

        except TimeoutError:
            for future, idx in future_to_index.items():
                if not future.done():
                    future.cancel()
                    aid = agents[idx].agent_id
                    error = AgentError(
                        f"Streaming agent {aid} timed out after {timeout}s",
                        agent_id=aid,
                    )
                    result.errors.append((idx, error))

    return result


# --------------------------------------------------------------------------
# Reroll — generate multiple candidates from a single agent
# --------------------------------------------------------------------------


def reroll(
    agent: ReasoningAgent,
    prompt: str,
    n_candidates: int = 3,
    timeout: Optional[float] = None,
    concurrent: bool = True,
) -> list[AgentResponse]:
    """Generate multiple candidate responses from a single agent.

    The caller (typically :class:`RAPTOROrchestrator`) selects the best
    candidate via entropy trajectory or disagreement analysis.

    Args:
        agent: The agent to query repeatedly.
        prompt: The reasoning prompt.
        n_candidates: Number of candidate responses to generate.
        timeout: Maximum seconds for all candidates combined.
        concurrent: If True, generate candidates in parallel threads.

    Returns:
        List of :class:`AgentResponse`, up to *n_candidates*.  Failed
        generations are omitted (logged but not raised).

    Raises:
        ValueError: If *n_candidates* < 1.
        AgentError: If all candidates fail.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    candidates: list[AgentResponse] = []
    errors: list[Exception] = []

    if concurrent and n_candidates > 1:
        executor = ThreadPoolExecutor(max_workers=n_candidates)
        futures = [
            executor.submit(agent.generate, prompt) for _ in range(n_candidates)
        ]
        try:
            completed = as_completed(futures, timeout=timeout)
            for future in completed:
                try:
                    resp = future.result()
                    candidates.append(resp)
                except Exception as exc:
                    errors.append(exc)
                    logger.warning("Reroll candidate failed: {err}", err=str(exc))
        except TimeoutError:
            for f in futures:
                if not f.done():
                    f.cancel()
            logger.warning(
                "Reroll timed out after {timeout}s with {n}/{total} candidates",
                timeout=timeout,
                n=len(candidates),
                total=n_candidates,
            )
        finally:
            # Non-blocking shutdown: don't wait for running threads
            executor.shutdown(wait=False, cancel_futures=True)
    else:
        deadline = (time.monotonic() + timeout) if timeout else None
        for _ in range(n_candidates):
            if deadline and time.monotonic() > deadline:
                logger.warning("Reroll timed out (sequential mode)")
                break
            try:
                resp = agent.generate(prompt)
                candidates.append(resp)
            except Exception as exc:
                errors.append(exc)
                logger.warning("Reroll candidate failed: {err}", err=str(exc))

    if not candidates:
        raise AgentError(
            f"All {n_candidates} reroll candidates failed",
            original=errors[0] if errors else None,
        )

    logger.info(
        "reroll: generated {n}/{total} candidates",
        n=len(candidates),
        total=n_candidates,
    )
    return candidates


# --------------------------------------------------------------------------
# Mock agents for testing
# --------------------------------------------------------------------------


@dataclass
class MockReasoningAgent:
    """A deterministic mock agent for testing.

    Satisfies the :class:`ReasoningAgent` protocol.  Always returns the
    same pre-configured reasoning steps and final answer, regardless of
    the prompt.
    """

    agent_id: str
    reasoning_steps: list[str]
    final_answer: str

    def generate(self, prompt: str) -> AgentResponse:
        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=list(self.reasoning_steps),
            final_answer=self.final_answer,
        )


@dataclass
class MockStreamingAgent:
    """A mock streaming agent for testing.

    Yields tokens from pre-configured reasoning steps, emitting
    STEP_COMPLETE events between steps and a DONE event at the end.
    """

    _agent_id: str
    reasoning_steps: list[str]
    final_answer: str
    tokens_per_step: int = 3  # How many TOKEN events per step

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def stream(self, prompt: str) -> Iterator[AgentStreamEvent]:
        for step_idx, step_text in enumerate(self.reasoning_steps):
            # Split step into tokens
            words = step_text.split()
            for word in words:
                yield AgentStreamEvent(
                    event_type=StreamEventType.TOKEN,
                    token=word + " ",
                    token_log_probs=TokenLogProbs(
                        token=word, log_prob=-0.5
                    ),
                )
            yield AgentStreamEvent(
                event_type=StreamEventType.STEP_COMPLETE,
                step_text=step_text,
                step_index=step_idx,
            )
        yield AgentStreamEvent(
            event_type=StreamEventType.DONE,
            final_answer=self.final_answer,
        )


@dataclass
class MockFailingAgent:
    """A mock agent that raises an exception on generate().

    Useful for testing error handling in poll_agents() and reroll().
    """

    agent_id: str
    error_message: str = "Mock failure"

    def generate(self, prompt: str) -> AgentResponse:
        raise RuntimeError(self.error_message)


@dataclass
class MockSlowAgent:
    """A mock agent that sleeps before responding.

    Useful for testing timeout handling.
    """

    agent_id: str
    delay_seconds: float
    reasoning_steps: list[str] = field(default_factory=lambda: ["thinking"])
    final_answer: str = "delayed answer"

    def generate(self, prompt: str) -> AgentResponse:
        time.sleep(self.delay_seconds)
        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=list(self.reasoning_steps),
            final_answer=self.final_answer,
        )


@dataclass
class MockVariableAgent:
    """A mock agent that returns different responses on successive calls.

    Useful for testing reroll() — each call returns a different candidate.
    """

    agent_id: str
    responses: list[tuple[list[str], str]]  # [(reasoning_steps, final_answer), ...]
    _call_count: int = field(default=0, init=False, repr=False)

    def generate(self, prompt: str) -> AgentResponse:
        idx = self._call_count % len(self.responses)
        self._call_count += 1
        steps, answer = self.responses[idx]
        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=list(steps),
            final_answer=answer,
        )
