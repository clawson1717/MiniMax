"""Integration utilities — LLM adapters, context management, and high-level entry point.

Provides:
  - OpenAIAgent, AnthropicAgent, LocalLLMAgent — ReasoningAgent-compatible
    adapters for major LLM providers
  - RAPTORContext — serializable context object holding prompt, conversation
    history, signal history, action history, and config
  - RAPTORResult — return type from run_with_raptor()
  - run_with_raptor() — high-level entry point that wires together the full
    RAPTOR pipeline: poll → signal → decide → act loop

All HTTP communication is routed through an injectable ``HttpClient`` protocol
so that tests can mock responses without touching the network.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np
from loguru import logger

from raptor.agents import (
    AgentError,
    ReasoningAgent,
    poll_agents,
    reroll,
    PollResult,
)
from raptor.config import Config, OrchestrationAction
from raptor.disagreement import AgentResponse
from raptor.orchestrator import RAPTOROrchestrator, OrchestrationDecision


# ══════════════════════════════════════════════════════════════════════
# HTTP Client Protocol (injectable for testing)
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class HttpClient(Protocol):
    """Minimal HTTP POST interface — swap in a mock for testing."""

    def post(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a POST request, return parsed JSON response body."""
        ...


class UrllibHttpClient:
    """Default HTTP client using stdlib ``urllib.request``."""

    def post(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        kw: dict[str, Any] = {}
        if timeout is not None:
            kw["timeout"] = timeout
        with urllib.request.urlopen(req, **kw) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ══════════════════════════════════════════════════════════════════════
# Chain-of-thought step extraction
# ══════════════════════════════════════════════════════════════════════

# Matches lines that start with a number followed by a period/paren/bracket,
# or lines starting with bullet-like markers (-, *)
_STEP_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)\]:]|[-*•])\s+", re.MULTILINE
)


def _extract_reasoning_steps(text: str) -> list[str]:
    """Split LLM response text into reasoning steps.

    Heuristic priority:
      1. Numbered steps (``1. ...``, ``1) ...``, ``1: ...``)
      2. Bullet-list items (``- ...``, ``* ...``, ``• ...``)
      3. Non-empty lines separated by blank lines
      4. Every non-empty line

    Returns at least one step (the full text) if no structure is detected.
    """
    # Try numbered / bulleted splits
    matches = list(_STEP_PATTERN.finditer(text))
    if len(matches) >= 2:
        steps: list[str] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            step_text = text[start:end].strip()
            if step_text:
                steps.append(step_text)
        if steps:
            return steps

    # Try paragraph splits (blocks separated by blank lines)
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    # Fall back to non-empty lines
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines if lines else [text.strip()] if text.strip() else ["(empty)"]


def _extract_logprobs_entropy(
    logprobs_data: list[dict[str, Any]] | None,
) -> list[float] | None:
    """Extract per-token entropy from OpenAI-style logprobs response.

    Each entry is expected to have ``logprob`` and optionally ``top_logprobs``
    (list of dicts with ``logprob`` values).  Returns per-token entropy values,
    or None if logprobs_data is empty/absent.
    """
    if not logprobs_data:
        return None
    entropies: list[float] = []
    for token_info in logprobs_data:
        top = token_info.get("top_logprobs")
        if top and isinstance(top, list):
            log_ps = [t.get("logprob", 0.0) for t in top if isinstance(t, dict)]
            if log_ps:
                probs = np.exp(np.array(log_ps, dtype=np.float64))
                probs = probs / probs.sum()
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log(probs)))
                entropies.append(entropy)
    return entropies if entropies else None


# ══════════════════════════════════════════════════════════════════════
# LLM Adapters
# ══════════════════════════════════════════════════════════════════════


class OpenAIAgent:
    """ReasoningAgent adapter for OpenAI chat completions API.

    Wraps ``POST /v1/chat/completions`` and extracts chain-of-thought
    reasoning steps from the response content.

    Args:
        model: Model name (e.g. ``"gpt-4o"``).
        api_key: OpenAI API key.
        agent_id: Identifier for this agent instance.
        base_url: API base URL (override for proxies / Azure).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        system_prompt: Optional system message prepended to every request.
        request_logprobs: If True, request ``logprobs`` and ``top_logprobs``.
        top_logprobs: Number of top logprobs to request (1-20).
        http_client: Injectable HTTP client (defaults to stdlib urllib).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        agent_id: str | None = None,
        *,
        base_url: str = "https://api.openai.com",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        request_logprobs: bool = False,
        top_logprobs: int = 5,
        http_client: HttpClient | None = None,
        timeout: float | None = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.agent_id = agent_id or f"openai-{model}"
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.request_logprobs = request_logprobs
        self.top_logprobs = top_logprobs
        self._http = http_client or UrllibHttpClient()
        self.timeout = timeout

    def generate(self, prompt: str) -> AgentResponse:
        """Send prompt to OpenAI and return structured AgentResponse."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = self._http.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                body=body,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise AgentError(
                f"OpenAI API request failed: {exc}",
                agent_id=self.agent_id,
                original=exc if isinstance(exc, Exception) else None,
            ) from exc

        # Parse response
        try:
            choice = resp["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(
                f"Unexpected OpenAI response structure: {exc}",
                agent_id=self.agent_id,
                original=exc,
            ) from exc

        reasoning_steps = _extract_reasoning_steps(content)

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=reasoning_steps,
            final_answer=content,
        )


class AnthropicAgent:
    """ReasoningAgent adapter for Anthropic Messages API.

    Wraps ``POST /v1/messages`` and extracts chain-of-thought reasoning
    steps from the response content.

    Args:
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
        api_key: Anthropic API key.
        agent_id: Identifier for this agent instance.
        base_url: API base URL.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        system_prompt: Optional system message.
        http_client: Injectable HTTP client.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        agent_id: str | None = None,
        *,
        base_url: str = "https://api.anthropic.com",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        http_client: HttpClient | None = None,
        timeout: float | None = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.agent_id = agent_id or f"anthropic-{model}"
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._http = http_client or UrllibHttpClient()
        self.timeout = timeout

    def generate(self, prompt: str) -> AgentResponse:
        """Send prompt to Anthropic and return structured AgentResponse."""
        messages = [{"role": "user", "content": prompt}]

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.system_prompt:
            body["system"] = self.system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            resp = self._http.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                body=body,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise AgentError(
                f"Anthropic API request failed: {exc}",
                agent_id=self.agent_id,
                original=exc if isinstance(exc, Exception) else None,
            ) from exc

        # Parse response — Anthropic returns content as a list of blocks
        try:
            content_blocks = resp["content"]
            text_parts = [
                block["text"]
                for block in content_blocks
                if block.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(
                f"Unexpected Anthropic response structure: {exc}",
                agent_id=self.agent_id,
                original=exc,
            ) from exc

        reasoning_steps = _extract_reasoning_steps(content)

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=reasoning_steps,
            final_answer=content,
        )


class LocalLLMAgent:
    """ReasoningAgent adapter for local OpenAI-compatible endpoints.

    Works with vLLM, llama.cpp server, Ollama, LMStudio, or any endpoint
    that speaks the OpenAI chat completions wire format.

    Args:
        endpoint: Full base URL (e.g. ``"http://localhost:8000"``).
        model: Model name/path as expected by the local server.
        agent_id: Identifier for this agent instance.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        system_prompt: Optional system message.
        api_key: Optional API key (some local servers require one).
        request_logprobs: If True, request logprobs.
        top_logprobs: Number of top logprobs.
        http_client: Injectable HTTP client.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        model: str = "local-model",
        agent_id: str | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        api_key: str | None = None,
        request_logprobs: bool = False,
        top_logprobs: int = 5,
        http_client: HttpClient | None = None,
        timeout: float | None = 120.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.agent_id = agent_id or f"local-{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.request_logprobs = request_logprobs
        self.top_logprobs = top_logprobs
        self._http = http_client or UrllibHttpClient()
        self.timeout = timeout

    def generate(self, prompt: str) -> AgentResponse:
        """Send prompt to local LLM and return structured AgentResponse."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = self._http.post(
                f"{self.endpoint}/v1/chat/completions",
                headers=headers,
                body=body,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise AgentError(
                f"Local LLM request failed: {exc}",
                agent_id=self.agent_id,
                original=exc if isinstance(exc, Exception) else None,
            ) from exc

        # Parse — same format as OpenAI
        try:
            choice = resp["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(
                f"Unexpected local LLM response structure: {exc}",
                agent_id=self.agent_id,
                original=exc,
            ) from exc

        reasoning_steps = _extract_reasoning_steps(content)

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=reasoning_steps,
            final_answer=content,
        )


# ══════════════════════════════════════════════════════════════════════
# RAPTOR Context
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ContextEntry:
    """A single entry in the conversation / action history."""

    step: int
    role: str  # "user" | "assistant" | "system" | "action"
    content: str
    action: str | None = None  # OrchestrationAction value, if role == "action"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RAPTORContext:
    """Serializable context object for a RAPTOR reasoning session.

    Holds the prompt, conversation history, signal history (from
    orchestrator decisions), and action history.  Can be serialized
    to/from JSON for persistence or replay.

    Attributes:
        prompt: The original user prompt.
        config: RAPTOR configuration used for this session.
        conversation_history: Ordered list of conversation entries.
        signal_history: List of signal vector dicts from each step.
        action_history: Ordered list of actions taken (OrchestrationAction values).
        decisions: Full OrchestrationDecision records (not serialized by default).
        metadata: Arbitrary key-value metadata.
    """

    prompt: str
    config: Config = field(default_factory=Config)
    conversation_history: list[ContextEntry] = field(default_factory=list)
    signal_history: list[dict[str, Any]] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)
    decisions: list[OrchestrationDecision] = field(default_factory=list, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_entry(
        self,
        role: str,
        content: str,
        action: str | None = None,
    ) -> None:
        """Append an entry to conversation history."""
        step = len(self.conversation_history)
        self.conversation_history.append(
            ContextEntry(step=step, role=role, content=content, action=action)
        )

    def add_action(self, action: OrchestrationAction) -> None:
        """Record an orchestration action."""
        self.action_history.append(action.value)
        self.add_entry("action", action.value, action=action.value)

    def add_decision(self, decision: OrchestrationDecision) -> None:
        """Record a full orchestration decision with signal history."""
        self.decisions.append(decision)
        self.signal_history.append(decision.signal_vector.to_dict())
        self.add_action(decision.action)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to a JSON-compatible dict (excluding decisions)."""
        return {
            "prompt": self.prompt,
            "conversation_history": [
                {
                    "step": e.step,
                    "role": e.role,
                    "content": e.content,
                    "action": e.action,
                    "timestamp": e.timestamp,
                }
                for e in self.conversation_history
            ],
            "signal_history": self.signal_history,
            "action_history": self.action_history,
            "metadata": self.metadata,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Config | None = None) -> "RAPTORContext":
        """Deserialize from a dict (inverse of to_dict)."""
        ctx = cls(
            prompt=data["prompt"],
            config=config or Config(),
            signal_history=data.get("signal_history", []),
            action_history=data.get("action_history", []),
            metadata=data.get("metadata", {}),
        )
        for entry_data in data.get("conversation_history", []):
            ctx.conversation_history.append(
                ContextEntry(
                    step=entry_data["step"],
                    role=entry_data["role"],
                    content=entry_data["content"],
                    action=entry_data.get("action"),
                    timestamp=entry_data.get("timestamp", 0.0),
                )
            )
        return ctx


# ══════════════════════════════════════════════════════════════════════
# RAPTOR Result
# ══════════════════════════════════════════════════════════════════════


@dataclass
class RAPTORResult:
    """Return type from ``run_with_raptor()``.

    Attributes:
        final_answer: The best answer selected by the RAPTOR loop.
        context: Full RAPTORContext with history and signals.
        steps_taken: Number of orchestration steps executed.
        final_action: The terminal action that ended the loop.
        all_responses: All agent responses collected during the run.
        escalated: True if the loop ended via ESCALATE.
        stopped: True if the loop ended via STOP.
    """

    final_answer: str
    context: RAPTORContext
    steps_taken: int
    final_action: OrchestrationAction
    all_responses: list[AgentResponse] = field(default_factory=list)
    escalated: bool = False
    stopped: bool = False


# ══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════


def run_with_raptor(
    agents: list[ReasoningAgent],
    prompt: str,
    config: Config | None = None,
    max_steps: int | None = None,
    poll_timeout: float | None = None,
    on_decision: Callable[[OrchestrationDecision, int], None] | None = None,
    verify_prompt_fn: Callable[[str, str], str] | None = None,
) -> RAPTORResult:
    """Run the full RAPTOR reasoning pipeline.

    This is the high-level entry point that wires together all RAPTOR
    components: agent polling, entropy tracking, disagreement monitoring,
    utility scoring, and orchestration.

    Pipeline per step:
      1. Poll all agents with the prompt
      2. Feed responses through the orchestrator (signals → utility → action)
      3. Execute the selected action:
         - **RESPOND**: accept the best answer, return result
         - **REROLL**: regenerate via ``reroll()`` on the first agent,
           then loop back to step 2 with new responses
         - **VERIFY**: construct a verification prompt and loop
         - **RETRIEVE**: placeholder — logs and continues
         - **ESCALATE**: flag for human review, return result
         - **STOP**: halt, return best available answer

    Args:
        agents: List of ReasoningAgent instances to poll.
        prompt: The reasoning question / task.
        config: RAPTOR configuration (uses defaults if None).
        max_steps: Override ``config.max_steps`` for max orchestration steps.
        poll_timeout: Timeout for each poll_agents call (seconds).
        on_decision: Optional callback invoked after each decision with
            ``(decision, step_number)``.
        verify_prompt_fn: Optional function ``(original_prompt, current_answer) → str``
            that generates a verification prompt.  Defaults to a simple
            "Please verify..." wrapper.

    Returns:
        :class:`RAPTORResult` with the final answer and full context.

    Raises:
        ValueError: If agents list is empty.
        AgentError: If all agents fail on every step.
    """
    if not agents:
        raise ValueError("agents list must be non-empty")

    cfg = config or Config()
    steps_limit = max_steps if max_steps is not None else cfg.max_steps

    # Create orchestrator and context
    orchestrator = RAPTOROrchestrator(cfg)
    context = RAPTORContext(prompt=prompt, config=cfg)
    context.add_entry("user", prompt)

    all_responses: list[AgentResponse] = []
    current_prompt = prompt
    best_answer = ""
    step = 0

    while step < steps_limit:
        # ── 1. Poll agents ───────────────────────────────────────────
        poll_result = poll_agents(
            agents, current_prompt, timeout=poll_timeout
        )

        if not poll_result.responses:
            logger.error(
                "All agents failed at step {step}. Errors: {errors}",
                step=step,
                errors=[str(e) for _, e in poll_result.errors],
            )
            # If we have a previous answer, return it; otherwise raise
            if best_answer:
                return RAPTORResult(
                    final_answer=best_answer,
                    context=context,
                    steps_taken=step,
                    final_action=OrchestrationAction.STOP,
                    all_responses=all_responses,
                    stopped=True,
                )
            raise AgentError(
                f"All {len(agents)} agents failed at step {step}",
                original=poll_result.errors[0][1] if poll_result.errors else None,
            )

        responses = poll_result.responses
        all_responses.extend(responses)

        # Track best answer (first response's final_answer as fallback)
        if responses:
            best_answer = responses[0].final_answer

        # ── 2. Orchestrator step ─────────────────────────────────────
        decision = orchestrator.step(current_prompt, responses)
        context.add_decision(decision)

        if on_decision:
            on_decision(decision, step)

        logger.info(
            "Step {step}: action={action} utility={utility:.3f} reason={reason}",
            step=step,
            action=decision.action.value,
            utility=decision.utility_score,
            reason=decision.reason[:80],
        )

        # ── 3. Execute action ────────────────────────────────────────
        action = decision.action

        if action == OrchestrationAction.RESPOND:
            # Accept current best answer
            # Select the best response via orchestrator's reroll_with_selection
            best_resp = orchestrator.reroll_with_selection(responses)
            best_answer = best_resp.final_answer
            context.add_entry("assistant", best_answer)
            return RAPTORResult(
                final_answer=best_answer,
                context=context,
                steps_taken=step + 1,
                final_action=OrchestrationAction.RESPOND,
                all_responses=all_responses,
            )

        elif action == OrchestrationAction.REROLL:
            # Reroll using the first agent, then re-evaluate
            try:
                reroll_candidates = reroll(
                    agents[0],
                    current_prompt,
                    n_candidates=min(3, cfg.max_rerolls),
                    timeout=poll_timeout,
                )
                # Use the orchestrator to pick the best candidate
                best_resp = orchestrator.reroll_with_selection(reroll_candidates)
                best_answer = best_resp.final_answer
                all_responses.extend(reroll_candidates)
                context.add_entry("assistant", f"[REROLL] {best_answer}")
                # Replace responses for next iteration
                responses = reroll_candidates
            except AgentError as exc:
                logger.warning("Reroll failed: {err}", err=str(exc))
                context.add_entry("system", f"[REROLL FAILED] {exc}")

        elif action == OrchestrationAction.VERIFY:
            # Generate a verification prompt
            if verify_prompt_fn:
                current_prompt = verify_prompt_fn(prompt, best_answer)
            else:
                current_prompt = (
                    f"Please verify the following answer to the question.\n\n"
                    f"Question: {prompt}\n\n"
                    f"Proposed answer: {best_answer}\n\n"
                    f"Is this answer correct? If not, provide the correct answer "
                    f"with step-by-step reasoning."
                )
            context.add_entry("system", f"[VERIFY] {current_prompt[:200]}...")

        elif action == OrchestrationAction.RETRIEVE:
            # Placeholder — in a full system this would call a retrieval service
            logger.info("RETRIEVE action triggered (placeholder — no retrieval backend)")
            context.add_entry("system", "[RETRIEVE] No retrieval backend configured")
            # Continue with the same prompt

        elif action == OrchestrationAction.ESCALATE:
            # Flag for human review
            context.add_entry("system", "[ESCALATE] Flagged for human review")
            return RAPTORResult(
                final_answer=best_answer,
                context=context,
                steps_taken=step + 1,
                final_action=OrchestrationAction.ESCALATE,
                all_responses=all_responses,
                escalated=True,
            )

        elif action == OrchestrationAction.STOP:
            # Halt — insufficient confidence
            context.add_entry("system", "[STOP] Halted — low confidence")
            return RAPTORResult(
                final_answer=best_answer,
                context=context,
                steps_taken=step + 1,
                final_action=OrchestrationAction.STOP,
                all_responses=all_responses,
                stopped=True,
            )

        step += 1

    # Max steps exhausted — return best available answer
    logger.warning(
        "Max steps ({max_steps}) exhausted, returning best available answer",
        max_steps=steps_limit,
    )
    context.add_entry("system", f"[MAX_STEPS] Exhausted {steps_limit} steps")
    return RAPTORResult(
        final_answer=best_answer,
        context=context,
        steps_taken=step,
        final_action=OrchestrationAction.RESPOND,
        all_responses=all_responses,
    )
