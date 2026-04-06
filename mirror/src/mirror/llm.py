"""LLM abstraction layer for MIRROR.

Provides a unified interface to language models via litellm,
plus a MockLLMClient for deterministic testing.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, prompt: str, model: str | None = None) -> str:
        """Send a single prompt and return the completion text."""
        ...

    @abstractmethod
    def complete_chat(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Send a chat-formatted message list and return the response text."""
        ...


class LLMClient(BaseLLMClient):
    """Production LLM client backed by litellm.

    Args:
        default_model: Default model identifier (e.g. 'gpt-4', 'claude-3-opus').
        **kwargs: Extra keyword arguments forwarded to litellm.completion().
    """

    def __init__(self, default_model: str = "gpt-4", **kwargs: Any) -> None:
        self.default_model = default_model
        self._extra_kwargs = kwargs

    def complete(self, prompt: str, model: str | None = None) -> str:
        """Send a single prompt completion request via litellm."""
        import litellm  # lazy import so tests don't need litellm configured

        model = model or self.default_model
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **self._extra_kwargs,
        )
        return response.choices[0].message.content  # type: ignore[union-attr]

    def complete_chat(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Send a chat completion request via litellm."""
        import litellm

        model = model or self.default_model
        response = litellm.completion(
            model=model,
            messages=messages,
            **self._extra_kwargs,
        )
        return response.choices[0].message.content  # type: ignore[union-attr]


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client for testing.

    Recognizes patterns in the prompt to return appropriate mock responses
    for each probe type (forward, reverse, cross-model).
    """

    def __init__(self, default_model: str = "mock-model") -> None:
        self.default_model = default_model
        self.call_log: list[dict[str, Any]] = []

    def complete(self, prompt: str, model: str | None = None) -> str:
        """Return a deterministic mock response based on prompt patterns."""
        model = model or self.default_model
        self.call_log.append({"type": "complete", "prompt": prompt, "model": model})
        return self._route(prompt)

    def complete_chat(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Return a deterministic mock response based on the last user message."""
        model = model or self.default_model
        self.call_log.append(
            {"type": "complete_chat", "messages": messages, "model": model}
        )
        # Use the last user message for routing
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        return self._route(last_user)

    def _route(self, text: str) -> str:
        """Route to the appropriate mock response based on text content."""
        text_lower = text.lower()

        # Forward consistency test — asks whether CoT leads to answer
        if "does this chain-of-thought" in text_lower or "forward" in text_lower:
            return self._forward_response(text)

        # Reverse consistency test — asks to reconstruct input from CoT + answer
        if "reconstruct" in text_lower or "original input" in text_lower:
            return self._reverse_response(text)

        # Cross-model test — asks to follow CoT and produce answer
        if "follow" in text_lower and "chain-of-thought" in text_lower:
            return self._cross_model_response(text)

        # Default fallback
        return "Mock response: I agree with the reasoning."

    def _forward_response(self, text: str) -> str:
        """Mock forward consistency evaluation."""
        # Check for explicit markers of bad reasoning in the sample data.
        # Note: we only check "wrong" (not "inconsistent") because the
        # prompt template itself contains the word "INCONSISTENT" as an
        # instruction option.
        if "wrong" in text.lower():
            return (
                "VERDICT: INCONSISTENT\n"
                "SCORE: 0.2\n"
                "The chain-of-thought does not logically lead to the stated answer."
            )
        return (
            "VERDICT: CONSISTENT\n"
            "SCORE: 0.9\n"
            "The chain-of-thought logically supports the answer."
        )

    def _reverse_response(self, text: str) -> str:
        """Mock reverse consistency evaluation."""
        # Try to extract and echo back input-like content
        if "inconsistent" in text.lower() or "wrong" in text.lower():
            return (
                "RECONSTRUCTED_INPUT: Something completely different\n"
                "SIMILARITY: 0.2\n"
                "The reconstructed input does not match the original."
            )
        return (
            "RECONSTRUCTED_INPUT: What is 23 * 47?\n"
            "SIMILARITY: 0.85\n"
            "The reconstructed input closely matches the original constraints."
        )

    def _cross_model_response(self, text: str) -> str:
        """Mock cross-model consistency evaluation."""
        # Extract the expected answer from the prompt
        answer_match = re.search(r"expected answer:\s*(\S+)", text, re.IGNORECASE)
        if "wrong" in text.lower():
            return (
                "DERIVED_ANSWER: 999\n"
                "MATCH: NO\n"
                "Following the chain-of-thought, I arrive at a different answer."
            )
        answer = answer_match.group(1) if answer_match else "1081"
        return (
            f"DERIVED_ANSWER: {answer}\n"
            "MATCH: YES\n"
            "Following the chain-of-thought step by step, I reach the same answer."
        )
