#!/usr/bin/env python3
"""How to implement a custom ReasoningAgent for RAPTOR.

RAPTOR's agent protocol is simple: any object with a
``generate(prompt: str) -> AgentResponse`` method works.

This example shows three patterns:
  1. A class-based agent with configurable behavior
  2. A probabilistic agent that introduces controlled randomness
  3. Plugging custom agents into the RAPTOR pipeline

Run with:
    python examples/custom_agent.py
"""

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from raptor.agents import ReasoningAgent, poll_agents
from raptor.config import Config, DisagreementConfig
from raptor.disagreement import AgentResponse
from raptor.integration import run_with_raptor


# ══════════════════════════════════════════════════════════════════════
# Pattern 1: Class-based custom agent
# ══════════════════════════════════════════════════════════════════════


class CalculatorAgent:
    """A custom agent that tries to evaluate simple math expressions.

    This demonstrates the minimal interface RAPTOR requires:
    just ``generate(prompt) -> AgentResponse``.
    """

    def __init__(self, agent_id: str, show_work: bool = True):
        self.agent_id = agent_id
        self.show_work = show_work

    def generate(self, prompt: str) -> AgentResponse:
        """Parse a simple math expression from the prompt and evaluate it."""
        steps = []

        # Step 1: Extract numbers from the prompt
        import re
        numbers = re.findall(r"\d+", prompt)
        steps.append(f"Extracted numbers from prompt: {numbers}")

        if len(numbers) >= 2:
            # Step 2: Try basic operations
            a, b = int(numbers[0]), int(numbers[1])

            if "+" in prompt or "sum" in prompt.lower() or "add" in prompt.lower():
                result = a + b
                steps.append(f"Detected addition: {a} + {b} = {result}")
            elif "×" in prompt or "*" in prompt or "times" in prompt.lower() or "multiply" in prompt.lower():
                result = a * b
                steps.append(f"Detected multiplication: {a} × {b} = {result}")
            elif "-" in prompt or "minus" in prompt.lower() or "subtract" in prompt.lower():
                result = a - b
                steps.append(f"Detected subtraction: {a} - {b} = {result}")
            elif "/" in prompt or "divide" in prompt.lower():
                result = a // b if b != 0 else 0
                steps.append(f"Detected division: {a} / {b} = {result}")
            else:
                result = a * b  # default to multiplication
                steps.append(f"Defaulting to multiplication: {a} × {b} = {result}")

            answer = str(result)
        else:
            steps.append("Could not extract enough numbers for computation.")
            answer = "unknown"

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=steps,
            final_answer=answer,
        )


# ══════════════════════════════════════════════════════════════════════
# Pattern 2: Probabilistic agent (simulates LLM uncertainty)
# ══════════════════════════════════════════════════════════════════════


class NoisyAgent:
    """An agent that has a configurable probability of giving a wrong answer.

    This is useful for testing how RAPTOR handles disagreement —
    you can control the error rate and see how the orchestrator responds.
    """

    def __init__(
        self,
        agent_id: str,
        correct_answer: str,
        wrong_answer: str,
        error_rate: float = 0.2,
        seed: int | None = None,
    ):
        self.agent_id = agent_id
        self.correct_answer = correct_answer
        self.wrong_answer = wrong_answer
        self.error_rate = error_rate
        self._rng = random.Random(seed)

    def generate(self, prompt: str) -> AgentResponse:
        """Return the correct answer most of the time, wrong answer sometimes."""
        is_wrong = self._rng.random() < self.error_rate

        if is_wrong:
            steps = [
                "Let me think about this...",
                f"I believe the answer is {self.wrong_answer}.",
            ]
            answer = self.wrong_answer
        else:
            steps = [
                "Analyzing the problem carefully.",
                "Applying the relevant method.",
                f"The answer is {self.correct_answer}.",
            ]
            answer = self.correct_answer

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=steps,
            final_answer=answer,
        )


# ══════════════════════════════════════════════════════════════════════
# Pattern 3: Deterministic hash-based agent (reproducible without RNG)
# ══════════════════════════════════════════════════════════════════════


class HashAgent:
    """An agent whose answer is deterministically derived from the prompt.

    Uses a hash function to select from a set of candidate answers.
    The same prompt always produces the same answer, but different
    agents (different agent_ids) produce different answers for the
    same prompt — simulating diversity without randomness.
    """

    def __init__(self, agent_id: str, candidates: list[str]):
        self.agent_id = agent_id
        self.candidates = candidates

    def generate(self, prompt: str) -> AgentResponse:
        # Hash prompt + agent_id to select a deterministic answer
        key = f"{self.agent_id}:{prompt}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        idx = h % len(self.candidates)
        answer = self.candidates[idx]

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=[
                f"Considering the prompt: {prompt[:50]}...",
                f"My analysis yields: {answer}",
            ],
            final_answer=answer,
        )


# ══════════════════════════════════════════════════════════════════════
# Protocol verification
# ══════════════════════════════════════════════════════════════════════


def verify_protocol(agent: object) -> bool:
    """Check that an agent satisfies the ReasoningAgent protocol."""
    return isinstance(agent, ReasoningAgent)


# ══════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    prompt = "What is 6 × 7?"

    # ── Demo 1: CalculatorAgent ──────────────────────────────────────
    print("=" * 60)
    print("Demo 1: CalculatorAgent")
    print("=" * 60)

    calc = CalculatorAgent(agent_id="calc-1")
    print(f"  Satisfies ReasoningAgent protocol: {verify_protocol(calc)}")

    result = calc.generate(prompt)
    print(f"  Answer: {result.final_answer}")
    for step in result.reasoning_steps:
        print(f"    → {step}")
    print()

    # ── Demo 2: NoisyAgent ensemble ──────────────────────────────────
    print("=" * 60)
    print("Demo 2: NoisyAgent ensemble (20% error rate)")
    print("=" * 60)

    noisy_agents = [
        NoisyAgent(
            agent_id=f"noisy-{i}",
            correct_answer="42",
            wrong_answer="36",
            error_rate=0.2,
            seed=i * 100,
        )
        for i in range(5)
    ]

    poll_result = poll_agents(noisy_agents, prompt)
    for resp in poll_result.responses:
        marker = "✓" if resp.final_answer == "42" else "✗"
        print(f"  {marker} {resp.agent_id}: {resp.final_answer}")
    print()

    # ── Demo 3: Plugging into RAPTOR ─────────────────────────────────
    print("=" * 60)
    print("Demo 3: Custom agents in the full RAPTOR pipeline")
    print("=" * 60)

    # Mix agent types — RAPTOR doesn't care, as long as they have generate()
    mixed_agents = [
        CalculatorAgent(agent_id="calc-main"),
        NoisyAgent("noisy-1", "42", "36", 0.1, seed=42),
        NoisyAgent("noisy-2", "42", "36", 0.1, seed=99),
        HashAgent("hash-1", ["42", "42", "42", "36"]),
        HashAgent("hash-2", ["42", "42", "42", "43"]),
    ]

    config = Config(
        max_rerolls=2,
        max_steps=5,
        log_signal_history=False,
        disagreement=DisagreementConfig(mode="embed", n_agents=5),
    )

    result = run_with_raptor(
        agents=mixed_agents,
        prompt=prompt,
        config=config,
    )

    print(f"  Final answer: {result.final_answer!r}")
    print(f"  Steps taken:  {result.steps_taken}")
    print(f"  Action path:  {result.context.action_history}")
    print()

    # ── Demo 4: Protocol check ───────────────────────────────────────
    print("=" * 60)
    print("Demo 4: Protocol verification")
    print("=" * 60)

    test_objects = [
        ("CalculatorAgent", CalculatorAgent("test")),
        ("NoisyAgent", NoisyAgent("test", "a", "b")),
        ("HashAgent", HashAgent("test", ["a"])),
        ("plain dict", {"generate": lambda p: None}),
        ("string", "not an agent"),
    ]

    for name, obj in test_objects:
        print(f"  {name:20s} → ReasoningAgent? {verify_protocol(obj)}")

    print()
    print("Done! All custom agents worked with the RAPTOR pipeline.")


if __name__ == "__main__":
    main()
