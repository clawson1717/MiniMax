#!/usr/bin/env python3
"""Basic RAPTOR usage — polling + orchestration with mock agents.

This example demonstrates the core RAPTOR loop without requiring any
API keys or external services. It uses MockReasoningAgent to simulate
a 5-agent ensemble answering a math question, then runs the full
RAPTOR pipeline (entropy tracking, disagreement monitoring, utility-
guided orchestration).

Run with:
    python examples/basic_usage.py
"""

import sys
from pathlib import Path

# Ensure the project src is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from raptor.config import Config, DisagreementConfig, UtilityConfig
from raptor.agents import MockReasoningAgent, poll_agents
from raptor.disagreement import DisagreementMonitor
from raptor.orchestrator import RAPTOROrchestrator
from raptor.integration import run_with_raptor


def main() -> None:
    # ── 1. Create mock agents ────────────────────────────────────────
    # Simulate an ensemble where most agents agree on "42" but one
    # agent diverges — exactly the scenario RAPTOR is designed to handle.

    agents = [
        MockReasoningAgent(
            agent_id="agent-0",
            reasoning_steps=[
                "The problem asks for 6 × 7.",
                "6 × 7 = 42.",
            ],
            final_answer="42",
        ),
        MockReasoningAgent(
            agent_id="agent-1",
            reasoning_steps=[
                "We need to compute 6 times 7.",
                "I know 6 × 7 is 42.",
            ],
            final_answer="42",
        ),
        MockReasoningAgent(
            agent_id="agent-2",
            reasoning_steps=[
                "Let me calculate 6 × 7 step by step.",
                "6 × 7 = 42.",
            ],
            final_answer="42",
        ),
        MockReasoningAgent(
            agent_id="agent-3",
            reasoning_steps=[
                "6 × 7... I think that's 43?",
                "Wait, let me recalculate. Actually it's 42.",
            ],
            final_answer="42",
        ),
        # This agent disagrees — it provides a wrong answer
        MockReasoningAgent(
            agent_id="agent-4-divergent",
            reasoning_steps=[
                "6 × 7 is the same as 6 + 6 + 6 + 6 + 6 + 6 + 6.",
                "That gives me 36.",  # wrong reasoning
            ],
            final_answer="36",
        ),
    ]

    prompt = "What is 6 × 7?"

    # ── 2. Manual polling (see what individual agents return) ────────
    print("=" * 60)
    print("Step 1: Manual agent polling")
    print("=" * 60)
    poll_result = poll_agents(agents, prompt)

    for resp in poll_result.responses:
        print(f"  {resp.agent_id}: answer={resp.final_answer!r}")
    print(f"  → {poll_result.n_success}/{len(agents)} agents succeeded")
    print()

    # ── 3. Disagreement monitoring ──────────────────────────────────
    print("=" * 60)
    print("Step 2: Disagreement analysis")
    print("=" * 60)

    monitor = DisagreementMonitor(DisagreementConfig(mode="embed"))
    signal = monitor.compute_signal(poll_result.responses)

    print(f"  Confidence score:  {signal.confidence_score:.3f}")
    print(f"  Disagreement tier: {signal.disagreement_tier}")
    print(f"  Evidence overlap:  {signal.evidence_overlap:.3f}")
    print()

    # ── 4. Full RAPTOR pipeline ─────────────────────────────────────
    print("=" * 60)
    print("Step 3: Full RAPTOR pipeline")
    print("=" * 60)

    config = Config(
        max_rerolls=3,
        max_steps=10,
        log_signal_history=False,  # Don't write log files for this demo
        disagreement=DisagreementConfig(mode="embed"),
    )

    # on_decision callback prints each orchestration step
    def on_decision(decision, step):
        print(f"  Step {step}: action={decision.action.value}, "
              f"utility={decision.utility_score:.3f}, "
              f"reason={decision.reason[:60]}")

    result = run_with_raptor(
        agents=agents,
        prompt=prompt,
        config=config,
        on_decision=on_decision,
    )

    print()
    print(f"  Final answer:   {result.final_answer!r}")
    print(f"  Steps taken:    {result.steps_taken}")
    print(f"  Final action:   {result.final_action.value}")
    print(f"  Escalated:      {result.escalated}")
    print(f"  Stopped:        {result.stopped}")
    print(f"  Signal history: {len(result.context.signal_history)} entries")
    print(f"  Action history: {result.context.action_history}")
    print()

    # ── 5. Inspect signals ──────────────────────────────────────────
    if result.context.signal_history:
        print("=" * 60)
        print("Step 4: Signal history")
        print("=" * 60)
        for i, sig in enumerate(result.context.signal_history):
            print(f"  Step {i}:")
            for k, v in sig.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {k}: {val}")

    print()
    print("Done! RAPTOR processed the query and selected the best answer.")


if __name__ == "__main__":
    main()
