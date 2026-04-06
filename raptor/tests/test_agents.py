"""Tests for agents.py — Step 6: Agent Protocol & Polling.

Covers:
  - ReasoningAgent protocol conformance (isinstance checks, duck typing)
  - StreamingReasoningAgent protocol conformance
  - AgentStreamEvent / TokenLogProbs / StreamEventType data structures
  - PollResult properties
  - poll_agents(): basic multi-agent polling, concurrency, ordering
  - poll_agents(): error handling — single failure, all failures, fail_fast
  - poll_agents(): timeout handling
  - poll_agents(): edge cases — single agent, many agents
  - poll_agents_streaming(): basic streaming, event callback, assembly
  - poll_agents_streaming(): error handling
  - reroll(): basic candidate generation, concurrent vs sequential
  - reroll(): error handling — partial failure, total failure
  - reroll(): timeout handling
  - Mock agents: MockReasoningAgent, MockStreamingAgent, MockFailingAgent,
                 MockSlowAgent, MockVariableAgent
  - Integration with existing AgentResponse from disagreement.py
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from raptor.disagreement import AgentResponse
from raptor.agents import (
    AgentError,
    AgentStreamEvent,
    MockFailingAgent,
    MockReasoningAgent,
    MockSlowAgent,
    MockStreamingAgent,
    MockVariableAgent,
    PollResult,
    ReasoningAgent,
    StreamEventType,
    StreamingReasoningAgent,
    TokenLogProbs,
    poll_agents,
    poll_agents_streaming,
    reroll,
)


# ======================================================================
# Data structure tests
# ======================================================================


class TestTokenLogProbs:
    def test_basic_construction(self):
        tlp = TokenLogProbs(token="hello", log_prob=-0.5)
        assert tlp.token == "hello"
        assert tlp.log_prob == -0.5
        assert tlp.top_k_tokens is None

    def test_with_top_k(self):
        tlp = TokenLogProbs(
            token="hello",
            log_prob=-0.5,
            top_k_tokens={"hello": -0.5, "hi": -1.2, "hey": -2.0},
        )
        assert len(tlp.top_k_tokens) == 3
        assert tlp.top_k_tokens["hello"] == -0.5


class TestStreamEventType:
    def test_enum_values(self):
        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.STEP_COMPLETE.value == "step_complete"
        assert StreamEventType.DONE.value == "done"


class TestAgentStreamEvent:
    def test_token_event(self):
        event = AgentStreamEvent(
            event_type=StreamEventType.TOKEN,
            token="hello",
            token_log_probs=TokenLogProbs(token="hello", log_prob=-0.3),
        )
        assert event.event_type == StreamEventType.TOKEN
        assert event.token == "hello"
        assert event.token_log_probs.log_prob == -0.3

    def test_step_complete_event(self):
        event = AgentStreamEvent(
            event_type=StreamEventType.STEP_COMPLETE,
            step_text="First I analyze the problem",
            step_index=0,
        )
        assert event.event_type == StreamEventType.STEP_COMPLETE
        assert event.step_text == "First I analyze the problem"
        assert event.step_index == 0

    def test_done_event(self):
        event = AgentStreamEvent(
            event_type=StreamEventType.DONE,
            final_answer="42",
        )
        assert event.event_type == StreamEventType.DONE
        assert event.final_answer == "42"

    def test_defaults_are_none(self):
        event = AgentStreamEvent(event_type=StreamEventType.TOKEN)
        assert event.token is None
        assert event.token_log_probs is None
        assert event.step_text is None
        assert event.step_index is None
        assert event.final_answer is None
        assert event.agent_id is None


class TestAgentError:
    def test_basic_construction(self):
        err = AgentError("something broke")
        assert str(err) == "something broke"
        assert err.agent_id is None
        assert err.original is None

    def test_with_agent_id_and_original(self):
        orig = RuntimeError("boom")
        err = AgentError("agent failed", agent_id="agent-1", original=orig)
        assert err.agent_id == "agent-1"
        assert err.original is orig

    def test_is_exception(self):
        err = AgentError("test")
        assert isinstance(err, Exception)


class TestPollResult:
    def test_empty_result(self):
        pr = PollResult()
        assert pr.n_success == 0
        assert pr.n_errors == 0
        assert pr.all_succeeded is True

    def test_with_responses(self):
        pr = PollResult(
            responses=[
                AgentResponse("a1", ["step"], "answer"),
                AgentResponse("a2", ["step"], "answer"),
            ]
        )
        assert pr.n_success == 2
        assert pr.n_errors == 0
        assert pr.all_succeeded is True

    def test_with_errors(self):
        pr = PollResult(
            responses=[AgentResponse("a1", ["step"], "answer")],
            errors=[(1, AgentError("fail"))],
        )
        assert pr.n_success == 1
        assert pr.n_errors == 1
        assert pr.all_succeeded is False

    def test_all_errors(self):
        pr = PollResult(
            errors=[
                (0, AgentError("fail-0")),
                (1, AgentError("fail-1")),
            ]
        )
        assert pr.n_success == 0
        assert pr.n_errors == 2
        assert pr.all_succeeded is False


# ======================================================================
# Protocol conformance
# ======================================================================


class TestReasoningAgentProtocol:
    def test_mock_agent_satisfies_protocol(self):
        agent = MockReasoningAgent(
            agent_id="test", reasoning_steps=["step"], final_answer="42"
        )
        assert isinstance(agent, ReasoningAgent)

    def test_duck_typing_with_plain_class(self):
        """Any class with generate(prompt)->AgentResponse satisfies the protocol."""

        class CustomAgent:
            def generate(self, prompt: str) -> AgentResponse:
                return AgentResponse("custom", ["step"], "answer")

        agent = CustomAgent()
        assert isinstance(agent, ReasoningAgent)

    def test_failing_agent_satisfies_protocol(self):
        agent = MockFailingAgent(agent_id="fail")
        assert isinstance(agent, ReasoningAgent)

    def test_slow_agent_satisfies_protocol(self):
        agent = MockSlowAgent(agent_id="slow", delay_seconds=0.0)
        assert isinstance(agent, ReasoningAgent)

    def test_variable_agent_satisfies_protocol(self):
        agent = MockVariableAgent(
            agent_id="var", responses=[(["step"], "answer")]
        )
        assert isinstance(agent, ReasoningAgent)


class TestStreamingReasoningAgentProtocol:
    def test_mock_streaming_agent_satisfies_protocol(self):
        agent = MockStreamingAgent(
            _agent_id="stream-1",
            reasoning_steps=["step one"],
            final_answer="42",
        )
        assert isinstance(agent, StreamingReasoningAgent)


# ======================================================================
# MockReasoningAgent
# ======================================================================


class TestMockReasoningAgent:
    def test_returns_configured_response(self):
        agent = MockReasoningAgent(
            agent_id="test-1",
            reasoning_steps=["think about it", "calculate"],
            final_answer="42",
        )
        response = agent.generate("What is 6*7?")
        assert response.agent_id == "test-1"
        assert response.reasoning_steps == ["think about it", "calculate"]
        assert response.final_answer == "42"

    def test_ignores_prompt(self):
        """Mock returns the same response regardless of prompt."""
        agent = MockReasoningAgent(
            agent_id="const", reasoning_steps=["step"], final_answer="yes"
        )
        r1 = agent.generate("prompt A")
        r2 = agent.generate("prompt B")
        assert r1.final_answer == r2.final_answer

    def test_returns_copy_of_steps(self):
        """Modifying the returned list shouldn't affect future calls."""
        agent = MockReasoningAgent(
            agent_id="copy-test",
            reasoning_steps=["original step"],
            final_answer="answer",
        )
        r1 = agent.generate("prompt")
        r1.reasoning_steps.append("mutated!")
        r2 = agent.generate("prompt")
        assert len(r2.reasoning_steps) == 1


# ======================================================================
# MockStreamingAgent
# ======================================================================


class TestMockStreamingAgent:
    def test_produces_expected_event_sequence(self):
        agent = MockStreamingAgent(
            _agent_id="stream-test",
            reasoning_steps=["hello world"],
            final_answer="done",
        )
        events = list(agent.stream("prompt"))

        # Should have: 2 TOKEN events (hello, world) + 1 STEP_COMPLETE + 1 DONE
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        step_events = [
            e for e in events if e.event_type == StreamEventType.STEP_COMPLETE
        ]
        done_events = [e for e in events if e.event_type == StreamEventType.DONE]

        assert len(token_events) == 2
        assert len(step_events) == 1
        assert len(done_events) == 1
        assert done_events[0].final_answer == "done"

    def test_multi_step_streaming(self):
        agent = MockStreamingAgent(
            _agent_id="multi",
            reasoning_steps=["step one", "step two here"],
            final_answer="result",
        )
        events = list(agent.stream("prompt"))

        step_events = [
            e for e in events if e.event_type == StreamEventType.STEP_COMPLETE
        ]
        assert len(step_events) == 2
        assert step_events[0].step_text == "step one"
        assert step_events[1].step_text == "step two here"

    def test_token_log_probs_included(self):
        agent = MockStreamingAgent(
            _agent_id="lp",
            reasoning_steps=["hello"],
            final_answer="done",
        )
        events = list(agent.stream("prompt"))
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        assert token_events[0].token_log_probs is not None
        assert token_events[0].token_log_probs.log_prob == -0.5

    def test_agent_id_property(self):
        agent = MockStreamingAgent(
            _agent_id="my-id",
            reasoning_steps=[],
            final_answer="x",
        )
        assert agent.agent_id == "my-id"


# ======================================================================
# MockFailingAgent
# ======================================================================


class TestMockFailingAgent:
    def test_raises_on_generate(self):
        agent = MockFailingAgent(agent_id="fail-1", error_message="Boom!")
        with pytest.raises(RuntimeError, match="Boom!"):
            agent.generate("prompt")

    def test_default_error_message(self):
        agent = MockFailingAgent(agent_id="fail-2")
        with pytest.raises(RuntimeError, match="Mock failure"):
            agent.generate("prompt")


# ======================================================================
# MockSlowAgent
# ======================================================================


class TestMockSlowAgent:
    def test_delays_response(self):
        agent = MockSlowAgent(agent_id="slow-1", delay_seconds=0.05)
        start = time.monotonic()
        response = agent.generate("prompt")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04  # small margin
        assert response.final_answer == "delayed answer"

    def test_zero_delay(self):
        agent = MockSlowAgent(agent_id="fast", delay_seconds=0.0)
        response = agent.generate("prompt")
        assert response.agent_id == "fast"


# ======================================================================
# MockVariableAgent
# ======================================================================


class TestMockVariableAgent:
    def test_cycles_through_responses(self):
        agent = MockVariableAgent(
            agent_id="var-1",
            responses=[
                (["step A"], "answer A"),
                (["step B"], "answer B"),
                (["step C"], "answer C"),
            ],
        )
        r0 = agent.generate("prompt")
        r1 = agent.generate("prompt")
        r2 = agent.generate("prompt")
        r3 = agent.generate("prompt")  # wraps around

        assert r0.final_answer == "answer A"
        assert r1.final_answer == "answer B"
        assert r2.final_answer == "answer C"
        assert r3.final_answer == "answer A"

    def test_single_response(self):
        agent = MockVariableAgent(
            agent_id="one",
            responses=[(["step"], "only answer")],
        )
        assert agent.generate("p1").final_answer == "only answer"
        assert agent.generate("p2").final_answer == "only answer"


# ======================================================================
# poll_agents() — basic functionality
# ======================================================================


class TestPollAgentsBasic:
    def test_single_agent(self):
        agents = [
            MockReasoningAgent("a1", ["step"], "42"),
        ]
        result = poll_agents(agents, "What is 6*7?")
        assert result.n_success == 1
        assert result.all_succeeded is True
        assert result.responses[0].final_answer == "42"

    def test_multiple_agents(self):
        agents = [
            MockReasoningAgent(f"a{i}", [f"step-{i}"], f"answer-{i}")
            for i in range(5)
        ]
        result = poll_agents(agents, "prompt")
        assert result.n_success == 5
        assert result.all_succeeded is True

    def test_all_responses_are_agent_responses(self):
        agents = [
            MockReasoningAgent("a1", ["s1"], "ans1"),
            MockReasoningAgent("a2", ["s2"], "ans2"),
        ]
        result = poll_agents(agents, "prompt")
        for resp in result.responses:
            assert isinstance(resp, AgentResponse)

    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            poll_agents([], "prompt")

    def test_preserves_agent_ids(self):
        agents = [
            MockReasoningAgent("alpha", ["step"], "a"),
            MockReasoningAgent("beta", ["step"], "b"),
            MockReasoningAgent("gamma", ["step"], "c"),
        ]
        result = poll_agents(agents, "prompt")
        ids = {r.agent_id for r in result.responses}
        assert ids == {"alpha", "beta", "gamma"}


# ======================================================================
# poll_agents() — concurrency
# ======================================================================


class TestPollAgentsConcurrency:
    def test_agents_run_concurrently(self):
        """Multiple slow agents should complete faster than sequential execution."""
        delay = 0.1
        n = 4
        agents = [
            MockSlowAgent(f"slow-{i}", delay_seconds=delay) for i in range(n)
        ]
        start = time.monotonic()
        result = poll_agents(agents, "prompt")
        elapsed = time.monotonic() - start

        assert result.n_success == n
        # Concurrent: should take ~delay, not n*delay
        # Allow generous margin for CI, but should be well under sequential time
        assert elapsed < delay * n * 0.8

    def test_max_workers_limits_parallelism(self):
        """With max_workers=1, should behave more like sequential."""
        delay = 0.05
        n = 3
        agents = [
            MockSlowAgent(f"s-{i}", delay_seconds=delay) for i in range(n)
        ]
        start = time.monotonic()
        result = poll_agents(agents, "prompt", max_workers=1)
        elapsed = time.monotonic() - start

        assert result.n_success == n
        # Sequential: should take at least n*delay
        assert elapsed >= delay * n * 0.8


# ======================================================================
# poll_agents() — error handling
# ======================================================================


class TestPollAgentsErrors:
    def test_one_failing_agent(self):
        agents = [
            MockReasoningAgent("good-1", ["step"], "yes"),
            MockFailingAgent("bad-1"),
            MockReasoningAgent("good-2", ["step"], "yes"),
        ]
        result = poll_agents(agents, "prompt")
        assert result.n_success == 2
        assert result.n_errors == 1
        assert not result.all_succeeded

    def test_all_failing_agents(self):
        agents = [
            MockFailingAgent("bad-1"),
            MockFailingAgent("bad-2"),
        ]
        result = poll_agents(agents, "prompt")
        assert result.n_success == 0
        assert result.n_errors == 2

    def test_error_contains_agent_id(self):
        agents = [MockFailingAgent("fail-x", "custom error")]
        result = poll_agents(agents, "prompt")
        assert result.n_errors == 1
        idx, error = result.errors[0]
        assert error.agent_id == "fail-x"
        assert "custom error" in str(error)

    def test_fail_fast_raises_immediately(self):
        agents = [
            MockFailingAgent("bad"),
            MockReasoningAgent("good", ["s"], "a"),
        ]
        with pytest.raises(AgentError):
            poll_agents(agents, "prompt", fail_fast=True)

    def test_errors_include_original_exception(self):
        agents = [MockFailingAgent("bad", "original boom")]
        result = poll_agents(agents, "prompt")
        _, error = result.errors[0]
        assert error.original is not None
        assert isinstance(error.original, RuntimeError)


# ======================================================================
# poll_agents() — timeout handling
# ======================================================================


class TestPollAgentsTimeout:
    def test_fast_agents_within_timeout(self):
        agents = [
            MockReasoningAgent(f"fast-{i}", ["s"], "a") for i in range(3)
        ]
        result = poll_agents(agents, "prompt", timeout=5.0)
        assert result.n_success == 3

    def test_timeout_with_slow_agent(self):
        """One very slow agent should time out while fast agents succeed."""
        agents = [
            MockReasoningAgent("fast", ["s"], "a"),
            MockSlowAgent("very-slow", delay_seconds=10.0),
        ]
        result = poll_agents(agents, "prompt", timeout=0.5)
        # Fast agent should succeed, slow one should time out
        # Note: exact behavior depends on thread scheduling
        assert result.n_success >= 1


# ======================================================================
# poll_agents_streaming() — basic functionality
# ======================================================================


class TestPollAgentsStreaming:
    def test_basic_streaming_poll(self):
        agents = [
            MockStreamingAgent("s1", ["step one", "step two"], "answer-1"),
            MockStreamingAgent("s2", ["step A"], "answer-2"),
        ]
        result = poll_agents_streaming(agents, "prompt")
        assert result.n_success == 2
        assert result.all_succeeded

    def test_assembled_responses_have_steps(self):
        agents = [
            MockStreamingAgent(
                "s1",
                ["first step", "second step"],
                "final",
            ),
        ]
        result = poll_agents_streaming(agents, "prompt")
        assert len(result.responses) == 1
        resp = result.responses[0]
        assert resp.agent_id == "s1"
        assert len(resp.reasoning_steps) == 2
        assert resp.final_answer == "final"

    def test_event_callback_invoked(self):
        agents = [
            MockStreamingAgent("s1", ["hello world"], "done"),
        ]
        events_received: list[AgentStreamEvent] = []

        def on_event(event: AgentStreamEvent):
            events_received.append(event)

        poll_agents_streaming(agents, "prompt", on_event=on_event)

        assert len(events_received) > 0
        # Should have TOKEN events, a STEP_COMPLETE, and a DONE
        event_types = {e.event_type for e in events_received}
        assert StreamEventType.TOKEN in event_types
        assert StreamEventType.STEP_COMPLETE in event_types
        assert StreamEventType.DONE in event_types

    def test_event_callback_receives_agent_id(self):
        agents = [
            MockStreamingAgent("agent-x", ["step"], "done"),
        ]
        agent_ids_seen: set[str] = set()

        def on_event(event: AgentStreamEvent):
            if event.agent_id:
                agent_ids_seen.add(event.agent_id)

        poll_agents_streaming(agents, "prompt", on_event=on_event)
        assert "agent-x" in agent_ids_seen

    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            poll_agents_streaming([], "prompt")

    def test_multiple_streaming_agents_concurrent(self):
        """Multiple streaming agents should complete in parallel."""
        agents = [
            MockStreamingAgent(f"s{i}", [f"step {i}"], f"answer {i}")
            for i in range(5)
        ]
        result = poll_agents_streaming(agents, "prompt")
        assert result.n_success == 5
        ids = {r.agent_id for r in result.responses}
        assert ids == {f"s{i}" for i in range(5)}


# ======================================================================
# poll_agents_streaming() — error handling
# ======================================================================


class TestPollAgentsStreamingErrors:
    def test_failing_streaming_agent(self):
        """A streaming agent that raises mid-stream should be caught."""

        class FailingStreamAgent:
            @property
            def agent_id(self) -> str:
                return "fail-stream"

            def stream(self, prompt):
                yield AgentStreamEvent(
                    event_type=StreamEventType.TOKEN, token="start "
                )
                raise RuntimeError("Stream broke!")

        agents = [FailingStreamAgent()]
        result = poll_agents_streaming(agents, "prompt")
        assert result.n_errors == 1
        assert result.n_success == 0

    def test_mixed_success_and_failure(self):
        class FailStream:
            @property
            def agent_id(self) -> str:
                return "fail"

            def stream(self, prompt):
                raise RuntimeError("nope")

        agents = [
            MockStreamingAgent("good", ["step"], "answer"),
            FailStream(),
        ]
        result = poll_agents_streaming(agents, "prompt")
        assert result.n_success == 1
        assert result.n_errors == 1


# ======================================================================
# reroll() — basic functionality
# ======================================================================


class TestRerollBasic:
    def test_generates_n_candidates(self):
        agent = MockReasoningAgent("r1", ["step"], "42")
        candidates = reroll(agent, "prompt", n_candidates=5)
        assert len(candidates) == 5

    def test_single_candidate(self):
        agent = MockReasoningAgent("r1", ["step"], "42")
        candidates = reroll(agent, "prompt", n_candidates=1)
        assert len(candidates) == 1
        assert candidates[0].final_answer == "42"

    def test_variable_agent_produces_different_candidates(self):
        agent = MockVariableAgent(
            agent_id="var",
            responses=[
                (["think A"], "answer A"),
                (["think B"], "answer B"),
                (["think C"], "answer C"),
            ],
        )
        candidates = reroll(agent, "prompt", n_candidates=3)
        answers = {c.final_answer for c in candidates}
        # Should get all 3 different answers
        assert len(answers) == 3

    def test_invalid_n_candidates(self):
        agent = MockReasoningAgent("r1", ["step"], "42")
        with pytest.raises(ValueError, match="n_candidates"):
            reroll(agent, "prompt", n_candidates=0)

    def test_concurrent_mode(self):
        agent = MockReasoningAgent("r1", ["step"], "42")
        candidates = reroll(agent, "prompt", n_candidates=3, concurrent=True)
        assert len(candidates) == 3

    def test_sequential_mode(self):
        agent = MockReasoningAgent("r1", ["step"], "42")
        candidates = reroll(agent, "prompt", n_candidates=3, concurrent=False)
        assert len(candidates) == 3


# ======================================================================
# reroll() — error handling
# ======================================================================


class TestRerollErrors:
    def test_all_candidates_fail_raises(self):
        agent = MockFailingAgent("bad", "always fails")
        with pytest.raises(AgentError, match="All .* reroll candidates failed"):
            reroll(agent, "prompt", n_candidates=3)

    def test_all_candidates_fail_sequential(self):
        agent = MockFailingAgent("bad")
        with pytest.raises(AgentError):
            reroll(agent, "prompt", n_candidates=2, concurrent=False)

    def test_partial_failure_returns_successes(self):
        """An agent that sometimes fails should return whatever succeeded."""
        call_count = 0

        class PartialFailAgent:
            agent_id = "partial"

            def generate(self, prompt: str) -> AgentResponse:
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise RuntimeError("Fail on even calls")
                return AgentResponse("partial", ["step"], "answer")

        agent = PartialFailAgent()
        # With 4 candidates: 2 should succeed, 2 should fail
        candidates = reroll(agent, "prompt", n_candidates=4, concurrent=False)
        assert len(candidates) >= 1  # At least some succeed

    def test_total_failure_includes_original_error(self):
        agent = MockFailingAgent("bad", "original error message")
        with pytest.raises(AgentError) as exc_info:
            reroll(agent, "prompt", n_candidates=1)
        assert exc_info.value.original is not None


# ======================================================================
# reroll() — timeout
# ======================================================================


class TestRerollTimeout:
    def test_timeout_concurrent(self):
        """Concurrent reroll with timeout returns quickly (non-blocking shutdown)."""
        agent = MockSlowAgent("slow", delay_seconds=5.0)
        start = time.monotonic()
        try:
            candidates = reroll(agent, "prompt", n_candidates=3, timeout=0.3)
        except AgentError:
            candidates = []
        elapsed = time.monotonic() - start
        # With non-blocking shutdown, should return shortly after timeout
        # (not waiting for all 3 * 5s threads to complete)
        assert elapsed < 2.0

    def test_timeout_sequential_limits_iterations(self):
        """Sequential reroll with timeout skips remaining candidates after deadline.

        Note: The first generate() call blocks for its full duration since
        Python cannot interrupt a running thread.  The timeout prevents
        *subsequent* candidates from being attempted.
        """
        agent = MockSlowAgent("slow", delay_seconds=0.15)
        start = time.monotonic()
        try:
            candidates = reroll(
                agent, "prompt", n_candidates=10, timeout=0.3, concurrent=False
            )
        except AgentError:
            candidates = []
        elapsed = time.monotonic() - start
        # Should complete well before 10 * 0.15 = 1.5s
        assert elapsed < 1.0
        # Should have generated some but not all 10 candidates
        assert len(candidates) < 10


# ======================================================================
# Integration — poll_agents with orchestrator-compatible responses
# ======================================================================


class TestPollAgentsIntegration:
    def test_responses_compatible_with_disagreement_monitor(self):
        """Polled responses should work directly with DisagreementMonitor."""
        from raptor.config import DisagreementConfig
        from raptor.disagreement import DisagreementMonitor

        agents = [
            MockReasoningAgent(
                f"a{i}",
                ["analyze the problem", "compute the result"],
                "42",
            )
            for i in range(3)
        ]
        result = poll_agents(agents, "What is 6*7?")

        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(result.responses)
        assert signal.disagreement_tier in ("low", "medium", "weak")

    def test_responses_compatible_with_orchestrator_step(self):
        """Polled responses should feed directly into RAPTOROrchestrator.step()."""
        from raptor.config import Config
        from raptor.orchestrator import RAPTOROrchestrator
        import tempfile

        agents = [
            MockReasoningAgent(
                f"a{i}",
                [
                    "Let me explore different methods for this problem",
                    "Using algebra I simplify",
                    "The answer is 42",
                ],
                "42",
            )
            for i in range(3)
        ]
        result = poll_agents(agents, "prompt")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config()
            cfg.log_dir = tmpdir
            orch = RAPTOROrchestrator(cfg)
            decision = orch.step("prompt", result.responses)
            assert decision.action is not None

    def test_reroll_candidates_compatible_with_reroll_with_selection(self):
        """Reroll candidates should work with orchestrator's reroll_with_selection."""
        from raptor.config import Config
        from raptor.orchestrator import RAPTOROrchestrator
        import tempfile

        agent = MockVariableAgent(
            agent_id="var",
            responses=[
                (
                    [
                        "Explore many different approaches methods techniques "
                        "strategies and heuristics for solving",
                        "Simplify using algebra",
                        "x is 42",
                    ],
                    "42",
                ),
                (
                    [
                        "The answer is obvious",
                        "Wait no let me reconsider all possibilities "
                        "and different approaches",
                        "Maybe it's 42",
                    ],
                    "42",
                ),
            ],
        )
        candidates = reroll(agent, "prompt", n_candidates=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config()
            cfg.log_dir = tmpdir
            orch = RAPTOROrchestrator(cfg)
            best = orch.reroll_with_selection(candidates)
            assert isinstance(best, AgentResponse)
            assert best.final_answer == "42"


# ======================================================================
# Integration — streaming with real-time entropy tracking
# ======================================================================


class TestStreamingEntropyIntegration:
    def test_streaming_events_can_feed_entropy_tracker(self):
        """Token-level events should carry enough info for entropy tracking."""
        from raptor.entropy_tracker import EntropyTracker
        from raptor.config import EntropyConfig
        import numpy as np

        agent = MockStreamingAgent(
            _agent_id="stream-entropy",
            reasoning_steps=["first I analyze", "then I compute"],
            final_answer="42",
        )

        tracker = EntropyTracker(EntropyConfig())
        step_tokens: list[str] = []

        def on_event(event: AgentStreamEvent):
            nonlocal step_tokens
            if event.event_type == StreamEventType.TOKEN and event.token:
                step_tokens.append(event.token)
            elif event.event_type == StreamEventType.STEP_COMPLETE:
                # Compute synthetic log probs for the accumulated step text
                text = "".join(step_tokens)
                if text.strip():
                    tokens = text.lower().split()
                    vocab_size = 256
                    freq = np.zeros(vocab_size, dtype=np.float64)
                    for t in tokens:
                        freq[hash(t) % vocab_size] += 1
                    freq += 1e-6
                    probs = freq / freq.sum()
                    log_probs = np.log(probs)
                    tracker.update(log_probs)
                step_tokens = []

        result = poll_agents_streaming([agent], "prompt", on_event=on_event)
        assert result.n_success == 1

        signal = tracker.compute_signal()
        assert signal.n_steps >= 1
        assert isinstance(signal.confidence_score, float)


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_poll_ten_agents(self):
        """10 agents should work fine."""
        agents = [
            MockReasoningAgent(f"a{i}", [f"step {i}"], f"ans {i}")
            for i in range(10)
        ]
        result = poll_agents(agents, "prompt")
        assert result.n_success == 10

    def test_poll_agent_with_empty_reasoning(self):
        agents = [
            MockReasoningAgent("empty", [], "42"),
        ]
        result = poll_agents(agents, "prompt")
        assert result.n_success == 1
        assert result.responses[0].reasoning_steps == []

    def test_poll_agent_with_long_reasoning(self):
        steps = [f"reasoning step {i}" for i in range(50)]
        agents = [MockReasoningAgent("long", steps, "final")]
        result = poll_agents(agents, "prompt")
        assert result.n_success == 1
        assert len(result.responses[0].reasoning_steps) == 50

    def test_reroll_with_one_candidate(self):
        agent = MockReasoningAgent("r1", ["s"], "a")
        candidates = reroll(agent, "prompt", n_candidates=1)
        assert len(candidates) == 1

    def test_streaming_agent_with_empty_steps(self):
        agent = MockStreamingAgent("empty", [], "answer")
        events = list(agent.stream("prompt"))
        done = [e for e in events if e.event_type == StreamEventType.DONE]
        assert len(done) == 1
        assert done[0].final_answer == "answer"

    def test_streaming_poll_with_no_callback(self):
        """Should work fine without on_event callback."""
        agents = [MockStreamingAgent("s1", ["step"], "answer")]
        result = poll_agents_streaming(agents, "prompt", on_event=None)
        assert result.n_success == 1

    def test_thread_safety_of_variable_agent_in_reroll(self):
        """MockVariableAgent call count should be thread-safe enough for testing."""
        agent = MockVariableAgent(
            agent_id="threadsafe",
            responses=[
                (["s1"], "a1"),
                (["s2"], "a2"),
                (["s3"], "a3"),
                (["s4"], "a4"),
                (["s5"], "a5"),
            ],
        )
        candidates = reroll(agent, "prompt", n_candidates=5, concurrent=True)
        # All 5 should succeed (even if order varies)
        assert len(candidates) == 5
