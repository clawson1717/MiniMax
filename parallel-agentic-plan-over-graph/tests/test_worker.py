"""Comprehensive tests for the Worker Agent (Executor) module."""

from __future__ import annotations

import time

import pytest

from src.models import TaskNode, TaskStatus
from src.worker import (
    MockTool,
    ReasoningWorker,
    WorkerProtocol,
    WorkerResult,
    code_tool,
    memory_tool,
    search_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "t1",
    description: str = "do something",
    **kwargs,
) -> TaskNode:
    """Shortcut to build a TaskNode for tests."""
    return TaskNode(id=node_id, description=description, **kwargs)


# ===================================================================
# WorkerProtocol
# ===================================================================


class TestWorkerProtocol:
    """Protocol compliance checks."""

    def test_reasoning_worker_satisfies_protocol(self):
        worker = ReasoningWorker("w1")
        assert isinstance(worker, WorkerProtocol)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(WorkerProtocol, "__protocol_attrs__") or True
        # runtime_checkable protocols support isinstance
        assert isinstance(ReasoningWorker("w1"), WorkerProtocol)

    def test_arbitrary_class_without_execute_fails_protocol(self):
        class NoExecute:
            pass

        assert not isinstance(NoExecute(), WorkerProtocol)

    def test_class_with_execute_satisfies_protocol(self):
        class CustomWorker:
            def execute(self, node):
                return None

        assert isinstance(CustomWorker(), WorkerProtocol)


# ===================================================================
# WorkerResult dataclass
# ===================================================================


class TestWorkerResult:
    """WorkerResult field defaults and construction."""

    def test_defaults(self):
        r = WorkerResult(node_id="n1", worker_id="w1", result="ok")
        assert r.tools_used == []
        assert r.duration_seconds == 0.0
        assert r.success is True
        assert r.error is None

    def test_all_fields(self):
        r = WorkerResult(
            node_id="n1",
            worker_id="w1",
            result={"key": "val"},
            tools_used=["search", "code"],
            duration_seconds=1.23,
            success=False,
            error="boom",
        )
        assert r.node_id == "n1"
        assert r.worker_id == "w1"
        assert r.result == {"key": "val"}
        assert r.tools_used == ["search", "code"]
        assert r.duration_seconds == 1.23
        assert r.success is False
        assert r.error == "boom"

    def test_result_can_hold_none(self):
        r = WorkerResult(node_id="n1", worker_id="w1", result=None)
        assert r.result is None

    def test_result_can_hold_list(self):
        r = WorkerResult(node_id="n1", worker_id="w1", result=[1, 2, 3])
        assert r.result == [1, 2, 3]


# ===================================================================
# MockTool
# ===================================================================


class TestMockTool:
    """MockTool callable behaviour."""

    def test_default_template(self):
        tool = MockTool("analyzer")
        out = tool("hello")
        assert out == "[analyzer] processed: hello"

    def test_custom_template(self):
        tool = MockTool("custom", response_template="CUSTOM:{input}")
        out = tool("data")
        assert out == "CUSTOM:data"

    def test_repr(self):
        tool = MockTool("xyz")
        assert "xyz" in repr(tool)

    def test_name_attribute(self):
        tool = MockTool("scanner")
        assert tool.name == "scanner"

    def test_deterministic(self):
        tool = MockTool("det")
        assert tool("a") == tool("a")


# ===================================================================
# Built-in mock tools
# ===================================================================


class TestBuiltinTools:
    """search_tool, code_tool, memory_tool."""

    def test_search_tool_output(self):
        out = search_tool("cats")
        assert "[search]" in out
        assert "cats" in out

    def test_code_tool_output(self):
        out = code_tool("def foo(): pass")
        assert "[code]" in out
        assert "def foo(): pass" in out

    def test_memory_tool_output(self):
        out = memory_tool("yesterday")
        assert "[memory]" in out
        assert "yesterday" in out


# ===================================================================
# ReasoningWorker — construction & registration
# ===================================================================


class TestWorkerConstruction:
    """Creation and tool registration."""

    def test_create_with_no_tools(self):
        w = ReasoningWorker("w1")
        assert w.worker_id == "w1"
        assert w.capabilities == []

    def test_create_with_initial_tools(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        assert "search" in w.capabilities

    def test_register_tool(self):
        w = ReasoningWorker("w1")
        w.register_tool("code", code_tool)
        assert "code" in w.capabilities

    def test_register_replaces_existing(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        new_fn = lambda x: "new"
        w.register_tool("search", new_fn)
        assert w._tools["search"] is new_fn

    def test_capabilities_sorted(self):
        w = ReasoningWorker("w1", tools={
            "z_tool": lambda x: x,
            "a_tool": lambda x: x,
            "m_tool": lambda x: x,
        })
        assert w.capabilities == ["a_tool", "m_tool", "z_tool"]

    def test_initial_tools_dict_is_copied(self):
        """Mutating the original dict shouldn't affect the worker."""
        original = {"search": search_tool}
        w = ReasoningWorker("w1", tools=original)
        original["code"] = code_tool
        assert "code" not in w.capabilities


# ===================================================================
# ReasoningWorker — execute with matching tools
# ===================================================================


class TestExecuteWithTools:
    """Execute when tools match the description."""

    def test_search_keyword_matches_search_tool(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        node = _make_node(description="search for relevant papers")
        result = w.execute(node)
        assert result.success is True
        assert "search" in result.tools_used
        assert "[search]" in result.result

    def test_code_keyword_matches_code_tool(self):
        w = ReasoningWorker("w1", tools={"code": code_tool})
        node = _make_node(description="implement a sorting function")
        result = w.execute(node)
        assert result.success is True
        assert "code" in result.tools_used

    def test_memory_keyword_matches_memory_tool(self):
        w = ReasoningWorker("w1", tools={"memory": memory_tool})
        node = _make_node(description="recall previous context from memory")
        result = w.execute(node)
        assert result.success is True
        assert "memory" in result.tools_used

    def test_find_keyword_triggers_search(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        node = _make_node(description="find all references")
        result = w.execute(node)
        assert "search" in result.tools_used

    def test_program_keyword_triggers_code(self):
        w = ReasoningWorker("w1", tools={"code": code_tool})
        node = _make_node(description="program a new module")
        result = w.execute(node)
        assert "code" in result.tools_used

    def test_case_insensitive_matching(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        node = _make_node(description="SEARCH the database")
        result = w.execute(node)
        assert "search" in result.tools_used

    def test_result_node_id_matches(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        node = _make_node(node_id="task-42", description="search for data")
        result = w.execute(node)
        assert result.node_id == "task-42"

    def test_result_worker_id_matches(self):
        w = ReasoningWorker("worker-7", tools={"search": search_tool})
        node = _make_node(description="search something")
        result = w.execute(node)
        assert result.worker_id == "worker-7"

    def test_duration_is_positive(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        node = _make_node(description="search")
        result = w.execute(node)
        assert result.duration_seconds >= 0.0


# ===================================================================
# ReasoningWorker — execute with multiple tools
# ===================================================================


class TestExecuteMultipleTools:
    """Execute when multiple tools match."""

    def test_multiple_tools_execute(self):
        w = ReasoningWorker("w1", tools={
            "search": search_tool,
            "memory": memory_tool,
        })
        node = _make_node(description="search and recall from memory")
        result = w.execute(node)
        assert result.success is True
        assert "search" in result.tools_used
        assert "memory" in result.tools_used
        assert "[search]" in result.result
        assert "[memory]" in result.result

    def test_multiple_tools_output_joined_by_newline(self):
        w = ReasoningWorker("w1", tools={
            "search": search_tool,
            "code": code_tool,
        })
        node = _make_node(description="search then write code")
        result = w.execute(node)
        lines = result.result.split("\n")
        assert len(lines) == 2

    def test_all_three_tools(self):
        w = ReasoningWorker("w1", tools={
            "search": search_tool,
            "code": code_tool,
            "memory": memory_tool,
        })
        node = _make_node(description="search, code implementation, recall memory")
        result = w.execute(node)
        assert len(result.tools_used) == 3


# ===================================================================
# ReasoningWorker — execute with no matching tools (fallback)
# ===================================================================


class TestExecuteFallback:
    """Execute with no tools or no keyword match → reasoning fallback."""

    def test_no_tools_registered(self):
        w = ReasoningWorker("w1")
        node = _make_node(description="analyze the situation")
        result = w.execute(node)
        assert result.success is True
        assert result.tools_used == []
        assert "[reasoning]" in result.result

    def test_tools_registered_but_no_match(self):
        w = ReasoningWorker("w1", tools={"search": search_tool})
        node = _make_node(description="analyze the situation carefully")
        result = w.execute(node)
        assert result.success is True
        assert result.tools_used == []
        assert "[reasoning]" in result.result

    def test_fallback_includes_description(self):
        w = ReasoningWorker("w1")
        desc = "evaluate trade-offs"
        node = _make_node(description=desc)
        result = w.execute(node)
        assert desc in result.result


# ===================================================================
# ReasoningWorker — error handling
# ===================================================================


class TestExecuteErrorHandling:
    """Tool exceptions are caught gracefully."""

    def test_tool_raises_exception(self):
        def bad_tool(text: str) -> str:
            raise RuntimeError("tool exploded")

        w = ReasoningWorker("w1", tools={"search": bad_tool})
        node = _make_node(description="search for data")
        result = w.execute(node)
        assert result.success is False
        assert result.error is not None
        assert "tool exploded" in result.error
        assert result.result is None

    def test_tool_raises_value_error(self):
        def val_err(text: str) -> str:
            raise ValueError("bad value")

        w = ReasoningWorker("w1", tools={"code": val_err})
        node = _make_node(description="write code")
        result = w.execute(node)
        assert result.success is False
        assert "bad value" in result.error

    def test_error_still_records_duration(self):
        def slow_fail(text: str) -> str:
            time.sleep(0.01)
            raise RuntimeError("slow boom")

        w = ReasoningWorker("w1", tools={"search": slow_fail})
        node = _make_node(description="search")
        result = w.execute(node)
        assert result.duration_seconds > 0.0

    def test_error_preserves_node_and_worker_ids(self):
        def bad(text: str) -> str:
            raise RuntimeError("oops")

        w = ReasoningWorker("worker-99", tools={"search": bad})
        node = _make_node(node_id="node-7", description="search")
        result = w.execute(node)
        assert result.node_id == "node-7"
        assert result.worker_id == "worker-99"

    def test_error_records_matched_tools(self):
        """Even on failure, tools_used should list what was attempted."""
        def bad(text: str) -> str:
            raise RuntimeError("fail")

        w = ReasoningWorker("w1", tools={"search": bad})
        node = _make_node(description="search for stuff")
        result = w.execute(node)
        assert "search" in result.tools_used


# ===================================================================
# ReasoningWorker — custom / dynamic tools
# ===================================================================


class TestCustomTools:
    """Dynamically registered and custom tools."""

    def test_custom_tool_keyword_match(self):
        """A tool named 'translate' matches when 'translate' is in description."""
        w = ReasoningWorker("w1")
        w.register_tool("translate", lambda t: f"translated: {t}")
        node = _make_node(description="translate this text")
        result = w.execute(node)
        assert "translate" in result.tools_used
        assert "translated:" in result.result

    def test_mock_tool_as_registered_tool(self):
        mt = MockTool("analyzer")
        w = ReasoningWorker("w1")
        w.register_tool("analyzer", mt)
        node = _make_node(description="run analyzer on data")
        result = w.execute(node)
        assert "analyzer" in result.tools_used
        assert "[analyzer]" in result.result

    def test_multiple_custom_tools(self):
        w = ReasoningWorker("w1")
        w.register_tool("alpha", lambda t: "a")
        w.register_tool("beta", lambda t: "b")
        node = _make_node(description="use alpha and beta together")
        result = w.execute(node)
        assert "alpha" in result.tools_used
        assert "beta" in result.tools_used
