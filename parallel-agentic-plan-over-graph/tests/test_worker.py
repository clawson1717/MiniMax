"""Comprehensive tests for PAPoG Worker Agent."""

import pytest

from src.models import TaskNode, TaskStatus
from src.worker import (
    ExecutionStrategy,
    MockExecutionStrategy,
    MockToolProvider,
    ReasoningWorker,
    ToolProvider,
    WorkerResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "t1",
    description: str = "do something",
    status: TaskStatus = TaskStatus.PENDING,
) -> TaskNode:
    node = TaskNode(id=task_id, description=description)
    node.status = status
    return node


# ---------------------------------------------------------------------------
# WorkerResult
# ---------------------------------------------------------------------------


class TestWorkerResult:
    def test_defaults(self):
        r = WorkerResult(task_id="t1", success=True)
        assert r.task_id == "t1"
        assert r.success is True
        assert r.result_data is None
        assert r.execution_time == 0.0
        assert r.error is None

    def test_with_all_fields(self):
        r = WorkerResult(
            task_id="t2",
            success=False,
            result_data={"key": "val"},
            execution_time=1.23,
            error="boom",
        )
        assert r.task_id == "t2"
        assert r.success is False
        assert r.result_data == {"key": "val"}
        assert r.execution_time == 1.23
        assert r.error == "boom"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_mock_execution_strategy_satisfies_protocol(self):
        assert isinstance(MockExecutionStrategy(), ExecutionStrategy)

    def test_mock_tool_provider_satisfies_protocol(self):
        assert isinstance(MockToolProvider(), ToolProvider)

    def test_custom_strategy_satisfies_protocol(self):
        class Custom:
            def execute(self, task, tools=None):
                return "custom"

        assert isinstance(Custom(), ExecutionStrategy)

    def test_custom_tool_provider_satisfies_protocol(self):
        class Custom:
            def get_tools(self):
                return {}

            def invoke(self, tool_name, **kwargs):
                return None

        assert isinstance(Custom(), ToolProvider)


# ---------------------------------------------------------------------------
# MockExecutionStrategy
# ---------------------------------------------------------------------------


class TestMockExecutionStrategy:
    def test_success(self):
        strategy = MockExecutionStrategy()
        task = _make_task(description="build a widget")
        result = strategy.execute(task)
        assert result == "Executed: build a widget"

    def test_failure_keyword(self):
        strategy = MockExecutionStrategy()
        task = _make_task(description="this will fail badly")
        with pytest.raises(RuntimeError, match="Simulated failure"):
            strategy.execute(task)

    def test_tools_included_in_result(self):
        strategy = MockExecutionStrategy()
        task = _make_task(description="use tools")
        result = strategy.execute(task, tools={"alpha": None, "beta": None})
        assert "tools: alpha, beta" in result

    def test_no_tools(self):
        strategy = MockExecutionStrategy()
        task = _make_task(description="no tools")
        result = strategy.execute(task)
        assert "tools:" not in result


# ---------------------------------------------------------------------------
# MockToolProvider
# ---------------------------------------------------------------------------


class TestMockToolProvider:
    def test_default_tools(self):
        provider = MockToolProvider()
        tools = provider.get_tools()
        assert "search" in tools
        assert "calculate" in tools

    def test_invoke_search(self):
        provider = MockToolProvider()
        result = provider.invoke("search", query="test")
        assert result == "results for 'test'"

    def test_invoke_calculate(self):
        provider = MockToolProvider()
        result = provider.invoke("calculate", expression="2+2")
        assert result == "computed '2+2'"

    def test_invoke_unknown_tool(self):
        provider = MockToolProvider()
        with pytest.raises(KeyError, match="Unknown tool"):
            provider.invoke("nonexistent")

    def test_custom_tools(self):
        provider = MockToolProvider(tools={"greet": lambda name="": f"hi {name}"})
        assert "greet" in provider.get_tools()
        assert provider.invoke("greet", name="Alice") == "hi Alice"

    def test_get_tools_returns_copy(self):
        provider = MockToolProvider()
        tools = provider.get_tools()
        tools["extra"] = lambda: None
        assert "extra" not in provider.get_tools()


# ---------------------------------------------------------------------------
# ReasoningWorker — construction
# ---------------------------------------------------------------------------


class TestWorkerConstruction:
    def test_defaults(self):
        worker = ReasoningWorker()
        assert worker.agent_id == "worker-0"
        assert isinstance(worker.strategy, MockExecutionStrategy)
        assert worker.tool_provider is None

    def test_custom_agent_id(self):
        worker = ReasoningWorker(agent_id="agent-42")
        assert worker.agent_id == "agent-42"

    def test_custom_strategy(self):
        class Custom:
            def execute(self, task, tools=None):
                return "custom"

        s = Custom()
        worker = ReasoningWorker(strategy=s)
        assert worker.strategy is s

    def test_strategy_setter(self):
        worker = ReasoningWorker()
        new_strategy = MockExecutionStrategy()
        worker.strategy = new_strategy
        assert worker.strategy is new_strategy

    def test_tool_provider_setter(self):
        worker = ReasoningWorker()
        provider = MockToolProvider()
        worker.tool_provider = provider
        assert worker.tool_provider is provider


# ---------------------------------------------------------------------------
# ReasoningWorker — execution
# ---------------------------------------------------------------------------


class TestWorkerExecution:
    def test_successful_execution(self):
        worker = ReasoningWorker(agent_id="w1")
        task = _make_task(description="build something")
        result = worker.execute(task)

        assert result.task_id == "t1"
        assert result.success is True
        assert "Executed: build something" in result.result_data
        assert result.error is None
        assert result.execution_time >= 0
        assert task.status == TaskStatus.COMPLETED
        assert task.assigned_agent == "w1"

    def test_failed_execution(self):
        worker = ReasoningWorker(agent_id="w2")
        task = _make_task(description="this will fail")
        result = worker.execute(task)

        assert result.task_id == "t1"
        assert result.success is False
        assert result.error is not None
        assert "Simulated failure" in result.error
        assert result.result_data is None
        assert task.status == TaskStatus.FAILED
        assert task.assigned_agent == "w2"

    def test_execution_time_is_positive(self):
        worker = ReasoningWorker()
        task = _make_task()
        result = worker.execute(task)
        assert result.execution_time >= 0

    def test_with_tool_provider(self):
        provider = MockToolProvider()
        worker = ReasoningWorker(tool_provider=provider)
        task = _make_task(description="use tools")
        result = worker.execute(task)

        assert result.success is True
        assert "tools:" in result.result_data
        assert "calculate" in result.result_data
        assert "search" in result.result_data

    def test_without_tool_provider(self):
        worker = ReasoningWorker()
        task = _make_task(description="no tools here")
        result = worker.execute(task)

        assert result.success is True
        assert "tools:" not in result.result_data


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_pending_to_completed(self):
        worker = ReasoningWorker()
        task = _make_task()
        assert task.status == TaskStatus.PENDING
        result = worker.execute(task)
        assert task.status == TaskStatus.COMPLETED
        assert result.success is True

    def test_pending_to_failed(self):
        worker = ReasoningWorker()
        task = _make_task(description="fail now")
        assert task.status == TaskStatus.PENDING
        result = worker.execute(task)
        assert task.status == TaskStatus.FAILED
        assert result.success is False

    def test_already_running_rejected(self):
        worker = ReasoningWorker()
        task = _make_task(status=TaskStatus.RUNNING)
        with pytest.raises(ValueError, match="expected pending"):
            worker.execute(task)

    def test_already_completed_rejected(self):
        worker = ReasoningWorker()
        task = _make_task(status=TaskStatus.COMPLETED)
        with pytest.raises(ValueError, match="expected pending"):
            worker.execute(task)

    def test_already_failed_rejected(self):
        worker = ReasoningWorker()
        task = _make_task(status=TaskStatus.FAILED)
        with pytest.raises(ValueError, match="expected pending"):
            worker.execute(task)

    def test_agent_id_set_on_task(self):
        worker = ReasoningWorker(agent_id="special-agent")
        task = _make_task()
        worker.execute(task)
        assert task.assigned_agent == "special-agent"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_description_still_executes(self):
        worker = ReasoningWorker()
        task = _make_task(description="")
        result = worker.execute(task)
        assert result.success is True
        assert result.result_data == "Executed: "

    def test_strategy_raises_unexpected_exception(self):
        class BrokenStrategy:
            def execute(self, task, tools=None):
                raise TypeError("unexpected type error")

        worker = ReasoningWorker(strategy=BrokenStrategy())
        task = _make_task()
        result = worker.execute(task)
        assert result.success is False
        assert "unexpected type error" in result.error
        assert task.status == TaskStatus.FAILED

    def test_multiple_workers_different_tasks(self):
        w1 = ReasoningWorker(agent_id="w1")
        w2 = ReasoningWorker(agent_id="w2")

        t1 = _make_task(task_id="a", description="task a")
        t2 = _make_task(task_id="b", description="task b")

        r1 = w1.execute(t1)
        r2 = w2.execute(t2)

        assert r1.success is True
        assert r2.success is True
        assert t1.assigned_agent == "w1"
        assert t2.assigned_agent == "w2"

    def test_result_stored_on_task_node(self):
        worker = ReasoningWorker()
        task = _make_task(description="store this")
        worker.execute(task)
        assert task.result == "Executed: store this"

    def test_error_stored_on_task_node(self):
        worker = ReasoningWorker()
        task = _make_task(description="fail this")
        worker.execute(task)
        assert "Simulated failure" in task.result
