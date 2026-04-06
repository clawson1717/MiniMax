"""Worker Agent (Executor) for PAPoG.

Executes individual TaskNodes via a pluggable ExecutionStrategy.
Ships with ``MockExecutionStrategy`` (deterministic, test-friendly)
and a ``ToolProvider`` protocol for future tool integration.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.models import TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# WorkerResult
# ---------------------------------------------------------------------------


@dataclass
class WorkerResult:
    """Outcome of executing a single TaskNode."""

    task_id: str
    success: bool
    result_data: Any = None
    execution_time: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# ToolProvider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for providing tools to execution strategies."""

    def get_tools(self) -> dict[str, Any]:
        """Return a mapping of tool names to callables."""
        ...

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a named tool with the given arguments."""
        ...


# ---------------------------------------------------------------------------
# Execution strategy interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionStrategy(Protocol):
    """Protocol that any execution backend must satisfy."""

    def execute(self, task: TaskNode, tools: dict[str, Any] | None = None) -> Any:
        """Execute the task and return the result data."""
        ...


class ExecutionStrategyABC(ABC):
    """Abstract base class alternative for execution strategies."""

    @abstractmethod
    def execute(self, task: TaskNode, tools: dict[str, Any] | None = None) -> Any:
        """Execute the task and return the result data."""


# ---------------------------------------------------------------------------
# Built-in strategies & tool providers
# ---------------------------------------------------------------------------


class MockToolProvider:
    """Deterministic mock tool provider for testing."""

    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        self._tools: dict[str, Any] = tools or {
            "search": lambda query="": f"results for '{query}'",
            "calculate": lambda expression="": f"computed '{expression}'",
        }

    def get_tools(self) -> dict[str, Any]:
        return dict(self._tools)

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name](**kwargs)


class MockExecutionStrategy(ExecutionStrategyABC):
    """Deterministic mock that simulates task execution.

    Returns a predictable result string based on the task description.
    If the description contains "fail", raises ``RuntimeError``.
    """

    def execute(self, task: TaskNode, tools: dict[str, Any] | None = None) -> Any:
        if "fail" in task.description.lower():
            raise RuntimeError(f"Simulated failure for task '{task.id}'")

        result = f"Executed: {task.description}"
        if tools:
            result += f" [tools: {', '.join(sorted(tools.keys()))}]"
        return result


# ---------------------------------------------------------------------------
# ReasoningWorker
# ---------------------------------------------------------------------------


class ReasoningWorker:
    """Executes TaskNodes using a pluggable ExecutionStrategy.

    Parameters
    ----------
    agent_id:
        Unique identifier for this worker agent.
    strategy:
        An ``ExecutionStrategy`` for task execution.  Defaults to
        ``MockExecutionStrategy``.
    tool_provider:
        Optional ``ToolProvider`` whose tools are passed to the strategy.
    """

    def __init__(
        self,
        agent_id: str = "worker-0",
        strategy: ExecutionStrategy | None = None,
        tool_provider: ToolProvider | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._strategy: ExecutionStrategy = strategy or MockExecutionStrategy()
        self._tool_provider = tool_provider

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def strategy(self) -> ExecutionStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: ExecutionStrategy) -> None:
        self._strategy = value

    @property
    def tool_provider(self) -> ToolProvider | None:
        return self._tool_provider

    @tool_provider.setter
    def tool_provider(self, value: ToolProvider | None) -> None:
        self._tool_provider = value

    def execute(self, task: TaskNode) -> WorkerResult:
        """Execute a single TaskNode and return a ``WorkerResult``.

        Updates the task's status through its lifecycle:
        PENDING -> RUNNING -> COMPLETED (or FAILED).

        Parameters
        ----------
        task:
            The ``TaskNode`` to execute.

        Returns
        -------
        WorkerResult
            Contains task_id, success flag, result data or error,
            and execution wall-clock time.

        Raises
        ------
        ValueError
            If the task is not in PENDING status.
        """
        if task.status != TaskStatus.PENDING:
            raise ValueError(
                f"Cannot execute task '{task.id}': "
                f"status is {task.status.value}, expected pending"
            )

        task.mark_running(self._agent_id)

        tools: dict[str, Any] | None = None
        if self._tool_provider is not None:
            tools = self._tool_provider.get_tools()

        start = time.monotonic()
        try:
            result_data = self._strategy.execute(task, tools=tools)
            elapsed = time.monotonic() - start
            task.mark_completed(result_data)
            return WorkerResult(
                task_id=task.id,
                success=True,
                result_data=result_data,
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            error_msg = str(exc)
            task.mark_failed(error_msg)
            return WorkerResult(
                task_id=task.id,
                success=False,
                error=error_msg,
                execution_time=elapsed,
            )
