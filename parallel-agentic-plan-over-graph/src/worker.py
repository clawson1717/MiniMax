"""Worker Agent (Executor) for PAPoG.

Provides the ReasoningWorker — a tool-aware executor that processes
TaskNodes by matching task descriptions to registered tools, executing
them in sequence, and returning structured WorkerResult objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from src.models import TaskNode


# ---------------------------------------------------------------------------
# Worker protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WorkerProtocol(Protocol):
    """Interface that all worker implementations must satisfy."""

    def execute(self, node: TaskNode) -> Any:
        """Execute the work described by *node* and return a result."""
        ...


# ---------------------------------------------------------------------------
# WorkerResult
# ---------------------------------------------------------------------------


@dataclass
class WorkerResult:
    """Structured output from a worker execution."""

    node_id: str
    worker_id: str
    result: Any
    tools_used: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# MockTool
# ---------------------------------------------------------------------------


class MockTool:
    """A deterministic callable for testing tool execution.

    Parameters
    ----------
    name:
        Human-readable tool name (included in output).
    response_template:
        Format string with a single ``{input}`` placeholder.
    """

    def __init__(self, name: str, response_template: str = "[{name}] processed: {input}") -> None:
        self.name = name
        self.response_template = response_template

    def __call__(self, input_text: str) -> str:
        return self.response_template.format(name=self.name, input=input_text)

    def __repr__(self) -> str:
        return f"MockTool(name={self.name!r})"


# ---------------------------------------------------------------------------
# Built-in mock tools
# ---------------------------------------------------------------------------


def search_tool(input_text: str) -> str:
    """Mock search tool returning formatted search results."""
    return f"[search] results for: {input_text}"


def code_tool(input_text: str) -> str:
    """Mock code tool returning formatted code output."""
    return f"[code] executed: {input_text}"


def memory_tool(input_text: str) -> str:
    """Mock memory tool returning formatted memory retrieval."""
    return f"[memory] recalled: {input_text}"


# ---------------------------------------------------------------------------
# ReasoningWorker
# ---------------------------------------------------------------------------

# Default keyword → tool-name mappings used for automatic tool selection
_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "search": ["search", "find", "lookup", "query", "look up"],
    "code": ["code", "implement", "program", "write code", "function", "script"],
    "memory": ["memory", "recall", "remember", "retrieve", "history"],
}


class ReasoningWorker:
    """Tool-aware worker that executes TaskNodes.

    The worker maintains a registry of named tools (callables).  When
    ``execute`` is called it inspects the node description, selects
    matching tools via keyword matching, runs them in sequence, and
    returns a :class:`WorkerResult`.

    Parameters
    ----------
    worker_id:
        Unique identifier for this worker.
    tools:
        Optional initial mapping of ``{name: callable}``.
    """

    def __init__(
        self,
        worker_id: str,
        tools: dict[str, Callable] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._tools: dict[str, Callable] = dict(tools) if tools else {}

    # -- tool management ----------------------------------------------------

    def register_tool(self, name: str, fn: Callable) -> None:
        """Add or replace a tool in the registry."""
        self._tools[name] = fn

    @property
    def capabilities(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    # -- execution ----------------------------------------------------------

    def _match_tools(self, description: str) -> list[str]:
        """Return tool names whose keywords appear in *description*.

        Matching is case-insensitive.  The returned list preserves the
        sorted order of tool names for determinism.
        """
        desc_lower = description.lower()
        matched: list[str] = []
        for tool_name in sorted(self._tools.keys()):
            keywords = _DEFAULT_KEYWORDS.get(tool_name, [tool_name])
            if any(kw in desc_lower for kw in keywords):
                matched.append(tool_name)
        return matched

    def execute(self, node: TaskNode) -> WorkerResult:
        """Execute the task described by *node*.

        Steps:
        1. Extract description from the node.
        2. Match registered tools by keyword.
        3. Run matching tools in sequence, collecting outputs.
        4. If no tools match, fall back to a default reasoning step.
        5. Return a ``WorkerResult`` with timing and status info.

        Errors raised by individual tools are caught; the result will
        have ``success=False`` and the error message in ``error``.
        """
        start = time.monotonic()
        description = node.description
        matched = self._match_tools(description)

        try:
            if matched:
                outputs: list[str] = []
                for tool_name in matched:
                    tool_fn = self._tools[tool_name]
                    output = tool_fn(description)
                    outputs.append(str(output))
                combined = "\n".join(outputs)
            else:
                combined = f"[reasoning] analyzed: {description}"
                matched = []

            duration = time.monotonic() - start
            return WorkerResult(
                node_id=node.id,
                worker_id=self.worker_id,
                result=combined,
                tools_used=list(matched),
                duration_seconds=duration,
                success=True,
                error=None,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return WorkerResult(
                node_id=node.id,
                worker_id=self.worker_id,
                result=None,
                tools_used=list(matched),
                duration_seconds=duration,
                success=False,
                error=str(exc),
            )
