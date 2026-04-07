"""Tool Integration Layer for PAPoG.

Provides a standardized Tool interface and registry for managing tools
that can be used by ReasoningWorker and other execution components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Protocol defining the interface for all tools.

    Any tool must implement these methods and properties to be compatible
    with the ToolRegistry and ReasoningWorker.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    def keywords(self) -> list[str]:
        """Keywords used to match this tool to task descriptions."""
        ...

    def execute(self, input: str) -> str:
        """Execute the tool with the given input and return result.

        Parameters
        ----------
        input:
            The input string for the tool (query, code, etc.)

        Returns
        -------
        str
            The result of executing the tool.
        """
        ...

    def can_handle(self, task_description: str) -> bool:
        """Check if this tool can handle the given task.

        Default implementation checks if any keyword appears in the
        task description (case-insensitive).

        Parameters
        ----------
        task_description:
            The description of the task to check.

        Returns
        -------
        bool
            True if this tool can handle the task.
        """
        ...


class ToolABC(ABC):
    """Abstract base class for tools.

    Provides default implementation of can_handle using keyword matching.
    Subclasses must implement name, description, keywords, and execute.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    @abstractmethod
    def keywords(self) -> list[str]:
        """Keywords used to match this tool to task descriptions."""
        ...

    @abstractmethod
    def execute(self, input: str) -> str:
        """Execute the tool with the given input and return result."""
        ...

    def can_handle(self, task_description: str) -> bool:
        """Check if this tool can handle the given task.

        Default implementation checks if any keyword appears in the
        task description (case-insensitive).

        Parameters
        ----------
        task_description:
            The description of the task to check.

        Returns
        -------
        bool
            True if this tool can handle the task.
        """
        task_lower = task_description.lower()
        return any(kw.lower() in task_lower for kw in self.keywords)


# Import tools for convenience
from src.tools.search_tool import SearchTool
from src.tools.python_repl_tool import PythonREPLTool
from src.tools.memory_tool import MemoryTool
from src.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolABC",
    "SearchTool",
    "PythonREPLTool",
    "MemoryTool",
    "ToolRegistry",
]
