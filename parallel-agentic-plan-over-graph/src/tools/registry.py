"""ToolRegistry - Central tool management for PAPoG.

Provides registration, lookup, and management of tools for use by
ReasoningWorker and other execution components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tools import Tool


class ToolRegistry:
    """Central registry for managing tools.

    Provides methods to register, unregister, and lookup tools.
    Integrates with ReasoningWorker via get_tools() method that returns
    a dict compatible with the ToolProvider protocol.

    Example
    -------
    >>> registry = ToolRegistry()
    >>> registry.register(SearchTool())
    >>> registry.register(PythonREPLTool())
    >>> tools = registry.get_tools()
    >>> tool = registry.find_by_keyword("search")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Parameters
        ----------
        tool:
            The tool instance to register.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Parameters
        ----------
        name:
            The name of the tool to remove.

        Returns
        -------
        bool
            True if the tool was removed, False if it wasn't found.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Parameters
        ----------
        name:
            The name of the tool to retrieve.

        Returns
        -------
        Tool | None
            The tool instance, or None if not found.
        """
        return self._tools.get(name)

    def find_by_keyword(self, keyword: str) -> Tool | None:
        """Find a tool that matches the given keyword.

        Parameters
        ----------
        keyword:
            The keyword to search for.

        Returns
        -------
        Tool | None
            The first tool whose keywords contain the given keyword,
            or None if no match found.
        """
        keyword_lower = keyword.lower()
        for tool in self._tools.values():
            if any(keyword_lower == kw.lower() for kw in tool.keywords):
                return tool
        return None

    def find_for_task(self, task_description: str) -> list[Tool]:
        """Find all tools that can handle the given task.

        Parameters
        ----------
        task_description:
            The task description to match against.

        Returns
        -------
        list[Tool]
            List of tools that can handle the task.
        """
        return [
            tool
            for tool in self._tools.values()
            if tool.can_handle(task_description)
        ]

    def get_all(self) -> list[Tool]:
        """Get all registered tools.

        Returns
        -------
        list[Tool]
            List of all registered tool instances.
        """
        return list(self._tools.values())

    def get_all_names(self) -> list[str]:
        """Get names of all registered tools.

        Returns
        -------
        list[str]
            List of tool names.
        """
        return list(self._tools.keys())

    def get_tools(self) -> dict[str, Tool]:
        """Get all tools as a dictionary.

        This method provides compatibility with the ToolProvider protocol
        used by ReasoningWorker.

        Returns
        -------
        dict[str, Tool]
            Dictionary mapping tool names to tool instances.
        """
        return dict(self._tools)

    def invoke(self, name: str, input: str) -> str:
        """Invoke a registered tool by name.

        Parameters
        ----------
        name:
            The name of the tool to invoke.
        input:
            The input to pass to the tool's execute method.

        Returns
        -------
        str
            The result of executing the tool.

        Raises
        ------
        KeyError
            If no tool with the given name is registered.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool.execute(input)

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered by name."""
        return name in self._tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        return f"ToolRegistry({self.get_all_names()})"


class ToolRegistryProvider:
    """Adapter to make ToolRegistry work with ToolProvider protocol.

    This class wraps a ToolRegistry and implements the ToolProvider
    protocol from worker.py, allowing seamless integration with
    ReasoningWorker.

    Parameters
    ----------
    registry:
        The ToolRegistry instance to adapt.

    Example
    -------
    >>> registry = ToolRegistry()
    >>> registry.register(SearchTool())
    >>> provider = ToolRegistryProvider(registry)
    >>> worker = ReasoningWorker(tool_provider=provider)
    """

    def __init__(self, registry: ToolRegistry) -> None:
        """Initialize with a registry."""
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """Get the underlying registry."""
        return self._registry

    def get_tools(self) -> dict[str, Any]:
        """Return a mapping of tool names to callable execute functions.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping tool names to execute functions.
        """
        return {
            name: tool.execute
            for name, tool in self._registry.get_tools().items()
        }

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a named tool with the given arguments.

        Parameters
        ----------
        tool_name:
            The name of the tool to invoke.
        **kwargs:
            Keyword arguments to pass to the tool. Uses 'input' key for execute.

        Returns
        -------
        Any
            The result of the tool execution.

        Raises
        ------
        KeyError
            If the tool is not found.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown tool: {tool_name}")

        # Extract input from kwargs, default to empty string
        input_str = kwargs.get("input", kwargs.get("query", ""))
        return tool.execute(input_str)
