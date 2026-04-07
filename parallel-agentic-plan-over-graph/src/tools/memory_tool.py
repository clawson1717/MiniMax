"""MemoryTool - Key-value store for cross-node information sharing in PAPoG.

Provides a simple in-memory key-value store that can be used to share
information between different task nodes during execution.
"""

from __future__ import annotations

import json
from typing import Any

from src.tools import ToolABC


class MemoryTool(ToolABC):
    """Key-value memory store for cross-node information sharing.

    This tool provides a simple in-memory key-value store that persists
    across task executions. Useful for passing intermediate results
    between tasks that don't have direct dependencies.

    Parameters
    ----------
    name:
        Tool identifier. Defaults to "memory".
    description:
        Human-readable description. Defaults to a memory store description.
    keywords:
        Keywords for task matching. Defaults to ["memory", "store", "save", "retrieve", "remember", "recall"].
    """

    # Class-level storage to share memory across all instances
    _global_store: dict[str, Any] = {}

    def __init__(
        self,
        name: str = "memory",
        description: str = "Store and retrieve values for cross-task communication",
        keywords: list[str] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._keywords = keywords if keywords is not None else [
            "memory",
            "store",
            "save",
            "retrieve",
            "remember",
            "recall",
            "cache",
        ]
        # Instance-level store (can be used for isolated memory)
        self._local_store: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Unique identifier for this tool."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        return self._description

    @property
    def keywords(self) -> list[str]:
        """Keywords used to match this tool to task descriptions."""
        return list(self._keywords)

    @classmethod
    def get_global_store(cls) -> dict[str, Any]:
        """Get the global memory store.

        Returns
        -------
        dict[str, Any]
            The global store dictionary.
        """
        return cls._global_store

    @classmethod
    def clear_global_store(cls) -> None:
        """Clear all entries from the global memory store."""
        cls._global_store.clear()

    def get_local_store(self) -> dict[str, Any]:
        """Get the instance-level local memory store.

        Returns
        -------
        dict[str, Any]
            The local store dictionary.
        """
        return self._local_store

    def clear_local_store(self) -> None:
        """Clear all entries from the local memory store."""
        self._local_store.clear()

    def store(self, key: str, value: Any, use_global: bool = True) -> str:
        """Store a value under the given key.

        Parameters
        ----------
        key:
            The key to store the value under.
        value:
            The value to store.
        use_global:
            If True, store in global memory (shared across all instances).
            If False, store in local instance memory.

        Returns
        -------
        str
            Confirmation message.
        """
        if not key or not key.strip():
            return "Error: Empty key. Please provide a valid key name."

        target_store = MemoryTool._global_store if use_global else self._local_store
        target_store[key.strip()] = value

        scope = "global" if use_global else "local"
        return f"Stored '{key}' in {scope} memory."

    def retrieve(self, key: str, use_global: bool = True) -> str:
        """Retrieve a value by key.

        Parameters
        ----------
        key:
            The key to look up.
        use_global:
            If True, search global memory first, then local.
            If False, search only local memory.

        Returns
        -------
        str
            The stored value (as string) or error message.
        """
        if not key or not key.strip():
            return "Error: Empty key. Please provide a valid key name."

        key = key.strip()

        if use_global:
            # Check global first, then local
            if key in MemoryTool._global_store:
                value = MemoryTool._global_store[key]
            elif key in self._local_store:
                value = self._local_store[key]
            else:
                return f"Error: Key '{key}' not found in memory."
        else:
            # Only check local
            if key in self._local_store:
                value = self._local_store[key]
            else:
                return f"Error: Key '{key}' not found in local memory."

        # Convert value to string representation
        if isinstance(value, str):
            return value
        elif isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        else:
            return str(value)

    def list_keys(self, use_global: bool = True) -> str:
        """List all stored keys.

        Parameters
        ----------
        use_global:
            If True, list global keys. If False, list local keys.

        Returns
        -------
        str
            Formatted list of keys.
        """
        target_store = MemoryTool._global_store if use_global else self._local_store

        if not target_store:
            scope = "global" if use_global else "local"
            return f"No keys stored in {scope} memory."

        keys = list(target_store.keys())
        scope = "global" if use_global else "local"
        return f"Keys in {scope} memory:\n" + "\n".join(f"  - {k}" for k in sorted(keys))

    def delete(self, key: str, use_global: bool = True) -> str:
        """Delete a value by key.

        Parameters
        ----------
        key:
            The key to delete.
        use_global:
            If True, delete from global memory. If False, delete from local.

        Returns
        -------
        str
            Confirmation or error message.
        """
        if not key or not key.strip():
            return "Error: Empty key. Please provide a valid key name."

        key = key.strip()
        target_store = MemoryTool._global_store if use_global else self._local_store

        if key in target_store:
            del target_store[key]
            scope = "global" if use_global else "local"
            return f"Deleted '{key}' from {scope} memory."
        else:
            scope = "global" if use_global else "local"
            return f"Error: Key '{key}' not found in {scope} memory."

    def execute(self, input: str) -> str:
        """Execute a memory operation.

        Input format:
        - "store:key=value" - Store value under key
        - "retrieve:key" - Retrieve value by key
        - "list" - List all keys
        - "delete:key" - Delete value by key

        Parameters
        ----------
        input:
            Operation command string.

        Returns
        -------
        str
            Result of the operation.
        """
        if not input or not input.strip():
            return "Error: Empty input. Use format: store:key=value, retrieve:key, list, or delete:key"

        input = input.strip()

        # Parse command
        if input.startswith("store:"):
            # Format: store:key=value
            remainder = input[6:]
            if "=" not in remainder:
                return "Error: Invalid store format. Use: store:key=value"

            key, value = remainder.split("=", 1)
            return self.store(key.strip(), value.strip())

        elif input.startswith("retrieve:"):
            # Format: retrieve:key
            key = input[9:]
            return self.retrieve(key.strip())

        elif input.startswith("delete:"):
            # Format: delete:key
            key = input[7:]
            return self.delete(key.strip())

        elif input == "list":
            return self.list_keys()

        else:
            return f"Error: Unknown command. Use: store:key=value, retrieve:key, list, or delete:key"

    def can_handle(self, task_description: str) -> bool:
        """Check if this tool can handle the given task.

        Parameters
        ----------
        task_description:
            The description of the task to check.

        Returns
        -------
        bool
            True if any keyword appears in the task description.
        """
        task_lower = task_description.lower()
        return any(kw.lower() in task_lower for kw in self._keywords)
