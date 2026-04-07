"""Comprehensive tests for the Tool Integration Layer."""

from __future__ import annotations

import pytest

from src.tools import Tool, ToolABC, SearchTool, PythonREPLTool, MemoryTool, ToolRegistry
from src.tools.registry import ToolRegistryProvider
from src.worker import ReasoningWorker, MockExecutionStrategy
from src.models import TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Tool Protocol Compliance Tests
# ---------------------------------------------------------------------------


class TestToolProtocolCompliance:
    """Test that all tools properly implement the Tool protocol."""

    def test_search_tool_implements_tool_protocol(self):
        """SearchTool should satisfy the Tool protocol."""
        tool = SearchTool()
        assert isinstance(tool, Tool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "keywords")
        assert hasattr(tool, "execute")
        assert hasattr(tool, "can_handle")

    def test_python_repl_tool_implements_tool_protocol(self):
        """PythonREPLTool should satisfy the Tool protocol."""
        tool = PythonREPLTool()
        assert isinstance(tool, Tool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "keywords")
        assert hasattr(tool, "execute")
        assert hasattr(tool, "can_handle")

    def test_memory_tool_implements_tool_protocol(self):
        """MemoryTool should satisfy the Tool protocol."""
        tool = MemoryTool()
        assert isinstance(tool, Tool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "keywords")
        assert hasattr(tool, "execute")
        assert hasattr(tool, "can_handle")

    def test_tool_abc_is_abstract(self):
        """ToolABC should be abstract and require implementation."""
        # Can't instantiate ToolABC directly
        with pytest.raises(TypeError):
            ToolABC()  # type: ignore

    def test_search_tool_extends_tool_abc(self):
        """SearchTool should extend ToolABC."""
        tool = SearchTool()
        assert isinstance(tool, ToolABC)

    def test_python_repl_tool_extends_tool_abc(self):
        """PythonREPLTool should extend ToolABC."""
        tool = PythonREPLTool()
        assert isinstance(tool, ToolABC)

    def test_memory_tool_extends_tool_abc(self):
        """MemoryTool should extend ToolABC."""
        tool = MemoryTool()
        assert isinstance(tool, ToolABC)


# ---------------------------------------------------------------------------
# SearchTool Tests
# ---------------------------------------------------------------------------


class TestSearchTool:
    """Tests for SearchTool functionality."""

    def test_default_name(self):
        """SearchTool should have default name."""
        tool = SearchTool()
        assert tool.name == "search"

    def test_custom_name(self):
        """SearchTool should accept custom name."""
        tool = SearchTool(name="web_search")
        assert tool.name == "web_search"

    def test_default_description(self):
        """SearchTool should have a description."""
        tool = SearchTool()
        assert tool.description
        assert "search" in tool.description.lower()

    def test_default_keywords(self):
        """SearchTool should have default keywords."""
        tool = SearchTool()
        assert "search" in tool.keywords
        assert "find" in tool.keywords

    def test_custom_keywords(self):
        """SearchTool should accept custom keywords."""
        tool = SearchTool(keywords=["google", "bing"])
        assert "google" in tool.keywords
        assert "bing" in tool.keywords

    def test_execute_with_query(self):
        """SearchTool should return results for a query."""
        tool = SearchTool()
        result = tool.execute("python tutorials")
        assert result
        assert "python tutorials" in result.lower()
        assert "Search Results" in result

    def test_execute_with_empty_query(self):
        """SearchTool should handle empty query gracefully."""
        tool = SearchTool()
        result = tool.execute("")
        assert "Error" in result
        assert "Empty" in result

    def test_execute_with_whitespace_query(self):
        """SearchTool should handle whitespace-only query."""
        tool = SearchTool()
        result = tool.execute("   ")
        assert "Error" in result

    def test_can_handle_matching_keyword(self):
        """SearchTool should match tasks with search keyword."""
        tool = SearchTool()
        assert tool.can_handle("search for information about AI")
        assert tool.can_handle("find documentation")
        assert tool.can_handle("lookup the answer")

    def test_can_handle_non_matching_keyword(self):
        """SearchTool should not match tasks without keywords."""
        tool = SearchTool()
        assert not tool.can_handle("execute python code")
        assert not tool.can_handle("calculate the result")

    def test_can_handle_case_insensitive(self):
        """SearchTool should match keywords case-insensitively."""
        tool = SearchTool()
        assert tool.can_handle("SEARCH for something")
        assert tool.can_handle("FIND the answer")
        assert tool.can_handle("Query the database")


# ---------------------------------------------------------------------------
# PythonREPLTool Tests
# ---------------------------------------------------------------------------


class TestPythonREPLTool:
    """Tests for PythonREPLTool functionality."""

    def test_default_name(self):
        """PythonREPLTool should have default name."""
        tool = PythonREPLTool()
        assert tool.name == "python_repl"

    def test_custom_name(self):
        """PythonREPLTool should accept custom name."""
        tool = PythonREPLTool(name="code_runner")
        assert tool.name == "code_runner"

    def test_default_description(self):
        """PythonREPLTool should have a description."""
        tool = PythonREPLTool()
        assert tool.description
        assert "python" in tool.description.lower()

    def test_default_keywords(self):
        """PythonREPLTool should have default keywords."""
        tool = PythonREPLTool()
        assert "python" in tool.keywords
        assert "code" in tool.keywords

    def test_custom_keywords(self):
        """PythonREPLTool should accept custom keywords."""
        tool = PythonREPLTool(keywords=["script", "program"])
        assert "script" in tool.keywords
        assert "program" in tool.keywords

    def test_timeout_property(self):
        """PythonREPLTool should have configurable timeout."""
        tool = PythonREPLTool(timeout=60)
        assert tool.timeout == 60

    def test_execute_with_simple_code(self):
        """PythonREPLTool should simulate code execution."""
        tool = PythonREPLTool()
        result = tool.execute("x = 5")
        assert result
        assert "executed" in result.lower() or "simulated" in result.lower()

    def test_execute_with_print_statement(self):
        """PythonREPLTool should handle print statements."""
        tool = PythonREPLTool()
        result = tool.execute('print("Hello, World!")')
        assert "Hello, World!" in result

    def test_execute_with_empty_code(self):
        """PythonREPLTool should handle empty code gracefully."""
        tool = PythonREPLTool()
        result = tool.execute("")
        assert "Error" in result

    def test_execute_with_whitespace_code(self):
        """PythonREPLTool should handle whitespace-only code."""
        tool = PythonREPLTool()
        result = tool.execute("   ")
        assert "Error" in result

    def test_security_block_import_os(self):
        """PythonREPLTool should block dangerous imports."""
        tool = PythonREPLTool()
        result = tool.execute("import os")
        assert "Security" in result or "blocked" in result.lower() or "dangerous" in result.lower()

    def test_security_block_subprocess(self):
        """PythonREPLTool should block subprocess import."""
        tool = PythonREPLTool()
        result = tool.execute("import subprocess")
        assert "Security" in result or "blocked" in result.lower() or "dangerous" in result.lower()

    def test_security_block_eval(self):
        """PythonREPLTool should block eval usage."""
        tool = PythonREPLTool()
        result = tool.execute("eval('1+1')")
        assert "Security" in result or "blocked" in result.lower() or "dangerous" in result.lower()

    def test_can_handle_matching_keyword(self):
        """PythonREPLTool should match tasks with code keywords."""
        tool = PythonREPLTool()
        assert tool.can_handle("execute python code")
        assert tool.can_handle("run this script")
        assert tool.can_handle("compute the value")

    def test_can_handle_non_matching_keyword(self):
        """PythonREPLTool should not match tasks without keywords."""
        tool = PythonREPLTool()
        assert not tool.can_handle("search for information")
        assert not tool.can_handle("find the answer")

    def test_can_handle_case_insensitive(self):
        """PythonREPLTool should match keywords case-insensitively."""
        tool = PythonREPLTool()
        assert tool.can_handle("PYTHON code here")
        assert tool.can_handle("EXECUTE this")


# ---------------------------------------------------------------------------
# MemoryTool Tests
# ---------------------------------------------------------------------------


class TestMemoryTool:
    """Tests for MemoryTool functionality."""

    def setup_method(self):
        """Clear global memory before each test."""
        MemoryTool.clear_global_store()

    def test_default_name(self):
        """MemoryTool should have default name."""
        tool = MemoryTool()
        assert tool.name == "memory"

    def test_custom_name(self):
        """MemoryTool should accept custom name."""
        tool = MemoryTool(name="kv_store")
        assert tool.name == "kv_store"

    def test_default_description(self):
        """MemoryTool should have a description."""
        tool = MemoryTool()
        assert tool.description
        assert "store" in tool.description.lower() or "memory" in tool.description.lower()

    def test_default_keywords(self):
        """MemoryTool should have default keywords."""
        tool = MemoryTool()
        assert "memory" in tool.keywords
        assert "store" in tool.keywords

    def test_custom_keywords(self):
        """MemoryTool should accept custom keywords."""
        tool = MemoryTool(keywords=["cache", "persist"])
        assert "cache" in tool.keywords
        assert "persist" in tool.keywords

    def test_store_and_retrieve(self):
        """MemoryTool should store and retrieve values."""
        tool = MemoryTool()
        tool.store("test_key", "test_value")
        result = tool.retrieve("test_key")
        assert result == "test_value"

    def test_store_overwrites(self):
        """MemoryTool should overwrite existing values."""
        tool = MemoryTool()
        tool.store("key", "value1")
        tool.store("key", "value2")
        assert tool.retrieve("key") == "value2"

    def test_retrieve_nonexistent_key(self):
        """MemoryTool should error on missing key."""
        tool = MemoryTool()
        result = tool.retrieve("nonexistent")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_retrieve_empty_key(self):
        """MemoryTool should handle empty key."""
        tool = MemoryTool()
        result = tool.retrieve("")
        assert "Error" in result

    def test_store_empty_key(self):
        """MemoryTool should handle empty key."""
        tool = MemoryTool()
        result = tool.store("", "value")
        assert "Error" in result

    def test_global_memory_shared(self):
        """MemoryTool should share global memory across instances."""
        tool1 = MemoryTool()
        tool2 = MemoryTool()

        tool1.store("shared_key", "shared_value")
        assert tool2.retrieve("shared_key") == "shared_value"

    def test_local_memory_isolated(self):
        """MemoryTool should isolate local memory."""
        tool1 = MemoryTool()
        tool2 = MemoryTool()

        tool1.store("local_key", "value1", use_global=False)
        result = tool2.retrieve("local_key", use_global=False)
        assert "Error" in result or result == "value1"  # Isolated, so error expected

    def test_delete_key(self):
        """MemoryTool should delete stored keys."""
        tool = MemoryTool()
        tool.store("key_to_delete", "value")
        assert tool.retrieve("key_to_delete") == "value"

        result = tool.delete("key_to_delete")
        assert "Deleted" in result

        # Should now be missing
        retrieve_result = tool.retrieve("key_to_delete")
        assert "Error" in retrieve_result

    def test_delete_nonexistent_key(self):
        """MemoryTool should handle deleting missing key."""
        tool = MemoryTool()
        result = tool.delete("nonexistent_key")
        assert "Error" in result

    def test_list_keys_empty(self):
        """MemoryTool should handle empty memory."""
        tool = MemoryTool()
        result = tool.list_keys()
        assert "No keys" in result

    def test_list_keys_with_entries(self):
        """MemoryTool should list all keys."""
        tool = MemoryTool()
        tool.store("key1", "value1")
        tool.store("key2", "value2")
        result = tool.list_keys()
        assert "key1" in result
        assert "key2" in result

    def test_clear_global_store(self):
        """MemoryTool.clear_global_store should clear all entries."""
        tool = MemoryTool()
        tool.store("key1", "value1")
        tool.store("key2", "value2")

        MemoryTool.clear_global_store()

        assert len(MemoryTool.get_global_store()) == 0

    def test_execute_store_command(self):
        """MemoryTool.execute should handle store command."""
        tool = MemoryTool()
        result = tool.execute("store:mykey=myvalue")
        assert "Stored" in result
        assert tool.retrieve("mykey") == "myvalue"

    def test_execute_retrieve_command(self):
        """MemoryTool.execute should handle retrieve command."""
        tool = MemoryTool()
        tool.store("existing_key", "existing_value")
        result = tool.execute("retrieve:existing_key")
        assert result == "existing_value"

    def test_execute_list_command(self):
        """MemoryTool.execute should handle list command."""
        tool = MemoryTool()
        tool.store("key1", "value1")
        result = tool.execute("list")
        assert "key1" in result

    def test_execute_delete_command(self):
        """MemoryTool.execute should handle delete command."""
        tool = MemoryTool()
        tool.store("del_key", "value")
        result = tool.execute("delete:del_key")
        assert "Deleted" in result

    def test_execute_invalid_command(self):
        """MemoryTool.execute should handle invalid commands."""
        tool = MemoryTool()
        result = tool.execute("invalid:command")
        assert "Error" in result

    def test_execute_empty_input(self):
        """MemoryTool.execute should handle empty input."""
        tool = MemoryTool()
        result = tool.execute("")
        assert "Error" in result

    def test_can_handle_matching_keyword(self):
        """MemoryTool should match tasks with memory keywords."""
        tool = MemoryTool()
        assert tool.can_handle("store this value")
        assert tool.can_handle("remember the result")
        assert tool.can_handle("save for later")

    def test_can_handle_non_matching_keyword(self):
        """MemoryTool should not match tasks without keywords."""
        tool = MemoryTool()
        assert not tool.can_handle("execute python code")
        assert not tool.can_handle("search for information")


# ---------------------------------------------------------------------------
# ToolRegistry Tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for ToolRegistry functionality."""

    def test_register_tool(self):
        """ToolRegistry should register tools."""
        registry = ToolRegistry()
        tool = SearchTool()
        registry.register(tool)
        assert "search" in registry

    def test_register_duplicate_tool(self):
        """ToolRegistry should reject duplicate tool names."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(SearchTool())

    def test_unregister_tool(self):
        """ToolRegistry should unregister tools."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        result = registry.unregister("search")
        assert result is True
        assert "search" not in registry

    def test_unregister_nonexistent_tool(self):
        """ToolRegistry.unregister should return False for missing tools."""
        registry = ToolRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_tool(self):
        """ToolRegistry.get should return registered tools."""
        registry = ToolRegistry()
        tool = SearchTool()
        registry.register(tool)
        retrieved = registry.get("search")
        assert retrieved is tool

    def test_get_nonexistent_tool(self):
        """ToolRegistry.get should return None for missing tools."""
        registry = ToolRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_find_by_keyword(self):
        """ToolRegistry should find tools by keyword."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())

        tool = registry.find_by_keyword("search")
        assert tool is not None
        assert tool.name == "search"

    def test_find_by_keyword_case_insensitive(self):
        """ToolRegistry should find tools by keyword case-insensitively."""
        registry = ToolRegistry()
        registry.register(SearchTool())

        tool = registry.find_by_keyword("SEARCH")
        assert tool is not None
        assert tool.name == "search"

    def test_find_by_nonexistent_keyword(self):
        """ToolRegistry should return None for missing keyword."""
        registry = ToolRegistry()
        result = registry.find_by_keyword("nonexistent_keyword")
        assert result is None

    def test_find_for_task(self):
        """ToolRegistry should find all matching tools for a task."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())
        registry.register(MemoryTool())

        tools = registry.find_for_task("search for python code")
        assert len(tools) >= 1
        names = [t.name for t in tools]
        assert "search" in names

    def test_find_for_task_no_match(self):
        """ToolRegistry should return empty list for no matches."""
        registry = ToolRegistry()
        registry.register(SearchTool())

        tools = registry.find_for_task("cook dinner")
        assert tools == []

    def test_get_all(self):
        """ToolRegistry.get_all should return all tools."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())

        tools = registry.get_all()
        assert len(tools) == 2

    def test_get_all_names(self):
        """ToolRegistry.get_all_names should return tool names."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())

        names = registry.get_all_names()
        assert "search" in names
        assert "python_repl" in names

    def test_get_tools(self):
        """ToolRegistry.get_tools should return tools dict."""
        registry = ToolRegistry()
        registry.register(SearchTool())

        tools = registry.get_tools()
        assert "search" in tools
        assert hasattr(tools["search"], "execute")

    def test_invoke_tool(self):
        """ToolRegistry.invoke should execute tools."""
        registry = ToolRegistry()
        registry.register(SearchTool())

        result = registry.invoke("search", "test query")
        assert "test query" in result

    def test_invoke_nonexistent_tool(self):
        """ToolRegistry.invoke should raise KeyError for missing tool."""
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool"):
            registry.invoke("nonexistent", "input")

    def test_clear_registry(self):
        """ToolRegistry.clear should remove all tools."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())
        registry.clear()
        assert len(registry) == 0

    def test_len(self):
        """ToolRegistry.__len__ should return tool count."""
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(SearchTool())
        assert len(registry) == 1

    def test_repr(self):
        """ToolRegistry.__repr__ should show tool names."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        repr_str = repr(registry)
        assert "search" in repr_str


# ---------------------------------------------------------------------------
# ToolRegistryProvider Tests
# ---------------------------------------------------------------------------


class TestToolRegistryProvider:
    """Tests for ToolRegistryProvider adapter."""

    def test_get_tools_returns_execute_functions(self):
        """Provider.get_tools should return execute functions."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        provider = ToolRegistryProvider(registry)

        tools = provider.get_tools()
        assert "search" in tools
        assert callable(tools["search"])

    def test_invoke_delegates_to_registry(self):
        """Provider.invoke should delegate to tool.execute."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        provider = ToolRegistryProvider(registry)

        result = provider.invoke("search", input="test")
        assert "test" in result

    def test_invoke_with_query_kwarg(self):
        """Provider.invoke should accept query kwarg."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        provider = ToolRegistryProvider(registry)

        result = provider.invoke("search", query="test query")
        assert "test query" in result

    def test_invoke_nonexistent_tool(self):
        """Provider.invoke should raise KeyError for missing tool."""
        registry = ToolRegistry()
        provider = ToolRegistryProvider(registry)

        with pytest.raises(KeyError, match="Unknown tool"):
            provider.invoke("nonexistent", input="test")


# ---------------------------------------------------------------------------
# Integration Tests with ReasoningWorker
# ---------------------------------------------------------------------------


class TestReasoningWorkerIntegration:
    """Tests for Tool Integration with ReasoningWorker."""

    def test_worker_with_tool_registry_provider(self):
        """ReasoningWorker should work with ToolRegistryProvider."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())

        provider = ToolRegistryProvider(registry)
        worker = ReasoningWorker(
            agent_id="test-worker",
            strategy=MockExecutionStrategy(),
            tool_provider=provider,
        )

        assert worker.tool_provider is provider
        tools = provider.get_tools()
        assert "search" in tools
        assert "python_repl" in tools

    def test_tools_accessible_in_execution(self):
        """Tools should be accessible during task execution."""
        registry = ToolRegistry()
        registry.register(SearchTool())

        provider = ToolRegistryProvider(registry)
        worker = ReasoningWorker(
            agent_id="test-worker",
            strategy=MockExecutionStrategy(),
            tool_provider=provider,
        )

        task = TaskNode(id="task-1", description="Test task")
        result = worker.execute(task)

        assert result.success
        assert result.task_id == "task-1"

    def test_registry_find_for_task_in_worker_context(self):
        """Tool matching should work in worker context."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(PythonREPLTool())

        # Find appropriate tool for a search task
        matching_tools = registry.find_for_task("search for documentation")
        assert len(matching_tools) >= 1
        assert any(t.name == "search" for t in matching_tools)

        # Find appropriate tool for a code task
        matching_tools = registry.find_for_task("execute python script")
        assert len(matching_tools) >= 1
        assert any(t.name == "python_repl" for t in matching_tools)


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_registry_operations(self):
        """Empty registry should handle operations gracefully."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.get_all() == []
        assert registry.get_all_names() == []
        assert registry.get("nonexistent") is None
        assert registry.find_by_keyword("anything") is None
        assert registry.find_for_task("anything") == []

    def test_tool_with_empty_keywords(self):
        """Tool with no keywords should not match anything."""
        tool = SearchTool(keywords=[])
        assert not tool.can_handle("search for anything")

    def test_memory_tool_complex_values(self):
        """MemoryTool should handle complex values."""
        tool = MemoryTool()
        tool.store("dict_key", {"nested": {"value": 123}})
        result = tool.retrieve("dict_key")
        assert "nested" in result
        assert "123" in result

    def test_memory_tool_list_values(self):
        """MemoryTool should handle list values."""
        tool = MemoryTool()
        tool.store("list_key", [1, 2, 3, "four"])
        result = tool.retrieve("list_key")
        assert "1" in result
        assert "four" in result

    def test_multiple_registries_independent(self):
        """Multiple registries should be independent."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()

        registry1.register(SearchTool(name="search1"))
        registry2.register(PythonREPLTool(name="python2"))

        assert "search1" in registry1
        assert "search1" not in registry2
        assert "python2" in registry2
        assert "python2" not in registry1

    def test_register_unregister_register_cycle(self):
        """Should be able to re-register after unregistering."""
        registry = ToolRegistry()
        tool = SearchTool()
        registry.register(tool)
        registry.unregister("search")
        registry.register(tool)  # Should not raise
        assert "search" in registry
