"""Comprehensive tests for the Architect agent and decomposition strategies."""

from __future__ import annotations

import pytest

from src.architect import (
    Architect,
    DecompositionStrategy,
    DecompositionStrategyABC,
    KeywordDecompositionStrategy,
    MockLLMStrategy,
)
from src.models import TaskGraph, TaskNode, TaskStatus


# ---- helpers ---------------------------------------------------------------

class _SingleNodeStrategy(DecompositionStrategyABC):
    """Minimal strategy that always produces one task."""

    def decompose(self, goal: str) -> TaskGraph:
        graph = TaskGraph()
        graph.add_node(TaskNode(id="only", description=goal))
        return graph


class _ProtocolOnlyStrategy:
    """Satisfies the Protocol (duck-typed) but does NOT inherit the ABC."""

    def decompose(self, goal: str) -> TaskGraph:
        graph = TaskGraph()
        graph.add_node(TaskNode(id="duck", description=goal))
        return graph


class _BadReturnStrategy:
    """Returns something that isn't a TaskGraph."""

    def decompose(self, goal: str):  # type: ignore[return]
        return {"not": "a graph"}


# ---- MockLLMStrategy tests -------------------------------------------------

class TestMockLLMStrategy:
    """Tests for the default MockLLMStrategy."""

    def test_produces_valid_taskgraph(self):
        strategy = MockLLMStrategy()
        graph = strategy.decompose("Build a website")
        assert isinstance(graph, TaskGraph)

    def test_has_four_nodes(self):
        graph = MockLLMStrategy().decompose("Build a website")
        assert len(graph.nodes) == 4

    def test_node_ids(self):
        graph = MockLLMStrategy().decompose("Build a website")
        assert set(graph.nodes.keys()) == {"analyse", "design", "implement", "test"}

    def test_is_valid_dag(self):
        graph = MockLLMStrategy().decompose("Build a website")
        graph.validate_dag()  # should not raise

    def test_dependencies_are_correct(self):
        graph = MockLLMStrategy().decompose("Ship feature X")
        assert graph.get_node("analyse").dependencies == []
        assert graph.get_node("design").dependencies == ["analyse"]
        assert graph.get_node("implement").dependencies == ["design"]
        assert sorted(graph.get_node("test").dependencies) == ["design", "implement"]

    def test_topological_order_valid(self):
        graph = MockLLMStrategy().decompose("anything")
        order = graph.topological_order()
        # analyse must come before design; design before implement; both before test
        assert order.index("analyse") < order.index("design")
        assert order.index("design") < order.index("implement")
        assert order.index("design") < order.index("test")
        assert order.index("implement") < order.index("test")

    def test_descriptions_embed_goal(self):
        goal = "Deploy the service"
        graph = MockLLMStrategy().decompose(goal)
        for node in graph.nodes.values():
            assert goal in node.description

    def test_all_nodes_start_pending(self):
        graph = MockLLMStrategy().decompose("x")
        for node in graph.nodes.values():
            assert node.status == TaskStatus.PENDING

    def test_metadata_present(self):
        graph = MockLLMStrategy().decompose("y")
        assert graph.get_node("analyse").metadata["phase"] == "planning"
        assert graph.get_node("test").metadata["phase"] == "verification"

    def test_empty_goal_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            MockLLMStrategy().decompose("")

    def test_whitespace_goal_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            MockLLMStrategy().decompose("   ")


# ---- KeywordDecompositionStrategy tests ------------------------------------

class TestKeywordDecompositionStrategy:
    """Tests for the keyword-based heuristic strategy."""

    def test_comma_separated(self):
        graph = KeywordDecompositionStrategy().decompose("fetch data, process data, store results")
        assert len(graph.nodes) == 3
        graph.validate_dag()

    def test_and_delimiter(self):
        graph = KeywordDecompositionStrategy().decompose("read input and transform and output")
        assert len(graph.nodes) == 3

    def test_then_delimiter(self):
        graph = KeywordDecompositionStrategy().decompose("plan then execute then verify")
        assert len(graph.nodes) == 3

    def test_sequential_dependencies(self):
        graph = KeywordDecompositionStrategy().decompose("A, B, C")
        assert graph.get_node("task_0").dependencies == []
        assert graph.get_node("task_1").dependencies == ["task_0"]
        assert graph.get_node("task_2").dependencies == ["task_1"]

    def test_single_phrase(self):
        graph = KeywordDecompositionStrategy().decompose("just one thing")
        assert len(graph.nodes) == 1
        assert graph.get_node("task_0").dependencies == []

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            KeywordDecompositionStrategy().decompose("")

    def test_valid_dag(self):
        graph = KeywordDecompositionStrategy().decompose("a, b, c, d, e")
        graph.validate_dag()

    def test_mixed_delimiters(self):
        graph = KeywordDecompositionStrategy().decompose("plan; design and implement, test then deploy")
        assert len(graph.nodes) == 5


# ---- Architect tests -------------------------------------------------------

class TestArchitect:
    """Tests for the Architect orchestrator."""

    def test_default_strategy_is_mock(self):
        arch = Architect()
        assert isinstance(arch.strategy, MockLLMStrategy)

    def test_decompose_returns_taskgraph(self):
        graph = Architect().decompose("Build a thing")
        assert isinstance(graph, TaskGraph)

    def test_decompose_validates_dag(self):
        # MockLLMStrategy always makes a valid DAG; just confirm no exception
        graph = Architect().decompose("Ship it")
        graph.validate_dag()

    def test_custom_strategy_injection_abc(self):
        arch = Architect(strategy=_SingleNodeStrategy())
        graph = arch.decompose("simple goal")
        assert len(graph.nodes) == 1
        assert "simple goal" in graph.get_node("only").description

    def test_custom_strategy_injection_protocol(self):
        arch = Architect(strategy=_ProtocolOnlyStrategy())
        graph = arch.decompose("duck typed")
        assert "duck" in graph.nodes

    def test_strategy_setter(self):
        arch = Architect()
        assert isinstance(arch.strategy, MockLLMStrategy)
        arch.strategy = KeywordDecompositionStrategy()
        assert isinstance(arch.strategy, KeywordDecompositionStrategy)

    def test_strategy_swap_changes_output(self):
        arch = Architect()
        g1 = arch.decompose("do X and Y and Z")
        assert len(g1.nodes) == 4  # MockLLMStrategy always 4

        arch.strategy = KeywordDecompositionStrategy()
        g2 = arch.decompose("do X and Y and Z")
        assert len(g2.nodes) == 3  # "do X", "Y", "Z"

    def test_empty_goal_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Architect().decompose("")

    def test_whitespace_goal_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Architect().decompose("  \t\n  ")

    def test_bad_return_type_raises(self):
        arch = Architect(strategy=_BadReturnStrategy())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TaskGraph"):
            arch.decompose("anything")

    def test_complex_goal(self):
        """A longer, realistic goal should still produce a valid graph."""
        goal = (
            "Build a distributed microservice architecture with user auth, "
            "payment processing, notification service, and monitoring dashboard"
        )
        graph = Architect().decompose(goal)
        assert len(graph.nodes) > 0
        graph.validate_dag()

    def test_protocol_isinstance_check(self):
        """Verify the Protocol is runtime-checkable."""
        assert isinstance(MockLLMStrategy(), DecompositionStrategy)
        assert isinstance(_ProtocolOnlyStrategy(), DecompositionStrategy)
        assert isinstance(KeywordDecompositionStrategy(), DecompositionStrategy)

    def test_keyword_via_architect(self):
        arch = Architect(strategy=KeywordDecompositionStrategy())
        graph = arch.decompose("gather, analyse, report")
        assert len(graph.nodes) == 3
        order = graph.topological_order()
        assert order == ["task_0", "task_1", "task_2"]

    def test_get_ready_nodes_on_decomposed_graph(self):
        """Ready nodes on a fresh graph should be the roots."""
        graph = Architect().decompose("anything")
        ready = graph.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "analyse"

    def test_serialisation_roundtrip(self):
        """Decomposed graph should survive to_dict / from_dict."""
        graph = Architect().decompose("roundtrip test")
        data = graph.to_dict()
        restored = TaskGraph.from_dict(data)
        assert set(restored.nodes.keys()) == set(graph.nodes.keys())
        restored.validate_dag()
