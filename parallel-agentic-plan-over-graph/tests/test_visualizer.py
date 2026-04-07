"""Tests for src.visualizer — graph export and timeline rendering."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.benchmark import BenchmarkResult, TimelineEntry
from src.models import TaskGraph, TaskNode, TaskStatus
from src.visualizer import (
    STATUS_COLORS,
    _build_dot_source,
    _truncate,
    export_dot,
    export_png,
    export_svg,
    render_execution_timeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Yield a temporary directory, cleaned up afterwards."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def simple_graph() -> TaskGraph:
    """A basic 3-node graph: A → B → C."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="A", description="First task", dependencies=[]))
    graph.add_node(TaskNode(id="B", description="Second task", dependencies=["A"]))
    graph.add_node(TaskNode(id="C", description="Third task", dependencies=["B"]))
    return graph


@pytest.fixture
def diamond_graph() -> TaskGraph:
    """Diamond: A → B, A → C, B → D, C → D."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="A", description="Root", dependencies=[]))
    graph.add_node(TaskNode(id="B", description="Left branch", dependencies=["A"]))
    graph.add_node(TaskNode(id="C", description="Right branch", dependencies=["A"]))
    graph.add_node(TaskNode(id="D", description="Join", dependencies=["B", "C"]))
    return graph


@pytest.fixture
def mixed_status_graph() -> TaskGraph:
    """Graph with nodes in different statuses."""
    graph = TaskGraph()
    n1 = TaskNode(id="done", description="Completed task", dependencies=[])
    n1.status = TaskStatus.COMPLETED
    n2 = TaskNode(id="running", description="Running task", dependencies=[])
    n2.status = TaskStatus.RUNNING
    n3 = TaskNode(id="fail", description="Failed task", dependencies=[])
    n3.status = TaskStatus.FAILED
    n4 = TaskNode(id="skip", description="Skipped task", dependencies=[])
    n4.status = TaskStatus.SKIPPED
    n5 = TaskNode(id="wait", description="Pending task", dependencies=[])
    n5.status = TaskStatus.PENDING
    for n in [n1, n2, n3, n4, n5]:
        graph.add_node(n)
    return graph


@pytest.fixture
def empty_graph() -> TaskGraph:
    """An empty graph with no nodes."""
    return TaskGraph()


@pytest.fixture
def single_node_graph() -> TaskGraph:
    """Graph with a single node."""
    graph = TaskGraph()
    graph.add_node(TaskNode(id="solo", description="Lone task", dependencies=[]))
    return graph


@pytest.fixture
def large_graph() -> TaskGraph:
    """Graph with 50 chained nodes."""
    graph = TaskGraph()
    for i in range(50):
        deps = [f"node_{i - 1}"] if i > 0 else []
        graph.add_node(
            TaskNode(id=f"node_{i}", description=f"Task number {i}", dependencies=deps)
        )
    return graph


# ---------------------------------------------------------------------------
# Truncation helper
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 40) == "hello"

    def test_long_string_truncated(self):
        result = _truncate("a" * 60, 40)
        assert len(result) == 40
        assert result.endswith("…")

    def test_exact_length(self):
        assert _truncate("a" * 40, 40) == "a" * 40


# ---------------------------------------------------------------------------
# DOT source generation
# ---------------------------------------------------------------------------


class TestBuildDotSource:
    def test_valid_dot_structure(self, simple_graph):
        dot = _build_dot_source(simple_graph)
        assert dot.startswith("digraph PAPoG {")
        assert dot.endswith("}")

    def test_contains_all_nodes(self, simple_graph):
        dot = _build_dot_source(simple_graph)
        for nid in ("A", "B", "C"):
            assert f'"{nid}"' in dot

    def test_contains_edges(self, simple_graph):
        dot = _build_dot_source(simple_graph)
        assert '"A" -> "B"' in dot
        assert '"B" -> "C"' in dot

    def test_diamond_edges(self, diamond_graph):
        dot = _build_dot_source(diamond_graph)
        assert '"A" -> "B"' in dot
        assert '"A" -> "C"' in dot
        assert '"B" -> "D"' in dot
        assert '"C" -> "D"' in dot

    def test_empty_graph(self, empty_graph):
        dot = _build_dot_source(empty_graph)
        assert "digraph PAPoG {" in dot
        # No node definitions
        assert "fillcolor" not in dot

    def test_single_node(self, single_node_graph):
        dot = _build_dot_source(single_node_graph)
        assert '"solo"' in dot
        # No edges
        assert "->" not in dot


# ---------------------------------------------------------------------------
# Status-based coloring
# ---------------------------------------------------------------------------


class TestStatusColoring:
    def test_all_statuses_have_colors(self):
        for status in TaskStatus:
            assert status in STATUS_COLORS

    def test_color_in_dot_output(self, mixed_status_graph):
        dot = _build_dot_source(mixed_status_graph)
        assert 'fillcolor="green3"' in dot  # COMPLETED
        assert 'fillcolor="dodgerblue"' in dot  # RUNNING
        assert 'fillcolor="red"' in dot  # FAILED
        assert 'fillcolor="orange"' in dot  # SKIPPED
        assert 'fillcolor="gray"' in dot  # PENDING


# ---------------------------------------------------------------------------
# export_dot
# ---------------------------------------------------------------------------


class TestExportDot:
    def test_writes_file(self, simple_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "test.dot")
        source = export_dot(simple_graph, outpath)
        assert Path(outpath).exists()
        content = Path(outpath).read_text()
        assert content == source
        assert "digraph" in content

    def test_empty_graph_export(self, empty_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "empty.dot")
        source = export_dot(empty_graph, outpath)
        assert "digraph" in source

    def test_large_graph_export(self, large_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "large.dot")
        source = export_dot(large_graph, outpath)
        assert source.count("->") == 49  # 50 nodes, 49 edges in chain


# ---------------------------------------------------------------------------
# export_png / export_svg (require dot binary)
# ---------------------------------------------------------------------------

_has_dot = shutil.which("dot") is not None


@pytest.mark.skipif(not _has_dot, reason="graphviz dot binary not installed")
class TestExportPng:
    def test_creates_png(self, simple_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "test.png")
        result_path = export_png(simple_graph, outpath)
        assert Path(result_path).exists()
        assert result_path.endswith(".png")


@pytest.mark.skipif(not _has_dot, reason="graphviz dot binary not installed")
class TestExportSvg:
    def test_creates_svg(self, simple_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "test.svg")
        result_path = export_svg(simple_graph, outpath)
        assert Path(result_path).exists()
        assert result_path.endswith(".svg")


class TestExportPngNoDot:
    """When dot binary is unavailable, export_png should raise RuntimeError."""

    @pytest.mark.skipif(_has_dot, reason="dot binary IS installed")
    def test_raises_without_dot(self, simple_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "fail.png")
        with pytest.raises(RuntimeError, match="dot"):
            export_png(simple_graph, outpath)

    @pytest.mark.skipif(_has_dot, reason="dot binary IS installed")
    def test_svg_raises_without_dot(self, simple_graph, tmp_dir):
        outpath = os.path.join(tmp_dir, "fail.svg")
        with pytest.raises(RuntimeError, match="dot"):
            export_svg(simple_graph, outpath)


# ---------------------------------------------------------------------------
# Execution timeline
# ---------------------------------------------------------------------------


class TestRenderExecutionTimeline:
    def test_basic_timeline(self, tmp_dir):
        result = BenchmarkResult(
            scenario_name="test",
            goal="test goal",
            execution_timeline=[
                TimelineEntry(
                    timestamp=0.0, node_id="A", from_status="pending", to_status="running"
                ),
                TimelineEntry(
                    timestamp=0.5, node_id="A", from_status="running", to_status="completed"
                ),
                TimelineEntry(
                    timestamp=0.1, node_id="B", from_status="pending", to_status="running"
                ),
                TimelineEntry(
                    timestamp=0.8, node_id="B", from_status="running", to_status="failed"
                ),
            ],
        )
        outpath = os.path.join(tmp_dir, "timeline.dot")
        source = render_execution_timeline(result, outpath)
        assert "digraph Timeline" in source
        assert '"A"' in source
        assert '"B"' in source
        assert 'fillcolor="green3"' in source  # A completed
        assert 'fillcolor="red"' in source  # B failed
        assert Path(outpath).exists()

    def test_empty_timeline(self, tmp_dir):
        result = BenchmarkResult(
            scenario_name="empty",
            goal="nothing",
            execution_timeline=[],
        )
        outpath = os.path.join(tmp_dir, "empty_timeline.dot")
        source = render_execution_timeline(result, outpath)
        assert "digraph Timeline" in source
        assert "No events" in source

    def test_timeline_writes_dot_for_png_path(self, tmp_dir):
        """Even when requesting .png, the DOT source should be written."""
        result = BenchmarkResult(
            scenario_name="test",
            goal="test",
            execution_timeline=[
                TimelineEntry(
                    timestamp=1.0, node_id="X", from_status="pending", to_status="running"
                ),
                TimelineEntry(
                    timestamp=2.0, node_id="X", from_status="running", to_status="completed"
                ),
            ],
        )
        outpath = os.path.join(tmp_dir, "timeline.png")
        source = render_execution_timeline(result, outpath)
        # DOT file should have been written
        dot_path = os.path.join(tmp_dir, "timeline.dot")
        assert Path(dot_path).exists()
        assert "digraph" in source
