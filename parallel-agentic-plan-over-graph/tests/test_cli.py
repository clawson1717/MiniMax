"""Tests for src.cli — argument parsing and subcommand dispatch."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import build_parser, main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Verify argument parsing for all subcommands."""

    def test_run_minimal(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--goal", "Build a chatbot"])
        assert args.command == "run"
        assert args.goal == "Build a chatbot"
        assert args.workers == 4
        assert args.max_retries == 2
        assert args.output_dir is None

    def test_run_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "run", "--goal", "Build X",
            "--workers", "8", "--max-retries", "5",
            "--output-dir", "/tmp/out",
        ])
        assert args.workers == 8
        assert args.max_retries == 5
        assert args.output_dir == "/tmp/out"

    def test_visualize_minimal(self):
        parser = build_parser()
        args = parser.parse_args(["visualize", "--goal", "Analyse data"])
        assert args.command == "visualize"
        assert args.goal == "Analyse data"
        assert args.format == "dot"
        assert args.output is None

    def test_visualize_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "visualize", "--goal", "Plan",
            "--output", "my_graph.png",
            "--format", "png",
        ])
        assert args.output == "my_graph.png"
        assert args.format == "png"

    def test_visualize_format_choices(self):
        parser = build_parser()
        for fmt in ("dot", "png", "svg"):
            args = parser.parse_args(["visualize", "--goal", "X", "--format", fmt])
            assert args.format == fmt

    def test_visualize_invalid_format(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["visualize", "--goal", "X", "--format", "pdf"])

    def test_scenario_minimal(self):
        parser = build_parser()
        args = parser.parse_args(["scenario", "--name", "deep_research"])
        assert args.command == "scenario"
        assert args.name == "deep_research"
        assert args.workers == 4

    def test_scenario_with_workers(self):
        parser = build_parser()
        args = parser.parse_args(["scenario", "--name", "test", "--workers", "2"])
        assert args.workers == 2

    def test_list_scenarios(self):
        parser = build_parser()
        args = parser.parse_args(["list-scenarios"])
        assert args.command == "list-scenarios"

    def test_no_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# Subcommand execution
# ---------------------------------------------------------------------------


class TestMainRun:
    """Test the 'run' subcommand executes without crashing."""

    def test_run_basic_goal(self, capsys):
        exit_code = main(["run", "--goal", "Build a simple chatbot"])
        captured = capsys.readouterr()
        assert "PAPoG Benchmark Results" in captured.out
        assert exit_code == 0

    def test_run_with_output_dir(self, tmp_dir, capsys):
        exit_code = main([
            "run", "--goal", "Build a chatbot",
            "--output-dir", tmp_dir,
        ])
        captured = capsys.readouterr()
        assert "PAPoG Benchmark Results" in captured.out
        assert Path(os.path.join(tmp_dir, "graph.dot")).exists()
        assert exit_code == 0


class TestMainVisualize:
    """Test the 'visualize' subcommand."""

    def test_visualize_dot(self, tmp_dir, capsys):
        outpath = os.path.join(tmp_dir, "viz.dot")
        exit_code = main(["visualize", "--goal", "Plan a project", "--output", outpath])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert Path(outpath).exists()
        content = Path(outpath).read_text()
        assert "digraph" in content

    def test_visualize_dot_default_name(self, capsys, monkeypatch, tmp_dir):
        monkeypatch.chdir(tmp_dir)
        exit_code = main(["visualize", "--goal", "Anything"])
        assert exit_code == 0
        assert Path(os.path.join(tmp_dir, "graph.dot")).exists()

    def test_visualize_png_fallback_no_dot(self, tmp_dir, capsys):
        """When dot binary is missing, PNG falls back to DOT."""
        _has_dot = shutil.which("dot") is not None
        if _has_dot:
            pytest.skip("dot binary is installed — can't test fallback")
        outpath = os.path.join(tmp_dir, "viz.png")
        exit_code = main([
            "visualize", "--goal", "Test",
            "--output", outpath,
            "--format", "png",
        ])
        # Should exit 1 with fallback
        assert exit_code == 1
        # DOT fallback should exist
        fallback = os.path.join(tmp_dir, "viz.dot")
        assert Path(fallback).exists()


class TestMainScenario:
    """Test the 'scenario' subcommand."""

    def test_run_deep_research(self, capsys):
        exit_code = main(["scenario", "--name", "deep_research"])
        captured = capsys.readouterr()
        assert "PAPoG Benchmark Results" in captured.out
        assert isinstance(exit_code, int)

    def test_run_invalid_scenario(self, capsys):
        exit_code = main(["scenario", "--name", "nonexistent_xyz"])
        assert exit_code == 1


class TestMainListScenarios:
    """Test the 'list-scenarios' subcommand."""

    def test_lists_scenarios(self, capsys):
        exit_code = main(["list-scenarios"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Available scenarios:" in captured.out
        # Should have at least deep_research
        assert "deep_research" in captured.out


class TestMainNoCommand:
    """Test that no command prints help and exits 0."""

    def test_no_args(self, capsys):
        exit_code = main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        # Should show usage info
        assert "papog" in captured.out.lower() or "usage" in captured.out.lower()
