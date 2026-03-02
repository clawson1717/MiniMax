"""Tests for CLI module."""

import subprocess
import sys
from pathlib import Path

import pytest

# Project root for subprocess calls
PROJECT_ROOT = Path(__file__).parent.parent


class TestCLI:
    """Test CLI commands."""

    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "benchmark" in result.stdout
        assert "evaluate" in result.stdout
        assert "efficiency" in result.stdout
        assert "visualize" in result.stdout

    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        # No command should show help (exit 1) or help (exit 0)
        assert "usage:" in result.stdout.lower() or "available commands" in result.stdout.lower()

    def test_run_help(self):
        """Test run command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "run", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "--agents" in result.stdout
        assert "--task" in result.stdout
        assert "--mode" in result.stdout

    def test_benchmark_help(self):
        """Test benchmark command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()

    def test_evaluate_help(self):
        """Test evaluate command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "evaluate", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "evaluate" in result.stdout.lower()

    def test_efficiency_help(self):
        """Test efficiency command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "efficiency", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "efficiency" in result.stdout.lower()

    def test_visualize_help(self):
        """Test visualize command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "visualize" in result.stdout.lower()

    def test_run_mock(self):
        """Test run command with mock agents."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "run", "--mock", "--agents", "2"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Multi-Agent" in result.stdout
        assert "2" in result.stdout

    def test_run_different_task_types(self):
        """Test run command with different task types."""
        for task_type in ["reasoning", "qa", "math"]:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "run", "--mock", "--task", task_type],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert result.returncode == 0
            assert task_type in result.stdout.lower()

    def test_run_different_modes(self):
        """Test run command with different routing modes."""
        for mode in ["broadcast", "targeted", "capacity_weighted"]:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "run", "--mock", "--mode", mode],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert result.returncode == 0
            assert mode in result.stdout

    def test_benchmark_mock(self):
        """Test benchmark command with mock models."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", "--mock"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Benchmark" in result.stdout

    def test_benchmark_with_output(self):
        """Test benchmark command with output file."""
        output_file = PROJECT_ROOT / "test_benchmark_output.json"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "benchmark", "--mock", "-o", str(output_file)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert result.returncode == 0
            assert output_file.exists()
        finally:
            if output_file.exists():
                output_file.unlink()

    def test_evaluate(self):
        """Test evaluate command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "evaluate"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "info_efficient" in result.stdout or "info-efficient" in result.stdout

    def test_evaluate_with_output(self):
        """Test evaluate command with output file."""
        output_file = PROJECT_ROOT / "test_eval_output.json"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "evaluate", "-o", str(output_file)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert result.returncode == 0
            assert output_file.exists()
        finally:
            if output_file.exists():
                output_file.unlink()

    def test_efficiency(self):
        """Test efficiency command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "efficiency"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Efficiency" in result.stdout

    def test_efficiency_with_output(self):
        """Test efficiency command with output file."""
        output_file = PROJECT_ROOT / "test_efficiency_output.json"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "efficiency", "-o", str(output_file)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert result.returncode == 0
            assert output_file.exists()
        finally:
            if output_file.exists():
                output_file.unlink()

    def test_visualize_capacity(self):
        """Test visualize capacity command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "capacity"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Capacity" in result.stdout
        assert "Agent" in result.stdout

    def test_visualize_routing(self):
        """Test visualize routing command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "routing"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Router" in result.stdout or "Routing" in result.stdout

    def test_visualize_efficiency(self):
        """Test visualize efficiency command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "efficiency"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Efficiency" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_full_pipeline(self):
        """Test running full CLI pipeline."""
        # Run
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "run", "--mock", "--agents", "3"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0

        # Benchmark
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", "--mock"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0

        # Evaluate
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "evaluate"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
