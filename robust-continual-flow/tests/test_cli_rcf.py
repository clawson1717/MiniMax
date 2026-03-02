"""
Tests for RCF CLI and Dashboard.
"""

import pytest
import subprocess
import sys
from pathlib import Path

from src.cli_rcf import Dashboard


class TestDashboard:
    """Tests for Dashboard class."""
    
    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = Dashboard()
        
        assert dashboard.refresh_rate == 0.5
        assert dashboard.uncertainty_history == []
        assert dashboard.fatigue_history == []
    
    def test_custom_refresh_rate(self):
        """Test custom refresh rate."""
        dashboard = Dashboard(refresh_rate=1.0)
        
        assert dashboard.refresh_rate == 1.0
    
    def test_render_header(self, capsys):
        """Test header rendering."""
        dashboard = Dashboard()
        dashboard.render_header("Test Header")
        
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" in captured.out
    
    def test_render_metrics(self, capsys):
        """Test metrics rendering."""
        dashboard = Dashboard()
        dashboard.render_metrics(
            step=5,
            max_steps=20,
            uncertainty=0.4,
            fatigue=0.3,
            compute_budget=4.5,
        )
        
        captured = capsys.readouterr()
        assert "Step" in captured.out
        assert "Uncertainty" in captured.out
        assert "Fatigue" in captured.out
        assert "0.400" in captured.out or "0.40" in captured.out
    
    def test_render_graph_empty(self, capsys):
        """Test graph rendering with no data."""
        dashboard = Dashboard()
        dashboard.render_graph([], "Test Graph")
        
        captured = capsys.readouterr()
        assert "No data yet" in captured.out
    
    def test_render_graph_with_data(self, capsys):
        """Test graph rendering with data."""
        dashboard = Dashboard()
        dashboard.render_graph([0.1, 0.3, 0.5, 0.7], "Test Graph")
        
        captured = capsys.readouterr()
        assert "Test Graph" in captured.out
        assert "Min:" in captured.out
        assert "Max:" in captured.out
    
    def test_render_alerts_empty(self, capsys):
        """Test alerts rendering with no alerts."""
        dashboard = Dashboard()
        dashboard.render_alerts([])
        
        captured = capsys.readouterr()
        # Should not print anything for empty alerts
        assert "Alerts" not in captured.out
    
    def test_render_alerts_with_data(self, capsys):
        """Test alerts rendering with alerts."""
        dashboard = Dashboard()
        dashboard.render_alerts(["Test alert 1", "Test alert 2"])
        
        captured = capsys.readouterr()
        assert "Alerts" in captured.out
        assert "Test alert 1" in captured.out
    
    def test_render_trajectory_summary(self, capsys):
        """Test trajectory summary rendering."""
        dashboard = Dashboard()
        dashboard.render_trajectory_summary()
        
        captured = capsys.readouterr()
        assert "Trajectory Graph" in captured.out


class TestCLICommands:
    """Tests for CLI commands."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli_rcf", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "RCF CLI" in result.stdout
    
    def test_cli_benchmark_help(self):
        """Test benchmark subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli_rcf", "benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "quality" in result.stdout.lower()
    
    def test_cli_analyze_help(self):
        """Test analyze subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli_rcf", "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "sessions" in result.stdout.lower()
    
    def test_cli_dashboard_help(self):
        """Test dashboard subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli_rcf", "dashboard", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "steps" in result.stdout.lower()
    
    def test_cli_analyze_runs(self):
        """Test that analyze command runs successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli_rcf", "analyze", "--sessions", "3"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Results:" in result.stdout or "sessions" in result.stdout.lower()
    
    def test_cli_benchmark_runs(self):
        """Test that benchmark command runs successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli_rcf", "benchmark", "--quality", "0.8"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "success" in result.stdout.lower() or "Benchmark" in result.stdout


class TestDashboardIntegration:
    """Integration tests for dashboard."""
    
    def test_demo_mode_runs(self):
        """Test that demo mode runs without errors."""
        dashboard = Dashboard(refresh_rate=0.01)  # Fast refresh for testing
        
        # Run a short demo
        import io
        import contextlib
        
        # Capture output to avoid cluttering test output
        with contextlib.redirect_stdout(io.StringIO()):
            dashboard.run_demo(steps=3)
        
        # Should have recorded some data
        assert len(dashboard.uncertainty_history) == 3
        assert len(dashboard.fatigue_history) == 3
    
    def test_metrics_accumulation(self):
        """Test that metrics accumulate over time."""
        dashboard = Dashboard()
        
        # Simulate multiple render cycles
        for i in range(5):
            dashboard.uncertainty_history.append(0.1 * i)
            dashboard.fatigue_history.append(0.05 * i)
        
        assert len(dashboard.uncertainty_history) == 5
        assert dashboard.fatigue_history[-1] > dashboard.fatigue_history[0]
