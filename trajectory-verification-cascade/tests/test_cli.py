import json
import os
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
from src.cli import TVCCLI, main
from src.agent import TVCReport, NodeStatus

class TestTVCCLI:
    """Unit tests for the TVC CLI."""

    @pytest.fixture
    def cli(self):
        return TVCCLI()

    @pytest.fixture
    def temp_task_file(self, tmp_path):
        task_data = {
            "task": "Test Task",
            "steps": ["Step 1", "Step 2"]
        }
        task_file = tmp_path / "task.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)
        return str(task_file)

    @pytest.fixture
    def temp_report_file(self, tmp_path):
        report_data = {
            "task": "Test Report",
            "success": True,
            "trajectory": [
                {
                    "id": "node-1",
                    "content": "Step 1 content",
                    "status": "VERIFIED",
                    "checklist_items": [{"criterion": "C1", "passed": True, "evidence": "E1"}]
                }
            ],
            "metrics": {"steps": 1}
        }
        report_file = tmp_path / "report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f)
        return str(report_file)

    def test_run_task_loading(self, cli, temp_task_file):
        """Tests if the CLI can load a task file and call the agent."""
        with patch('src.cli.TVCAgent') as MockAgent:
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.process_task.return_value = TVCReport(
                task="Test Task", success=True, trajectory=[], metrics={}
            )
            cli.run_task(temp_task_file)
            mock_agent_instance.process_task.assert_called_once()

    def test_run_task_output_export(self, cli, temp_task_file, tmp_path):
        """Tests if the CLI exports the report to JSON."""
        output_file = str(tmp_path / "output.json")
        cli.run_task(temp_task_file, output_file=output_file)
        
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            data = json.load(f)
            assert data["task"] == "Test Task"
            assert "trajectory" in data

    def test_benchmark_command(self, cli):
        """Tests if the benchmark command runs and produces results."""
        with patch.object(cli.benchmark_suite, 'run_benchmark') as mock_run:
            mock_run.return_value = []
            with patch.object(cli.benchmark_suite, 'calculate_summary_metrics') as mock_calc:
                mock_calc.return_value = {
                    "total_tasks": 0,
                    "verification_accuracy": 0,
                    "adversarial_failure_detection_rate": 0,
                    "pruning_efficiency": 0,
                    "avg_steps": 0
                }
                cli.run_benchmark()
                mock_run.assert_called_once()

    def test_visualize_trajectory(self, cli, temp_report_file):
        """Tests the visualization command."""
        # This mostly tests that it doesn't crash
        with patch('rich.tree.Tree.add') as mock_add:
            cli.visualize_trajectory(temp_report_file)
            assert mock_add.called

    def test_main_cli_arg_parsing(self, temp_task_file):
        """Tests the main entry point with arguments."""
        with patch('sys.argv', ['cli.py', 'run', temp_task_file]):
            with patch('src.cli.TVCCLI.run_task') as mock_run:
                main()
                mock_run.assert_called_once_with(temp_task_file, None)

    def test_main_benchmark_cli(self):
        """Tests the benchmark command via main."""
        with patch('sys.argv', ['cli.py', 'benchmark']):
            with patch('src.cli.TVCCLI.run_benchmark') as mock_bench:
                main()
                mock_bench.assert_called_once_with(None)
