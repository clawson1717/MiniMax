import pytest
import subprocess
import sys
import os

def test_cli_help():
    """Verify the CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Adversarial Domain Diffuser CLI" in result.stdout

def test_cli_run_basic():
    """Verify basic run command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "run", "--iterations", "1"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Adversarial Loop Results" in result.stdout

def test_cli_run_visualize():
    """Verify run command with visualization works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "run", "--iterations", "2", "--visualize"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Gap Closing Progress" in result.stdout

def test_cli_eval_basic():
    """Verify basic eval command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "eval", "--domain", "MedicalQA", "--iterations", "1"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Evaluation Results: MedicalQA" in result.stdout
