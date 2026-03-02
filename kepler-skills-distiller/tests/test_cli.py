"""
Tests for Kepler Skills Distiller CLI
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import sys


class TestCLI:
    """Tests for CLI commands."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "Kepler Skills Distiller CLI" in result.stdout
    
    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        # Should exit with 1 and show help
        assert result.returncode == 1
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()
    
    def test_discover_help(self):
        """Test discover command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "discover", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "discover" in result.stdout.lower()
    
    def test_distill_help(self):
        """Test distill command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "distill", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "distill" in result.stdout.lower()
    
    def test_benchmark_help(self):
        """Test benchmark command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()
    
    def test_benchmark_run_mock(self, temp_dir):
        """Test running benchmark with mock models."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", "--mock", "--max-complexity", "1"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Benchmark" in result.stdout
        assert "Expert accuracy" in result.stdout
    
    def test_benchmark_with_output(self, temp_dir):
        """Test benchmark with output directory."""
        output_dir = temp_dir / "benchmark_results"
        
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", 
             "--mock", "--max-complexity", "1", 
             "--output-dir", str(output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        
        assert result.returncode == 0
        # Output directory should be created
        assert output_dir.exists()
    
    def test_visualize_help(self):
        """Test visualize command help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "visualize" in result.stdout.lower()
    
    def test_visualize_capacity(self):
        """Test capacity visualization."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "capacity"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "Capacity" in result.stdout
    
    def test_visualize_curriculum(self):
        """Test curriculum visualization."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "curriculum"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "Curriculum" in result.stdout
        assert "Remember" in result.stdout
        assert "Create" in result.stdout
    
    def test_visualize_skills_empty(self, temp_dir):
        """Test skills visualization with no skills."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "skills",
             "--skills-dir", str(temp_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "No skills" in result.stdout
    
    def test_visualize_skills_with_data(self, temp_dir):
        """Test skills visualization with generated skills."""
        skills_dir = temp_dir / "skills"
        skills_dir.mkdir()
        
        # Create a mock skill
        skill = {
            "name": "test_skill",
            "domain": "physics",
            "procedural_steps": ["step1", "step2", "step3"],
        }
        
        with open(skills_dir / "test_skill.json", "w") as f:
            json.dump(skill, f)
        
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "visualize", "skills",
             "--skills-dir", str(skills_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "test_skill" in result.stdout
        assert "physics" in result.stdout
    
    def test_list_skills_empty(self, temp_dir):
        """Test list-skills with no skills."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "list-skills",
             "--skills-dir", str(temp_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "No skills" in result.stdout
    
    def test_list_skills_with_data(self, temp_dir):
        """Test list-skills with generated skills."""
        skills_dir = temp_dir / "skills"
        skills_dir.mkdir()
        
        # Create mock skills
        for i in range(3):
            skill = {
                "name": f"skill_{i}",
                "domain": "physics",
            }
            with open(skills_dir / f"skill_{i}.json", "w") as f:
                json.dump(skill, f)
        
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "list-skills",
             "--skills-dir", str(skills_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "Found 3 skills" in result.stdout
        assert "skill_0" in result.stdout
        assert "skill_1" in result.stdout
        assert "skill_2" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI."""
    
    def test_discover_mock(self):
        """Test discover command with mock data."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "discover", "--verbose"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Discovery" in result.stdout
    
    def test_benchmark_compare(self):
        """Test benchmark with comparison."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark", 
             "--mock", "--compare", "--max-complexity", "2"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Expert accuracy" in result.stdout
        assert "Apprentice accuracy" in result.stdout
        assert "Improvement ratio" in result.stdout