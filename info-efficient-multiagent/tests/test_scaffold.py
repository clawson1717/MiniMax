"""Tests for project scaffold."""

import pytest
from pathlib import Path


class TestProjectScaffold:
    """Tests to verify project structure is correct."""

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_dir = Path(__file__).parent.parent / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        tests_dir = Path(__file__).parent.parent / "tests"
        assert tests_dir.exists()
        assert tests_dir.is_dir()

    def test_data_directory_exists(self):
        """Test that data directory exists."""
        data_dir = Path(__file__).parent.parent / "data"
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists()

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists()

    def test_src_package_init(self):
        """Test that src is a valid Python package."""
        import src
        assert hasattr(src, "__version__")

    def test_numpy_available(self):
        """Test that numpy is available."""
        import numpy as np
        assert np is not None


def test_basic_import():
    """Test that the package can be imported."""
    import src
    assert src.__version__ == "0.1.0"
