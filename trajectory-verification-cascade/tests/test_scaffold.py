"""Tests for project scaffold."""

import pytest
from pathlib import Path


class TestProjectScaffold:
    """Tests for project structure."""

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
        """Test that src/__init__.py exists and has version."""
        init_file = Path(__file__).parent.parent / "src" / "__init__.py"
        assert init_file.exists()

        content = init_file.read_text()
        assert "__version__" in content

    def test_numpy_available(self):
        """Test that numpy is available."""
        import numpy as np
        assert np is not None

    def test_pytest_available(self):
        """Test that pytest is available."""
        assert pytest is not None


def test_basic_import():
    """Test that the package can be imported."""
    import sys
    src_path = str(Path(__file__).parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Import should not raise
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trajectory_verification_cascade",
        Path(__file__).parent.parent / "src" / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "__version__")
    assert module.__version__ == "0.1.0"
