"""Sanity tests for the PAPoG scaffold."""

import src
from src import main


def test_package_imports():
    assert src is not None


def test_version_is_set():
    assert hasattr(src, "__version__")
    assert isinstance(src.__version__, str)
    assert src.__version__ == "0.1.0"


def test_main_has_entrypoint():
    assert hasattr(main, "main")
    assert callable(main.main)


def test_project_name_constant():
    assert "PAPoG" in main.PROJECT_NAME
