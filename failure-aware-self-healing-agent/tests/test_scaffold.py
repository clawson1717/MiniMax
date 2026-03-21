"""Tests for the Scaffold class."""

import pytest
from src import __version__
from src.scaffold import Scaffold


class TestScaffold:
    """Tests for Scaffold class."""

    def test_init_stores_name(self):
        """Init should store the provided name."""
        s = Scaffold("test-project")
        assert s.name == "test-project"

    def test_get_info_returns_expected_dict(self):
        """get_info should return the correct dict structure."""
        s = Scaffold("my-agent")
        info = s.get_info()
        assert info == {
            "name": "my-agent",
            "version": "0.1.0",
            "status": "scaffolded",
        }

    def test_version_is_correct(self):
        """__version__ should be '0.1.0'."""
        assert __version__ == "0.1.0"
