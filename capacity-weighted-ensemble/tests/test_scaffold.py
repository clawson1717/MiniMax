"""Basic scaffold tests to verify project structure."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_version():
    """Test that we can import the version from src package."""
    from src import __version__
    assert __version__ is not None


def test_version_is_correct():
    """Test that the version matches expected value."""
    from src import __version__
    assert __version__ == "0.1.0"
