import sys
import os

# Add the root directory to the python path so 'src' is importable
# Current directory is tests/, so the parent is the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_version_import():
    from src import __version__
    assert __version__ == "0.1.0"
