"""Simple scaffold module for FASHA."""

from src import __version__


class Scaffold:
    """Basic scaffold class for the FASHA project."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_info(self) -> dict:
        """Return project information."""
        return {
            "name": self.name,
            "version": __version__,
            "status": "scaffolded",
        }
