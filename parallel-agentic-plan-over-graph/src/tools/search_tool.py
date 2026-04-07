"""SearchTool - Simulated web search for PAPoG.

This is a skeleton implementation that returns mock search results.
In a production system, this would integrate with actual search APIs.
"""

from __future__ import annotations

from src.tools import ToolABC


class SearchTool(ToolABC):
    """Simulated web search tool.

    Returns mock search results for any query. This skeleton provides
    the proper interface for future integration with real search APIs
    (e.g., Brave Search, Google Custom Search, etc.).

    Parameters
    ----------
    name:
        Tool identifier. Defaults to "search".
    description:
        Human-readable description. Defaults to a web search description.
    keywords:
        Keywords for task matching. Defaults to ["search", "find", "lookup", "query"].
    """

    def __init__(
        self,
        name: str = "search",
        description: str = "Search the web for information on a given topic",
        keywords: list[str] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._keywords = keywords if keywords is not None else ["search", "find", "lookup", "query", "web"]

    @property
    def name(self) -> str:
        """Unique identifier for this tool."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        return self._description

    @property
    def keywords(self) -> list[str]:
        """Keywords used to match this tool to task descriptions."""
        return list(self._keywords)

    def execute(self, input: str) -> str:
        """Execute a simulated web search.

        Parameters
        ----------
        input:
            The search query string.

        Returns
        -------
        str
            Formatted mock search results.
        """
        if not input or not input.strip():
            return "Error: Empty search query. Please provide a search term."

        query = input.strip()

        # Return mock search results
        # In production, this would call a real search API
        results = self._generate_mock_results(query)
        return results

    def _generate_mock_results(self, query: str) -> str:
        """Generate mock search results for testing.

        Parameters
        ----------
        query:
            The search query.

        Returns
        -------
        str
            Formatted mock results.
        """
        # Simulate 3 mock results
        return f"""Search Results for: "{query}"

1. {query.title()} - Wikipedia
   https://en.wikipedia.org/wiki/{query.replace(' ', '_')}
   Comprehensive article about {query} with detailed information and references.

2. {query.title()} Guide - Official Documentation
   https://docs.example.com/{query.replace(' ', '-')}
   Official documentation and guides for {query}.

3. Understanding {query.title()} - Tutorial
   https://tutorial.example.com/{query.replace(' ', '-')}
   Step-by-step tutorial explaining {query} concepts.

[Mock search results - integrate with real search API for production]"""

    def can_handle(self, task_description: str) -> bool:
        """Check if this tool can handle the given task.

        Parameters
        ----------
        task_description:
            The description of the task to check.

        Returns
        -------
        bool
            True if any keyword appears in the task description.
        """
        task_lower = task_description.lower()
        return any(kw.lower() in task_lower for kw in self._keywords)
