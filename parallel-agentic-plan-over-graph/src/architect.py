"""Architect Agent — decomposes high-level goals into executable TaskGraphs.

Uses the Strategy pattern: callers can inject any ``DecompositionStrategy``
to control *how* a goal string gets broken into ``TaskNode`` objects.
Ships with ``MockLLMStrategy`` (deterministic, test-friendly) and
``KeywordDecompositionStrategy`` (keyword-based heuristic).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from src.models import TaskGraph, TaskNode


# ---------------------------------------------------------------------------
# Strategy interfaces
# ---------------------------------------------------------------------------

@runtime_checkable
class DecompositionStrategy(Protocol):
    """Protocol that any decomposition backend must satisfy."""

    def decompose(self, goal: str) -> TaskGraph:
        """Break *goal* into a populated ``TaskGraph``."""
        ...


class DecompositionStrategyABC(ABC):
    """Abstract base class alternative for strategies that prefer ABC."""

    @abstractmethod
    def decompose(self, goal: str) -> TaskGraph:
        """Break *goal* into a populated ``TaskGraph``."""


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

class MockLLMStrategy(DecompositionStrategyABC):
    """Deterministic mock that simulates an LLM decomposition.

    Always produces four tasks with a diamond dependency shape::

        analyse ──► design ──► implement ──► test
                  └──────────────────────────► test
    """

    def decompose(self, goal: str) -> TaskGraph:
        if not goal or not goal.strip():
            raise ValueError("Goal must be a non-empty string")

        graph = TaskGraph()
        nodes = [
            TaskNode(
                id="analyse",
                description=f"Analyse requirements for: {goal}",
                dependencies=[],
                metadata={"phase": "planning"},
            ),
            TaskNode(
                id="design",
                description=f"Design solution for: {goal}",
                dependencies=["analyse"],
                metadata={"phase": "planning"},
            ),
            TaskNode(
                id="implement",
                description=f"Implement solution for: {goal}",
                dependencies=["design"],
                metadata={"phase": "execution"},
            ),
            TaskNode(
                id="test",
                description=f"Test solution for: {goal}",
                dependencies=["design", "implement"],
                metadata={"phase": "verification"},
            ),
        ]
        for node in nodes:
            graph.add_node(node)

        graph.validate_dag()
        return graph


class KeywordDecompositionStrategy(DecompositionStrategyABC):
    """Heuristic strategy that creates tasks from keyword phrases.

    Splits the goal on common delimiters (commas, "and", "then") and
    chains the resulting tasks sequentially.  Useful for simple
    comma-separated goal lists.
    """

    DELIMITERS = [",", " and ", " then ", ";"]

    def decompose(self, goal: str) -> TaskGraph:
        if not goal or not goal.strip():
            raise ValueError("Goal must be a non-empty string")

        # Normalise delimiters to commas, then split
        normalised = goal
        for delim in self.DELIMITERS:
            normalised = normalised.replace(delim, ",")

        parts = [p.strip() for p in normalised.split(",") if p.strip()]

        if not parts:
            raise ValueError("Goal must contain at least one actionable phrase")

        graph = TaskGraph()
        prev_id: str | None = None
        for idx, part in enumerate(parts):
            node_id = f"task_{idx}"
            deps = [prev_id] if prev_id else []
            graph.add_node(
                TaskNode(
                    id=node_id,
                    description=part,
                    dependencies=deps,
                )
            )
            prev_id = node_id

        graph.validate_dag()
        return graph


# ---------------------------------------------------------------------------
# Architect agent
# ---------------------------------------------------------------------------

class Architect:
    """Decomposes a high-level goal into a validated ``TaskGraph``.

    Parameters
    ----------
    strategy:
        Any object satisfying ``DecompositionStrategy``.  Defaults to
        ``MockLLMStrategy`` when omitted.
    """

    def __init__(self, strategy: DecompositionStrategy | None = None) -> None:
        self._strategy: DecompositionStrategy = strategy or MockLLMStrategy()

    @property
    def strategy(self) -> DecompositionStrategy:
        """The active decomposition strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: DecompositionStrategy) -> None:
        self._strategy = value

    def decompose(self, goal: str) -> TaskGraph:
        """Decompose *goal* into a valid DAG of tasks.

        Delegates to the configured strategy, then validates the result.

        Raises
        ------
        ValueError
            If the goal is empty or the resulting graph is not a DAG.
        TypeError
            If the strategy returns something other than a ``TaskGraph``.
        """
        if not goal or not goal.strip():
            raise ValueError("Goal must be a non-empty string")

        result = self._strategy.decompose(goal)

        if not isinstance(result, TaskGraph):
            raise TypeError(
                f"Strategy must return a TaskGraph, got {type(result).__name__}"
            )

        # Belt-and-suspenders: validate even if the strategy already did
        result.validate_dag()
        return result
