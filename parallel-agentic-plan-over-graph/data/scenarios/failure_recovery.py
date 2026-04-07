"""Failure-recovery scenario — exercises re-planning under failure.

A small research graph where one critical node is designed to fail,
forcing the DynamicRePlanner to generate a replacement sub-graph so
downstream work can continue.

Graph structure
~~~~~~~~~~~~~~~

::

    gather_data ─► *process_data_FAIL* ─► generate_report
                                           ▲
    validate_sources ──────────────────────┘

``process_data_FAIL`` contains the word "fail" in its description so
``MockExecutionStrategy`` raises ``RuntimeError`` when executing it.
The replanner should produce an alternative sub-graph and rewire
``generate_report``'s dependency.
"""

from __future__ import annotations

from src.models import TaskGraph, TaskNode


SCENARIO_GOAL = (
    "Process data with intentional failure to exercise re-planning"
)


def build_graph() -> TaskGraph:
    """Construct the failure-recovery TaskGraph.

    Returns
    -------
    TaskGraph
        A validated DAG with one node guaranteed to fail.
    """
    graph = TaskGraph()

    graph.add_node(TaskNode(
        id="gather_data",
        description="Search for raw data on the research topic",
        dependencies=[],
        metadata={"phase": "research", "tools": ["search"]},
    ))

    graph.add_node(TaskNode(
        id="validate_sources",
        description="Query and validate data sources for reliability",
        dependencies=[],
        metadata={"phase": "research", "tools": ["search"]},
    ))

    # This node will FAIL because description contains "fail"
    graph.add_node(TaskNode(
        id="process_data",
        description="Process and transform the gathered data — this task will fail on purpose",
        dependencies=["gather_data"],
        metadata={"phase": "processing", "should_fail": True},
    ))

    graph.add_node(TaskNode(
        id="generate_report",
        description="Generate final report from processed data and validated sources",
        dependencies=["process_data", "validate_sources"],
        metadata={"phase": "reporting", "tools": ["memory"]},
    ))

    graph.validate_dag()
    return graph
