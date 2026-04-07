"""Deep Research scenario — multi-step research on GPU shortage impact.

Goal: "Analyze the impact of 2024 GPU shortage on LLM training techniques"

Decomposed into a realistic DAG of parallel and sequential research
subtasks that exercise the full PAPoG pipeline: parallel fan-out for
independent research threads, convergence at synthesis nodes, and a
final report compilation step.

Graph structure
~~~~~~~~~~~~~~~

::

    gather_gpu_supply_data ──────────────┐
    gather_llm_training_trends ──────────┤
                                         ├─► synthesize_findings ─► compile_report
    analyze_cost_impact ─────────────────┤
    survey_alternative_hardware ─────────┘
    review_efficient_training_methods ───┘

The first four tasks can run in parallel; ``synthesize_findings``
merges their results and ``compile_report`` produces the final output.
"""

from __future__ import annotations

from src.models import TaskGraph, TaskNode


SCENARIO_GOAL = (
    "Analyze the impact of 2024 GPU shortage on LLM training techniques"
)


def build_graph() -> TaskGraph:
    """Construct the deep-research TaskGraph.

    Returns
    -------
    TaskGraph
        A validated DAG with six nodes arranged in a fan-out / fan-in
        pattern.
    """
    graph = TaskGraph()

    # Phase 1 — parallel research tasks (no dependencies between them)
    graph.add_node(TaskNode(
        id="gather_gpu_supply_data",
        description="Search for and gather data on 2024 GPU supply chain disruptions, availability, and pricing trends",
        dependencies=[],
        metadata={"phase": "research", "tools": ["search"]},
    ))

    graph.add_node(TaskNode(
        id="gather_llm_training_trends",
        description="Search for recent trends in LLM training techniques, focusing on changes during 2024",
        dependencies=[],
        metadata={"phase": "research", "tools": ["search"]},
    ))

    graph.add_node(TaskNode(
        id="analyze_cost_impact",
        description="Compute and analyze the cost impact of GPU shortages on LLM training budgets and timelines",
        dependencies=[],
        metadata={"phase": "research", "tools": ["python_repl", "search"]},
    ))

    graph.add_node(TaskNode(
        id="survey_alternative_hardware",
        description="Search for alternative hardware accelerators (TPUs, custom ASICs, cloud spot instances) adopted due to GPU scarcity",
        dependencies=[],
        metadata={"phase": "research", "tools": ["search"]},
    ))

    graph.add_node(TaskNode(
        id="review_efficient_training_methods",
        description="Search for and catalog efficient training methods (LoRA, QLoRA, distillation, pruning) that gained adoption during the shortage",
        dependencies=[],
        metadata={"phase": "research", "tools": ["search", "memory"]},
    ))

    # Phase 2 — synthesis (depends on all research tasks)
    graph.add_node(TaskNode(
        id="synthesize_findings",
        description="Store and synthesize findings from all research threads into a coherent analysis with key insights",
        dependencies=[
            "gather_gpu_supply_data",
            "gather_llm_training_trends",
            "analyze_cost_impact",
            "survey_alternative_hardware",
            "review_efficient_training_methods",
        ],
        metadata={"phase": "synthesis", "tools": ["memory"]},
    ))

    # Phase 3 — final report
    graph.add_node(TaskNode(
        id="compile_report",
        description="Compile the final research report with executive summary, methodology, findings, and recommendations",
        dependencies=["synthesize_findings"],
        metadata={"phase": "reporting", "tools": ["memory"]},
    ))

    graph.validate_dag()
    return graph
